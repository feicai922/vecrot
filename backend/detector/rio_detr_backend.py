"""RiO-DETR / RiO-DETRv2 旋转框检测后端（YAML 配置 + checkpoint）。

## 这个模型与其它后端的关系

| 后端 | 加载方式 | 环境 |
|---|---|---|
| ``strip_s_gzda`` / ``mmrotate_obb`` | mmcv ``Config`` + ``build_detector`` | vecrot-det（mmcv-full + torch 2.0） |
| ``rtdetr_hf`` | HF ``from_pretrained`` | vecrot-vllm（transformers） |
| **``rio_detr``** | 自有 ``engine.core.YAMLConfig`` + ``deploy()`` | **vecrot-rio（torch 2.2 + timm）** |

RiO-DETR 出自 DEIM/D-FINE 血统（`engine/` 里是 RT-DETRv4 的骨架），
既不是 mmrotate 也不是 HF Transformers 格式，因此必须是独立后端。
一份后端同时支持 RiO-DETR 与 RiO-DETRv2：两者的推理接口一致
（``YAMLConfig(config, resume=ckpt)`` → ``model.deploy()`` → ``postprocessor.deploy()``）。

## 推理口径（与官方 ``tools/inference/torch_inf.py`` 逐条对齐）

1. 输入按**保比例缩放 + 右/下补 0** 到 ``postprocessor.input_shape``（本项目实测为 800×800）；
2. 像素值 **只做 /255**，不做 ImageNet 归一化（训练侧 transforms 就是 ``ConvertPILImage(scale=true)``）；
3. 后处理用**原始窗口尺寸**做 ``orig_target_sizes``，输出坐标直接回到窗口像素；
4. 输出是 ``(labels, boxes, scores)``，``boxes`` 形如 ``[B, Q, 5]`` =
   ``cx, cy, w, h, angle``，其中 **angle 已是弧度**（模型内部是 [-1,1] × π）；
5. NMS-free：最多 ``num_top_queries``（通常 300）个查询，按分数阈值筛即可。

所以这里的坐标可以直接交给 ``RawBox``，不需要像 mmrotate 那样解析按类分组的嵌套列表。

## 环境要求（Windows 侧新环境 ``vecrot-rio``）

``torch 2.2.2`` + ``torchvision 0.17.2`` + ``timm>=1.0.27`` + ``opencv-python-headless``
+ ``numba`` + ``pycocotools`` + ``PyYAML`` + ``numpy``。
注意 ``engine/__init__.py`` 在 import 阶段就会 ``import cv2`` 并加载 ``engine.data``，
所以 cv2/numba/pycocotools 是**必需**的，不是只有评测才用。

与 ``vecrot-det`` 的 torch 2.0.1 / mmcv-full 冲突，所以必须单独一套环境；
与 ``vecrot-vllm`` 的 torch 2.4.1 也不同（本模型在 2.2.2 上验证）。
"""
from __future__ import annotations

import hashlib
import os
from typing import Sequence

from .backends import DetectorNotAvailable, RawBox

#: 权重不匹配时允许忽略的 key 片段（训练专用结构，没有推理期权重）
_BENIGN_MISSING = ("loss", "assigner", "matcher", "sampler", "teacher")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_state_dict(path: str) -> tuple[dict, str]:
    """从 checkpoint 里取出纯 ``state_dict``。

    训练脚本（``engine/solver``）存的是 ``{"model":…, "ema": {"module":…}, …}``，
    推理必须优先用 EMA 权重 —— 那是训练时真正用于评测的模型。
    返回 ``(state_dict, 来源标记)``。
    """
    import torch

    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise DetectorNotAvailable(
            "DETECTOR_WEIGHTS_INCOMPATIBLE",
            f"权重 {path} 不是 state_dict 也不是训练 checkpoint 字典")
    if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]:
        return ckpt["ema"]["module"], "ema"
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"], "model"
    # 已经是裸 state_dict
    return ckpt, "raw"


class RioDetrDetector:
    """RiO-DETR / RiO-DETRv2 旋转框检测。"""

    name = "rio_detr"

    def __init__(self, options: dict):
        self.repo_dir = options.get("detector_repo") or ""
        self.config = options.get("detector_config") or ""
        self.weights = options.get("detector_weights") or ""
        self.device = options.get("device") or "cuda:0"
        #: 该模型输出的是「每类分数经过 sigmoid 的 top-k 查询」，
        #: 官方脚本默认 0.4（DETR 系分数普遍低于 RPN 系，0.05 会进一堆废框）
        self.score_thr = float(options.get("score_thr", 0.4))
        self.num_classes = options.get("num_classes")
        self._model = None
        self._post = None
        #: 由 postprocessor 决定（800×800），用于 report 与大图小目标提示
        self.input_size: tuple[int, int] | None = None

    # ---------------------------------------------------------- 前置检查

    def _check_prereqs(self) -> None:
        if not self.repo_dir or not os.path.isdir(self.repo_dir):
            raise DetectorNotAvailable(
                "DETECTOR_REPO_MISSING",
                "未配置 RiO-DETR 源码目录（model.json 的 repo 或 manifest）；"
                "期望位于 third_party/RiO-DETRv2")
        if not os.path.isdir(os.path.join(self.repo_dir, "engine", "core")):
            raise DetectorNotAvailable(
                "DETECTOR_SOURCE_INCOMPLETE",
                f"{self.repo_dir} 里没有 engine/core，这不是 RiO-DETR 的源码树")
        if not self.config or not os.path.isfile(self.config):
            raise DetectorNotAvailable(
                "DETECTOR_CONFIG_MISSING",
                f"RiO-DETR 配置不存在：{self.config!r}"
                "（需要 .yml；官方训练产物的 configs/ 下就是它）")
        if not self.weights or not os.path.isfile(self.weights):
            raise DetectorNotAvailable(
                "DETECTOR_WEIGHTS_MISSING", f"检测权重不存在：{self.weights!r}")
        try:
            import torch        # noqa: F401
            import torchvision  # noqa: F401 - engine 的 transforms 依赖它
            import yaml         # noqa: F401
        except ImportError as exc:
            raise DetectorNotAvailable(
                "DETECTOR_ENV_INCOMPLETE",
                f"当前解释器不是 RiO-DETR 环境（缺 {exc.name}）。"
                "需要 torch 2.2.2 + torchvision 0.17.2 + timm + opencv + numba "
                "+ pycocotools + PyYAML；见 docs/安装教程-自建环境.md 的 vecrot-rio。") from exc

    # ---------------------------------------------------------- 加载

    def _force_no_pretrained(self, cfg) -> list[str]:
        """把配置里的 ``pretrained: true`` 全部关掉。

        权重由 checkpoint 提供；保持 true 会让 ``HGNetv2`` 去 ``./pretrain/hgnetv2/``
        找骨干文件，部署机上通常没有，于是加载直接失败（官方推理脚本也这么做）。
        """
        touched = []
        for name, section in (cfg.yaml_cfg or {}).items():
            if isinstance(section, dict) and section.get("pretrained") is True:
                section["pretrained"] = False
                touched.append(str(name))
        return touched

    def prepare(self) -> dict:
        self._check_prereqs()
        import sys

        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)
        try:
            from engine.core import YAMLConfig
        except ImportError as exc:
            raise DetectorNotAvailable(
                "DETECTOR_ENV_INCOMPLETE",
                f"无法 import engine.core（{exc}）。若提示缺 cv2/numba/pycocotools，"
                "说明当前解释器没装齐 RiO-DETR 的运行依赖。") from exc

        cfg = YAMLConfig(self.config, resume=self.weights)
        disabled = self._force_no_pretrained(cfg)

        state, weight_source = _read_state_dict(self.weights)
        model = cfg.model                      # 触发构建
        result = model.load_state_dict(state, strict=False)
        missing = [k for k in (getattr(result, "missing_keys", None) or [])]
        unexpected = [k for k in (getattr(result, "unexpected_keys", None) or [])]
        serious = [k for k in missing
                   if not any(t in k.lower() for t in _BENIGN_MISSING)]
        if serious:
            raise DetectorNotAvailable(
                "DETECTOR_WEIGHTS_INCOMPATIBLE",
                f"权重有 {len(serious)} 个结构未匹配（示例 {serious[:3]}）；"
                "通常意味着 checkpoint 与这份 YAML 配置不是同一次训练产物")

        deployed = model.deploy().to(self.device).eval()
        post = cfg.postprocessor.deploy()
        shape = getattr(post, "input_shape", None)
        if not shape:
            raise DetectorNotAvailable(
                "DETECTOR_CONFIG_INCOMPLETE",
                "YAML 的 PostProcessorOBB.input_shape 为空；推理需要固定的输入尺寸"
                "（例如 input_shape: [800, 800]）")
        self.input_size = (int(shape[0]), int(shape[1]))
        n_cls = int(getattr(post, "num_classes", 0) or 0) or self.num_classes or 1
        self.num_classes = n_cls
        self._model = deployed
        self._post = post

        return {
            "backend": self.name,
            "framework": "RiO-DETR (engine.core.YAMLConfig + deploy)",
            "repo_dir": self.repo_dir,
            "config": self.config,
            "config_sha256": _sha256(self.config),
            "weights": self.weights,
            "weights_sha256": _sha256(self.weights),
            "weights_source": weight_source,
            "device": self.device,
            "num_classes": n_cls,
            "input_size": list(self.input_size),
            "box_type": "obb",
            "angle_unit": "rad",
            "pretrained_disabled": disabled,
            "weight_loading": {
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "serious_missing": serious,
                "matched_all_structural": not serious,
            },
            "note": "NMS-free：输出 top-300 查询，按分数阈值筛；"
                    f"输入固定 {self.input_size[0]}×{self.input_size[1]}，"
                    "窗口会被整体缩放，小目标会变小。",
            "verified_on_this_machine": False,
        }

    # ---------------------------------------------------------- 推理

    def predict_windows(self, windows: Sequence[dict]) -> list[list[RawBox]]:
        if self._model is None:
            raise DetectorNotAvailable("DETECTOR_NOT_PREPARED", "prepare() 未成功调用")
        return [self._predict_one(w) for w in windows]

    def _predict_one(self, w: dict) -> list[RawBox]:
        import numpy as np
        import torch
        from PIL import Image

        rgb = np.asarray(w["rgb"], dtype=np.uint8)
        h, w_ = rgb.shape[:2]
        target_h, target_w = self.input_size

        padded, _scale = _resize_and_pad(rgb, (target_h, target_w))
        # 只做 /255：训练侧是 ConvertPILImage(scale=true)，没有 mean/std
        tensor = torch.from_numpy(
            np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32) / 255.0
        ).unsqueeze(0).to(self.device)
        orig_size = torch.tensor([[float(h), float(w_)]], dtype=torch.float32,
                                 device=self.device)

        with torch.inference_mode():
            labels, boxes, scores = self._post(self._model(tensor), orig_size)

        boxes_np = boxes[0].float().cpu().numpy()
        scores_np = scores[0].float().cpu().numpy()
        out: list[RawBox] = []
        for row, score in zip(boxes_np, scores_np):
            s = float(score)
            if s < self.score_thr:
                continue
            cx, cy, bw, bh, ang = [float(v) for v in row[:5]]
            if not all(np.isfinite([cx, cy, bw, bh, ang])):
                continue
            if bw <= 0 or bh <= 0:
                continue
            out.append(RawBox(cx, cy, bw, bh, ang, s))
        return out

    def close(self) -> None:
        self._model = None
        self._post = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                          # noqa: BLE001
            pass


def _resize_and_pad(rgb, size: tuple[int, int]):
    """保比例缩放到 ``size``，右/下补黑边（与官方 torch_inf 完全一致）。

    返回 ``(padded_uint8_HWC, scale)``。补边只在右/下侧，因此把 padding 空间的
    坐标除以 ``scale`` 就回到原始像素 —— 这正是 postprocessor 里
    ``boxes * target_wh / scale`` 的假设，后端这里不需要再做偏移修正。
    """
    import numpy as np
    from PIL import Image

    target_h, target_w = size
    height, width = rgb.shape[:2]
    scale = min(target_w / width, target_h / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = np.asarray(
        Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.BILINEAR))
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return padded, scale
