"""水平框（HBB）检测后端：任意 mmdetection 2.x 权重。

面向"用户自选本地模型"的场景：给一个 `.pth` 权重和一个 `.py` 配置，
就按 mmdet 的推理流程跑，输出水平框。

## 与旋转框（OBB）的关系

开发文档 §1.1 明确要求：HBB 通过**独立适配器**扩展，不能靠"把角度置 0"来假装支持。
原因是：

* HBB 模型的框是 `(x1, y1, x2, y2)`，不存在 w/h 交换与角度归一化问题；
* 若强行转成 OBB，`w ≥ h` 的约定会引入一次不必要的边交换，
  下游的旋转 IoU 去重与转正裁剪都会受影响（§7.1）。

所以这里直接输出 ``RawBox(cx, cy, w, h, angle=0)``，并在 report 里记
``box_type: "hbb"``。

## 用户需要提供什么

* ``detector_weights``：``.pth`` / ``.ckpt`` 权重
* ``detector_config``：mmdet 的 ``.py`` 配置（用于 build_detector 与 test_cfg）
* ``num_classes``：可选，默认从 config 的 ``model.roi_head.bbox_head`` /
  ``model.bbox_head`` 读取；读不到就要求显式给

缺任何一项都明确报错，不猜。
"""
from __future__ import annotations

import hashlib
import os
from typing import Sequence

from .backends import DetectorNotAvailable, RawBox

_TRAIN_ONLY_KEYS = ("optimizer", "lr_config", "runner", "custom_hooks",
                    "evaluation", "checkpoint_config", "log_config", "workflow",
                    "fp16", "gpu_number")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class HbbDetector:
    """mmdet 2.x 水平框检测。"""

    name = "mmdet_hbb"

    def __init__(self, options: dict):
        self.repo_dir = options.get("mmdet_repo") or ""
        self.weights = options.get("detector_weights") or ""
        self.config = options.get("detector_config") or ""
        self.device = options.get("device") or "cuda:0"
        self.score_thr = float(options.get("score_thr", 0.05))
        self.nms_iou = float(options.get("nms_iou", 0.5))
        self.max_per_img = int(options.get("max_per_img", 2000))
        self.num_classes = options.get("num_classes")
        self.derived_dir = options.get("derived_dir") or ""
        self._model = None

    # ---------------------------------------------------------- 前置

    def _check_prereqs(self) -> None:
        if not self.weights or not os.path.isfile(self.weights):
            raise DetectorNotAvailable("DETECTOR_WEIGHTS_MISSING",
                                       f"HBB 权重不存在：{self.weights!r}")
        if not self.config or not os.path.isfile(self.config):
            raise DetectorNotAvailable(
                "DETECTOR_CONFIG_MISSING",
                f"HBB 配置不存在：{self.config!r}；mmdet 需要 .py 配置才能 build 模型")
        try:
            import torch        # noqa: F401
            import mmcv         # noqa: F401
            import mmdet        # noqa: F401
        except ImportError as exc:
            raise DetectorNotAvailable(
                "DETECTOR_ENV_INCOMPLETE",
                f"当前解释器不是检测环境（缺 {exc.name}）；"
                "请用 deployment manifest 里 environments.detector 的 python.exe") from exc

        import mmdet

        major = int(str(mmdet.__version__).split(".")[0])
        if major >= 3:
            raise DetectorNotAvailable(
                "DETECTOR_MMDET_INCOMPATIBLE",
                f"检测到 mmdet {mmdet.__version__}；本适配器面向 mmdet 2.x。"
                "mmdet 3.x 的 API（基于 BaseDetector + data_preprocessor）不同，"
                "需另写适配器，不能混用。")

    def _derive_config(self):
        """从用户配置派生：展开 base、剔训练字段、套用推理阈值。"""
        from mmcv import Config, ConfigDict

        if self.repo_dir and self.repo_dir not in __import__("sys").path:
            __import__("sys").path.insert(0, self.repo_dir)

        try:
            cfg = Config.fromfile(self.config)
        except ImportError as exc:
            # 典型原因：配置里的 custom_imports 需要某个仓库根在 sys.path 上
            # （例如 mmrotate 的 configs.gzda.gzda_dataset），但用户选的
            # 是 mmdet 配置、没设 mmdet_repo / strip_rcnn_repo。
            raise DetectorNotAvailable(
                "DETECTOR_CONFIG_UNIMPORTABLE",
                "无法加载检测配置 {}（自定义模块导入失败：{}）。常见原因：配置里的"
                "custom_imports 需要对应仓库根在 sys.path 上。如果你用的是 mmrotate 配置，请改用"
                "「旋转框 - 其他 MMRotate 权重」；如果用的是 mmdet 配置，"
                "请在 manifest 的 models.detector 里设 mmdet_repo 指向 mmdet 仓库。"
                .format(self.config, exc)) from exc
        notes = {
            "source_config": self.config,
            "source_config_sha256": _sha256(self.config),
            "removed_keys": [],
            "overrides": {},
        }
        for key in _TRAIN_ONLY_KEYS:
            if key in cfg:
                cfg.pop(key)
                notes["removed_keys"].append(key)

        data = cfg.get("data") or {}
        for key in ("train", "val", "samples_per_gpu", "workers_per_gpu"):
            if key in data:
                data.pop(key)
                notes["removed_keys"].append(f"data.{key}")
        if "test" in data:
            data["test"] = {k: v for k, v in data["test"].items()
                            if k not in ("ann_file", "img_prefix")}
        cfg.data = data

        # 类别数：优先用调用方给的，其次从配置读
        n = self.num_classes
        if n is None:
            for path in (("roi_head", "bbox_head", "num_classes"),
                         ("bbox_head", "num_classes")):
                node = cfg.model
                ok = True
                for k in path:
                    if hasattr(node, k):
                        node = node[k]
                    else:
                        ok = False
                        break
                if ok and isinstance(node, int):
                    n = node
                    break
        if n is None:
            raise DetectorNotAvailable(
                "DETECTOR_CONFIG_INCOMPLETE",
                "无法从配置读出 num_classes；请在模型设置里显式填写类别数")
        self.num_classes = int(n)

        # 推理阈值：统一写到 model.test_cfg（mmdet 2.x 的 simple_test 读这里）
        rcnn = dict(nms_pre=self.max_per_img, min_bbox_size=0,
                    score_thr=self.score_thr, nms=dict(type="nms", iou_threshold=self.nms_iou),
                    max_per_img=self.max_per_img)
        if "test_cfg" not in cfg.model or cfg.model.get("test_cfg") is None:
            cfg.model.test_cfg = ConfigDict(dict(rcnn=rcnn))
        else:
            cfg.model.test_cfg = ConfigDict(cfg.model.test_cfg)
            if "rcnn" in cfg.model.test_cfg:
                cfg.model.test_cfg["rcnn"] = ConfigDict(rcnn)
            else:
                cfg.model.test_cfg["rcnn"] = ConfigDict(rcnn)
        notes["overrides"] = {
            "num_classes": self.num_classes,
            "score_thr": self.score_thr,
            "nms_iou": self.nms_iou,
            "max_per_img": self.max_per_img,
        }

        derived = ""
        if self.derived_dir:
            os.makedirs(self.derived_dir, exist_ok=True)
            derived = os.path.join(self.derived_dir, "hbb_infer.py")
            cfg.dump(derived)
            notes["derived_config"] = derived
            notes["derived_config_sha256"] = _sha256(derived)
        return cfg, notes

    # ---------------------------------------------------------- 加载

    def prepare(self) -> dict:
        self._check_prereqs()
        from mmcv.runner import load_checkpoint
        from mmdet.models import build_detector

        cfg, notes = self._derive_config()
        model = build_detector(cfg.model)
        result = load_checkpoint(model, self.weights, map_location="cpu")
        missing = list(getattr(result, "missing_keys", None) or [])
        unexpected = list(getattr(result, "unexpected_keys", None) or [])
        benign = ("loss", "assigner", "sampler")
        serious = [k for k in missing if not any(t in k.lower() for t in benign)]
        if serious:
            raise DetectorNotAvailable(
                "DETECTOR_WEIGHTS_INCOMPATIBLE",
                f"HBB 权重有 {len(serious)} 个结构未匹配（示例 {serious[:3]}）；"
                "通常意味着权重与配置不是同一训练产物")
        model.to(self.device)
        model.eval()
        self._model = model
        self._cfg = cfg
        return {
            "backend": self.name,
            "box_type": "hbb",
            "weights": self.weights,
            "weights_sha256": _sha256(self.weights),
            "device": self.device,
            "num_classes": self.num_classes,
            "config_derivation": notes,
            "weight_loading": {
                "missing_keys": missing, "unexpected_keys": unexpected,
                "matched_all_structural": not serious,
            },
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

        from .backends import rgb_to_model_array

        rgb = w["rgb"]
        arr = rgb_to_model_array(rgb)
        h, w_ = arr.shape[1], arr.shape[2]
        img_meta = dict(img_shape=(h, w_, 3), ori_shape=(h, w_, 3), pad_shape=(h, w_, 3),
                        scale_factor=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                        flip=False, filename=str(w.get("window_id", "")))
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            result = self._model(return_loss=False, img=[tensor],
                                 img_metas=[[img_meta]], rescale=False)
        per_image = result[0] if isinstance(result, (list, tuple)) else result
        out: list[RawBox] = []
        for per_class in per_image:
            boxes = np.asarray(per_class)
            if boxes.size == 0:
                continue
            if boxes.ndim == 3:
                boxes = boxes.reshape(-1, boxes.shape[-1])
            if boxes.ndim != 2 or boxes.shape[1] < 5:
                continue
            for row in boxes:
                x1, y1, x2, y2, score = [float(v) for v in row[:5]]
                if score < self.score_thr:
                    continue
                bw, bh = abs(x2 - x1), abs(y2 - y1)
                if bw <= 0 or bh <= 0:
                    continue
                # HBB → 中心点 + 宽高，角度固定 0。不做 w≥h 边交换（§7.1）
                out.append(RawBox((x1 + x2) / 2, (y1 + y2) / 2, bw, bh, 0.0, score))
        return out

    def close(self) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                          # noqa: BLE001
            pass
