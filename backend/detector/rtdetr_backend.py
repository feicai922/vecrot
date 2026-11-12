"""RT-DETR 水平框检测后端（HuggingFace Transformers 版）。

## 与 mmdet 系后端的区别

`vecrot/models/detector_rt-detr_bbox` 是 **HF Transformers 格式**：
``model.safetensors`` + ``config.json`` + ``preprocessor_config.json``，
模型类是 ``RTDetrForObjectDetection``。不能用 ``mmdet.build_detector`` 加载。

因此这是独立后端，依赖：

* ``transformers``（含 ``RTDetrForObjectDetection`` / ``RTDetrImageProcessor``）
* ``torch``

本机实测：``deployment/envs/vector`` 有 transformers 4.57.1 + torch 2.4.1，
RTDetr API 可用，因此**不需要单独建环境**，用 vector 环境即可。

## 该模型的固有约束（来自 INFERENCE_GUIDE.md）

| 项 | 值 | 影响 |
|---|---|---|
| 输入尺寸 | **640×640 固定** | 大瓦片会被缩放，小目标进一步变小 |
| 类别数 | **1**（`LABEL_0` = 屋顶） | 无需 num_classes 参数 |
| 输出 | 只有框，**无掩膜** | 轮廓仍由 VectorLLM 生成 |
| AP_small | 0.248（弱） | 高分辨率影像上漏检偏多 |

预处理由 ``RTDetrImageProcessor`` 负责（resize 640 + ImageNet 归一化），
所以**不能**复用 ``rgb_to_model_array()``（那是 mmdet 的 BGR→RGB + 自定义 mean/std）。
"""
from __future__ import annotations

import hashlib
import os
from typing import Sequence

from .backends import DetectorNotAvailable, RawBox


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _dir_fingerprint(path: str) -> str:
    """权重目录指纹：主权重 + config 的联合哈希（大文件用各取头尾的方式不必要，
    这里直接对 safetensors 全量哈希一次，306MB 约 0.5s，可接受）。"""
    parts = []
    for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
        p = os.path.join(path, name)
        if os.path.isfile(p):
            parts.append(f"{name}:{_sha256(p)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


class RtdetrDetector:
    """RT-DETR 水平框检测（HF Transformers）。"""

    name = "rtdetr_hf"

    def __init__(self, options: dict):
        #: 模型目录（含 model.safetensors / config.json / preprocessor_config.json）
        self.model_dir = options.get("detector_weights") or ""
        self.device = options.get("device") or "cuda:0"
        self.score_thr = float(options.get("score_thr", 0.3))
        #: 模型固定 640×640；保留字段用于 report 追溯
        self.input_size = 640
        self._model = None
        self._processor = None

    # ---------------------------------------------------------- 前置

    def _check_prereqs(self) -> None:
        if not self.model_dir or not os.path.isdir(self.model_dir):
            raise DetectorNotAvailable(
                "DETECTOR_WEIGHTS_MISSING",
                f"RT-DETR 模型目录不存在：{self.model_dir!r}")
        need = ("config.json", "preprocessor_config.json")
        miss = [n for n in need if not os.path.isfile(os.path.join(self.model_dir, n))]
        if miss:
            raise DetectorNotAvailable(
                "DETECTOR_CONFIG_MISSING",
                f"RT-DETR 目录缺少 {miss}；期望 config.json + preprocessor_config.json"
                "（HF Transformers 格式，不是 mmdet 的 .py 配置）")
        has_weights = any(
            os.path.isfile(os.path.join(self.model_dir, n))
            for n in ("model.safetensors", "pytorch_model.bin"))
        if not has_weights:
            raise DetectorNotAvailable(
                "DETECTOR_WEIGHTS_MISSING",
                f"RT-DETR 目录缺少权重（model.safetensors 或 pytorch_model.bin）："
                f"{self.model_dir}")
        try:
            import torch        # noqa: F401
            from transformers import (RTDetrForObjectDetection,   # noqa: F401
                                      RTDetrImageProcessor)
        except ImportError as exc:
            raise DetectorNotAvailable(
                "DETECTOR_ENV_INCOMPLETE",
                f"当前解释器不支持 RT-DETR（缺 {exc.name} 或无 RTDetr API）。"
                "本机可用 deployment/envs/vector 的 python.exe"
                "（transformers 4.57.1 + torch 2.4.1，实测 RTDetr API 可用）。") from exc

    # ---------------------------------------------------------- 加载

    def prepare(self) -> dict:
        self._check_prereqs()
        import torch
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        processor = RTDetrImageProcessor.from_pretrained(self.model_dir)
        model = RTDetrForObjectDetection.from_pretrained(self.model_dir)
        model.to(self.device)
        model.eval()
        self._model = model
        self._processor = processor

        cfg = getattr(model, "config", None)
        id2label = getattr(cfg, "id2label", {}) or {}
        num_classes = len(id2label) or 1
        # 固定输入尺寸来自 preprocessor_config
        size = getattr(processor, "size", None) or {}
        self.input_size = int(size.get("height") or 640)

        return {
            "backend": self.name,
            "weights": self.model_dir,
            "weights_fingerprint": _dir_fingerprint(self.model_dir),
            "device": self.device,
            "num_classes": num_classes,
            "id2label": {str(k): str(v) for k, v in id2label.items()},
            "input_size": self.input_size,
            "box_type": "hbb",
            "framework": "transformers.RTDetrForObjectDetection",
            "note": "RT-DETR 固定 640×640 输入；小目标 AP_small≈0.25，"
                    "高分辨率影像上漏检偏多。",
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
        image = Image.fromarray(rgb)

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = self._model(**inputs)
        # target_sizes 用**原始窗口尺寸**，post_process 负责把 640 空间的框映射回来
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=[(h, w_)], threshold=self.score_thr)

        out: list[RawBox] = []
        if not results:
            return out
        res = results[0]
        boxes = res["boxes"].float().cpu().numpy()
        scores = res["scores"].float().cpu().numpy()
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            s = float(score)
            if s < self.score_thr:
                continue
            bw, bh = abs(float(x2) - float(x1)), abs(float(y2) - float(y1))
            if bw <= 0 or bh <= 0:
                continue
            # HBB → 中心 + 宽高，角度 0（不为了 w≥h 交换边，§7.1）
            out.append(RawBox((float(x1) + float(x2)) / 2,
                              (float(y1) + float(y2)) / 2,
                              bw, bh, 0.0, s))
        return out

    def close(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                          # noqa: BLE001
            pass
