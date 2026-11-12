"""检测后端接口与内置后端。

每个后端负责：加载模型 → 对一批窗口做推理 → 返回 **窗口局部像素坐标**的旋转框。
坐标逆变换（窗口局部 → 源像素）由 worker 统一处理，后端不做，避免每处各写一遍漂移。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class DetectorNotAvailable(RuntimeError):
    """后端在当前环境不可用；worker 应转成 MODEL_UNAVAILABLE 失败而不是崩。"""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass
class RawBox:
    """窗口局部像素坐标下的检测框（rbox 约定见 backend.geo.boxes）。"""

    cx: float
    cy: float
    w: float
    h: float
    angle: float
    score: float
    class_name: str = "building"


class DetectorBackend(Protocol):
    name: str

    def prepare(self) -> dict:
        """加载模型。返回可写进 report 的加载信息（路径、哈希、缺 key 等）。"""
        ...

    def predict_windows(self, windows: Sequence[dict]) -> list[list[RawBox]]:
        """一批窗口 → 每窗口的候选框（窗口局部像素）。"""
        ...

    def close(self) -> None:
        ...


# --------------------------------------------------------------- null 后端


class NullDetector:
    """永远返回空结果：用于验证「无检测 → 两个合法空层 + 成功」的路径。"""

    name = "null"

    def prepare(self) -> dict:
        return {"backend": "null", "note": "no detection performed; used for pipeline validation"}

    def predict_windows(self, windows: Sequence[dict]) -> list[list[RawBox]]:
        return [[] for _ in windows]

    def close(self) -> None:
        pass


# ------------------------------------------------------------- jsonl 后端


class JsonlDetector:
    """从已有检测结果读框，按窗口 ID 对齐。

    文件格式（每行一个窗口）::

        {"window_id": 0, "detections": [
            {"rbox_xywha_rad": [cx, cy, w, h, angle], "score": 0.9, "category": "building"}]}

    坐标必须是**窗口局部像素**。这个后端让没有 GPU 的开发者也能验证去重、坐标链、
    导出与插件，而不必假装跑过模型。
    """

    name = "jsonl"

    def __init__(self, path: str, score_threshold: float = 0.0):
        self.path = path
        self.score_threshold = float(score_threshold)
        self._data: dict[int, list[RawBox]] = {}

    def prepare(self) -> dict:
        if not os.path.exists(self.path):
            raise DetectorNotAvailable("DETECTIONS_MISSING", f"检测结果文件不存在：{self.path}")
        self._data.clear()
        with open(self.path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DetectorNotAvailable(
                        "DETECTIONS_MALFORMED", f"{self.path}:{lineno} 不是合法 JSON：{exc}")
                wid = rec.get("window_id")
                if not isinstance(wid, int):
                    raise DetectorNotAvailable(
                        "DETECTIONS_MALFORMED", f"{self.path}:{lineno} 缺少整数 window_id")
                boxes = []
                for d in rec.get("detections") or []:
                    if d.get("category", "building") != "building":
                        continue
                    score = float(d.get("score", 0.0))
                    if score < self.score_threshold:
                        continue
                    r = d.get("rbox_xywha_rad") or d.get("rbox")
                    if not r or len(r) != 5:
                        continue
                    boxes.append(RawBox(float(r[0]), float(r[1]), float(r[2]),
                                        float(r[3]), float(r[4]), score))
                self._data[wid] = boxes
        return {"backend": "jsonl", "path": self.path, "windows": len(self._data)}

    def predict_windows(self, windows: Sequence[dict]) -> list[list[RawBox]]:
        return [list(self._data.get(int(w["window_id"]), [])) for w in windows]

    def close(self) -> None:
        self._data.clear()


# ------------------------------------------------------------ 后端工厂


def build_detector(profile: str, options: dict | None = None) -> DetectorBackend:
    """按 profile 选择后端。未知 profile 明确报错，不静默退回。"""
    options = options or {}
    profile = (profile or "").strip()
    if profile in ("null", "none"):
        return NullDetector()
    if profile == "jsonl":
        path = options.get("detections_jsonl")
        if not path:
            raise DetectorNotAvailable("DETECTOR_CONFIG_INVALID",
                                       "detector_profile=jsonl 需要 detections_jsonl 路径")
        return JsonlDetector(path, score_threshold=0.0)
    if profile in ("strip_s_gzda", "strip_rcnn"):
        from .strip_rcnn_backend import StripRcnnDetector

        return StripRcnnDetector(options)
    if profile == "mmrotate_obb":
        # 任意 MMRotate 系的旋转框检测权重（用户自选 .pth + config）
        from .strip_rcnn_backend import StripRcnnDetector

        return StripRcnnDetector(options)
    if profile == "mmdet_hbb":
        # mmdetection 系的水平框检测权重（用户自选 .pth + config）
        from .hbb_backend import HbbDetector

        return HbbDetector(options)
    if profile == "rtdetr_hf":
        # HuggingFace Transformers 版 RT-DETR（目录形式：safetensors + config）
        from .rtdetr_backend import RtdetrDetector

        return RtdetrDetector(options)
    raise DetectorNotAvailable(
        "DETECTOR_PROFILE_UNKNOWN",
        f"未知 detector_profile：{profile!r}"
        "（可选 null / jsonl / strip_s_gzda / mmrotate_obb / mmdet_hbb / rtdetr_hf）")


def rgb_to_model_array(rgb: np.ndarray) -> np.ndarray:
    """(H,W,3) uint8 RGB → 模型输入张量：RGB 顺序、CHW、float32、ImageNet 归一化。

    Strip-R-CNN 配置里是 ``to_rgb=True`` + mean/std 归一化，我们在这一层统一转换，
    避免 PIL/OpenCV 的 RGB/BGR 顺序在调用链上被重复翻转（§6.2）。
    """
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"期望 (H,W,3) 的 RGB 数组，实际 {arr.shape}")
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))
