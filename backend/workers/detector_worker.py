#!/usr/bin/env python
"""检测 worker：独立进程、独立环境（vecrot-det）。

请求（JSON）::

    {"schema_version": "1.0",
     "job_id": "...",
     "profile": "strip_s_gzda",
     "options": {...},
     "windows": [{"window_id": 0, "image": "abs/path.tif(npy/png)",
                  "width": 1024, "height": 1024, "xoff": 0, "yoff": 0}]}

结果（JSONL，每窗口一行）::

    {"window_id": 0, "detections": [{"rbox_xywha_rad": [...], "score": 0.9}]}

坐标约定：**窗口局部像素中心**。源像素换算与地理换算由协调器统一完成，
这样检测环境不需要 GIS 依赖，也不会出现两套换算。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.contracts.events import EventEmitter, worker_log      # noqa: E402


def load_window_image(path: str):
    """读窗口影像：.npy（协调器直接给内存快照）或图片文件。"""
    import numpy as np

    if path.lower().endswith(".npy"):
        return np.load(path)
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True, help="请求 JSON 路径")
    ap.add_argument("--out", required=True, help="结果 JSONL 路径")
    args = ap.parse_args()

    with open(args.request, encoding="utf-8") as f:
        req = json.load(f)
    job_id = req.get("job_id", "unknown")
    emitter = EventEmitter(job_id)
    emitter.emit("ready", message="detector worker started", extra={
        "python": sys.executable,
        "pid": os.getpid(),
    })

    try:
        from backend.detector.backends import build_detector

        backend = build_detector(req.get("profile", "null"), req.get("options") or {})
        info = backend.prepare() or {}
        emitter.emit("ready", message="detector backend ready", extra={"backend": info})
    except Exception as exc:                       # noqa: BLE001
        worker_log(traceback.format_exc())
        emitter.emit("failed", message=f"{type(exc).__name__}: {exc}",
                         code=getattr(exc, "code", "DETECTOR_INIT_FAILED"))
        return 1

    windows = req.get("windows") or []
    total = len(windows)
    done = failed = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    try:
        with open(args.out, "w", encoding="utf-8") as out:
            for w in windows:
                wid = int(w["window_id"])
                try:
                    rgb = load_window_image(w["image"])
                    if rgb.ndim != 3 or rgb.shape[2] != 3:
                        raise ValueError(f"窗口影像形状异常：{rgb.shape}")
                    boxes = backend.predict_windows([{
                        "window_id": wid, "rgb": rgb,
                        "scale": w.get("scale", (1.0, 1.0)),
                    }])[0]
                    payload = {
                        "window_id": wid,
                        "status": "ok",
                        "detections": [{
                            "rbox_xywha_rad": [b.cx, b.cy, b.w, b.h, b.angle],
                            "score": b.score,
                            "category": b.class_name,
                        } for b in boxes],
                    }
                    done += 1
                except Exception as exc:            # noqa: BLE001 - 单窗口失败不拖垮整个作业
                    failed += 1
                    payload = {"window_id": wid, "status": "error",
                               "error": f"{type(exc).__name__}: {exc}",
                               "detections": []}
                    worker_log(f"window {wid} failed: {exc}")
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                out.flush()
                emitter.emit("progress", stage="DETECTING", done=done + failed,
                             total=total, message=f"windows {done + failed}/{total}",
                             extra={"failed": failed})
    finally:
        try:
            backend.close()
        except Exception:                          # noqa: BLE001
            pass

    # 场景分类统计（只有带 scene_head 的模型才非零）：
    # skipped 表示这些窗口被判为「无建筑」，模型内部直接返回空结果，
    # 省掉了 RPN/RoI/NMS 以及后续的轮廓生成。
    scene = dict(getattr(backend, "scene_stats", {}) or {})
    if scene.get("probs"):
        probs = scene.pop("probs")
        scene["skipped_probs_min"] = min(probs)
        scene["skipped_probs_max"] = max(probs)
    emitter.emit("stage_finished", stage="DETECTING", done=done, total=total,
                 message=f"done={done} failed={failed}"
                         + (f" scene_skipped={scene.get('skipped_windows', 0)}"
                            if scene.get("has_scene_head") else ""),
                 extra={"scene": scene})
    if failed and total and failed == total:
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(20)
