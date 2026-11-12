#!/usr/bin/env python
"""轮廓生成 worker：独立进程、独立环境（vecrot-vllm）。

请求（JSON）::

    {"schema_version": "1.0", "job_id": "...",
     "profile": "vecrot", "options": {...},
     "tasks": [{"det_id": "d000001", "crop_image": "abs/crop.npy",
                "crop_meta": {"obj_box_in_crop": [...], "crop_box_in_crop": [...],
                              "transform": {...}}}]}

结果（JSONL，每 det 一行）::

    {"det_id": "d000001", "status": "ok|abstained|empty|error|skipped",
     "polygon_crop_px": [[u,v],...], "abstain_z": -3.2, "reason": "",
     "elapsed_s": 0.18}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.contracts.events import EventEmitter, worker_log      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.request, encoding="utf-8") as f:
        req = json.load(f)
    job_id = req.get("job_id", "unknown")
    emitter = EventEmitter(job_id)
    emitter.emit("ready", message="vector worker started", extra={
        "python": sys.executable, "pid": os.getpid()})

    try:
        from backend.vectorizer.backends import build_vectorizer

        backend = build_vectorizer(req.get("profile", "null"), req.get("options") or {})
        info = backend.prepare() or {}
        emitter.emit("ready", message="vector backend ready", extra={"backend": info})
    except Exception as exc:                       # noqa: BLE001
        worker_log(traceback.format_exc())
        emitter.emit("failed", message=f"{type(exc).__name__}: {exc}",
                         code=getattr(exc, "code", "VECTOR_INIT_FAILED"))
        return 1

    tasks_in = req.get("tasks") or []
    total = len(tasks_in)
    done = failed = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # 按批喂给后端，逐条落盘：中途被杀也能保留已完成部分
    batch = max(int((req.get("options") or {}).get("vector_batch", 4)), 1)
    try:
        with open(args.out, "w", encoding="utf-8") as out:
            for s in range(0, total, batch):
                chunk = tasks_in[s:s + batch]
                loaded = []
                for t in chunk:
                    import numpy as np

                    path = t.get("crop_image", "")
                    try:
                        arr = np.load(path) if path.lower().endswith(".npy") else _read_image(path)
                        loaded.append({"det_id": t["det_id"], "crop_rgb": arr,
                                       "crop_meta": t.get("crop_meta") or {}})
                    except Exception as exc:        # noqa: BLE001 - 单个坏 crop 不拖垮作业
                        failed += 1
                        done += 1
                        out.write(json.dumps({
                            "det_id": t.get("det_id"), "status": "error",
                            "reason": f"crop_load_failed:{type(exc).__name__}: {exc}"},
                            ensure_ascii=False) + "\n")
                        worker_log(f"crop load failed for {t.get('det_id')}: {exc}")
                if not loaded:
                    continue
                results = backend.predict(loaded)
                if len(results) != len(loaded):
                    raise RuntimeError(
                        f"后端返回 {len(results)} 条结果但有 {len(loaded)} 个输入")
                for task, r in zip(loaded, results):
                    done += 1
                    out.write(json.dumps({
                        "det_id": task["det_id"],
                        "status": r.status,
                        "polygon_crop_px": r.polygon_crop_px,
                        "abstain_z": r.abstain_z,
                        "reason": r.reason,
                        "elapsed_s": r.elapsed_s,
                        "extra": r.extra,
                    }, ensure_ascii=False) + "\n")
                out.flush()
                emitter.emit("progress", stage="VECTORIZING", done=done, total=total,
                             message=f"polygons {done}/{total}", extra={"failed": failed})
    finally:
        try:
            backend.close()
        except Exception:                          # noqa: BLE001
            pass

    emitter.emit("stage_finished", stage="VECTORIZING", done=done, total=total,
                 message=f"done={done} failed={failed}")
    return 0 if failed < max(total, 1) else 1


def _read_image(path: str):
    import numpy as np
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(20)
