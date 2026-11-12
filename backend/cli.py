#!/usr/bin/env python
"""协调器命令行入口（QGIS 插件用 QProcess 调这里）。

用法::

    python -m backend.cli run --job job.json --manifest deployment/manifest.json \
        [--work work] [--cancel-file path]

契约：
* stdout 每行一个 JSON 事件（UTF-8、无缓冲）。
* 退出码：0 成功 / 10 部分成功 / 20 取消 / 其它失败。
* ``--cancel-file`` 出现即请求取消（跨进程取消信号，Windows 友好）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.contracts.events import EventEmitter             # noqa: E402
from backend.contracts.jobspec import JobSpecError, parse_job_spec   # noqa: E402


def _load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    # manifest 里的模型路径是相对**仓库根**写的（manifest 本身在 deployment/ 下）
    dep_dir = os.path.dirname(os.path.abspath(path))
    manifest["_root"] = os.path.dirname(dep_dir)
    manifest["_manifest_dir"] = dep_dir
    manifest["_repo_root"] = REPO_ROOT
    return manifest


def cmd_run(args: argparse.Namespace) -> int:
    with open(args.job, encoding="utf-8") as f:
        raw = json.load(f)

    job_id_hint = raw.get("job_id", "unknown") if isinstance(raw, dict) else "unknown"
    emitter = EventEmitter(job_id_hint)
    try:
        spec = parse_job_spec(raw)
    except JobSpecError as exc:
        emitter.emit("failed", stage="VALIDATING", message=str(exc),
                     code=exc.code, extra={"detail": exc.message})
        return 1

    manifest = _load_manifest(args.manifest)
    work_root = os.path.abspath(args.work or os.path.join(REPO_ROOT, "work"))

    from backend.coordinator.runner import Coordinator

    coord = Coordinator(spec, manifest, work_root, emitter=emitter)
    stop_watch = _start_cancel_watch(args.cancel_file, coord)
    try:
        return coord.run()
    finally:
        stop_watch.set()


def _start_cancel_watch(path: str | None, coord) -> threading.Event:
    stop = threading.Event()
    if not path:
        return stop

    def watch() -> None:
        while not stop.wait(0.2):
            if os.path.exists(path):
                coord.request_cancel()
                return

    threading.Thread(target=watch, daemon=True).start()
    return stop


def cmd_inspect(args: argparse.Namespace) -> int:
    """只做预检，不启动模型。UI 选完影像即可调用。"""
    sys.path.insert(0, REPO_ROOT)
    from backend.geo import raster as rgeo

    try:
        info = rgeo.inspect_raster(args.raster, tuple(args.bands))
    except rgeo.RasterError as exc:
        print(json.dumps({"ok": False, "code": exc.code, "message": exc.message},
                         ensure_ascii=False))
        return 1
    print(json.dumps({
        "ok": True,
        "width": info.width, "height": info.height, "count": info.count,
        "dtypes": list(info.dtypes), "crs": info.crs,
        "transform": list(info.transform),
        "bounds": list(info.bounds or []),
        "approx_resolution": list(info.approx_resolution),
        "rotated_affine": info.is_rotated,
        "warnings": info.warnings,
    }, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="vecrot-backend")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="执行作业")
    p_run.add_argument("--job", required=True)
    p_run.add_argument("--manifest", required=True)
    p_run.add_argument("--work", default=None)
    p_run.add_argument("--cancel-file", default=None)
    p_run.set_defaults(func=cmd_run)

    p_insp = sub.add_parser("inspect", help="栅格预检")
    p_insp.add_argument("--raster", required=True)
    p_insp.add_argument("--bands", type=int, nargs=3, default=[1, 2, 3])
    p_insp.set_defaults(func=cmd_inspect)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
