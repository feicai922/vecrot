"""演示：用合成影像 + jsonl 检测结果跑完整闭环，产出可直接在 QGIS 打开的双 GPKG。

这不是模型推理结果 —— 轮廓是 null 后端的矩形，仅用于验证坐标链与导出。
真实运行请见 docs/INSTALL.md。

用法：python deployment/demo.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "tests"))
sys.path.insert(0, REPO)

from helpers import (aoi_for_raster, make_raster, pixel_aoi,
                     write_detections_jsonl, write_manifest)      # noqa: E402


def main() -> int:
    demo_dir = os.path.join(REPO, "work", "demo")
    os.makedirs(demo_dir, exist_ok=True)
    raster = make_raster(os.path.join(demo_dir, "synthetic_scene.tif"),
                         width=1200, height=900)
    aoi, crs = aoi_for_raster(raster)

    # 三个窗口内各放几个框（窗口局部坐标，tile 1024 / overlap 256）
    dets = write_detections_jsonl(os.path.join(demo_dir, "detections.jsonl"), [
        {"window_id": 0, "boxes": [[120, 120, 60, 36, 0.0, 0.96],
                                   [300, 200, 44, 30, 0.4, 0.91],
                                   [520, 140, 80, 40, -0.25, 0.88]]},
        {"window_id": 1, "boxes": [[200, 300, 50, 32, 0.8, 0.93]]},
    ])
    manifest = write_manifest(demo_dir, detections_jsonl=dets,
                             detector_profile="jsonl", vector_profile="null")

    out_dir = os.path.join(demo_dir, "output")
    job = {
        "schema_version": "1.0",
        "job_id": "demo_synthetic",
        "input": {"raster_path": raster, "bands_rgb": [1, 2, 3],
                  "aoi_geojson": aoi, "aoi_crs": crs},
        "inference": {"tile_size": 1024, "tile_overlap": 256,
                      "score_threshold": 0.3, "vector_batch": 2},
        "output": {"directory": out_dir, "crs_policy": "source",
                   "aoi_policy": "intersect_keep_complete"},
    }
    job_path = os.path.join(demo_dir, "job.json")
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)

    print(f"影像：{raster}")
    print(f"作业：{job_path}")
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "backend.cli", "run",
         "--job", job_path, "--manifest", manifest,
         "--work", os.path.join(demo_dir, "work")],
        cwd=REPO, text=True, encoding="utf-8", errors="replace")
    proc.check_returncode()

    print("\n成果：")
    for name in ("boxes.gpkg", "buildings.gpkg", "report.json", "manifest.json"):
        path = os.path.join(out_dir, name)
        mark = "✓" if os.path.exists(path) else "✗"
        print(f"  {mark} {path}")

    with open(os.path.join(out_dir, "report.json"), encoding="utf-8") as f:
        report = json.load(f)
    print("\n计数：", json.dumps(report["counts"], ensure_ascii=False))
    print("\n在 QGIS 里打开 boxes.gpkg（图层 boxes）与 buildings.gpkg（图层 buildings）。")
    print("注意：轮廓是 null 后端的矩形，不是模型输出。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
