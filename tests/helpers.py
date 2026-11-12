"""测试工具：合成 GeoTIFF 与测试用 manifest。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def make_raster(path: str, width: int = 512, height: int = 512,
                crs: str = "EPSG:32648", gt: tuple | None = None,
                dst_dtype: str = "uint8") -> str:
    """写一张带仿射与 CRS 的合成 GeoTIFF（含几个亮方块模拟建筑）。"""
    import rasterio
    from rasterio.transform import Affine

    if gt is None:
        gt = Affine(0.5, 0.0, 400000.0, 0.0, -0.5, 3000000.0)
    data = np.zeros((3, height, width), dtype=np.uint8)
    data[:, ::8, :] = 40
    data[:, :, ::8] = 40
    for (x0, y0, w, h) in ((60, 60, 40, 24), (200, 120, 64, 32), (330, 300, 30, 30)):
        data[:, y0:y0 + h, x0:x0 + w] = 235
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height,
                       count=3, dtype=dst_dtype, crs=crs, transform=gt) as dst:
        dst.write(data)
    return path


def rot_gt_raster(path: str, width: int = 256, height: int = 256) -> str:
    """带旋转项的仿射（模拟斜置影像），验证不省略旋转项。"""
    import rasterio
    from rasterio.transform import Affine

    gt = Affine(0.5, 0.08, 400000.0, 0.08, -0.5, 3000000.0)
    data = np.full((3, height, width), 90, dtype=np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height,
                       count=3, dtype="uint8", crs="EPSG:32648", transform=gt) as dst:
        dst.write(data)
    return path


def write_manifest(tmpdir: str, detections_jsonl: str | None = None,
                   detector_profile: str = "jsonl", vector_profile: str = "null") -> str:
    """测试用 manifest：coordinator/detector/vector 全部指向当前解释器。

    这只适用于 null / jsonl 后端 —— 真正的模型后端必须在独立环境里运行。
    """
    exe = sys.executable
    manifest = {
        "schema_version": "1.0",
        "environments": {
            "coordinator": {"python": exe},
            "detector": {"python": exe},
            "vector": {"python": exe},
        },
        "models": {
            "detector": {"profile": detector_profile},
            "vectorizer": {"profile": vector_profile},
            "detections_jsonl": detections_jsonl,
        },
    }
    path = os.path.join(tmpdir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return path


def aoi_for_raster(raster_path: str, inset: float = 0.0) -> tuple[dict, str]:
    """按影像范围生成 AOI（GeoJSON）+ CRS。"""
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(raster_path) as src:
        b = src.bounds
        crs = src.crs.to_string()
        lonlat = transform_bounds(src.crs, "EPSG:4326", *b)
    return {
        "type": "Polygon",
        "coordinates": [[
            [lonlat[0], lonlat[1]], [lonlat[2], lonlat[1]],
            [lonlat[2], lonlat[3]], [lonlat[0], lonlat[3]],
            [lonlat[0], lonlat[1]],
        ]],
    }, "EPSG:4326"


def pixel_aoi(raster_path: str, x0: int, y0: int, x1: int, y1: int) -> tuple[dict, dict]:
    """按影像像素范围生成 AOI（直接在影像 CRS 里），返回 (geojson, 期望边界的像素框)。"""
    import rasterio

    with rasterio.open(raster_path) as src:
        T = src.transform
        crs = src.crs.to_string()
        xs, ys = [], []
        for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)):
            X = T.c + px * T.a + py * T.b
            Y = T.f + px * T.d + py * T.e
            xs.append(X)
            ys.append(Y)
    return (
        {"type": "Polygon", "coordinates": [[[x, y] for x, y in zip(xs, ys)]]},
        {"crs": crs, "pixels": [x0, y0, x1, y1]},
    )


def write_detections_jsonl(path: str, windows: list[dict]) -> str:
    """写 jsonl 检测结果：windows = [{window_id, boxes:[[cx,cy,w,h,ang,score]]}]。"""
    with open(path, "w", encoding="utf-8") as f:
        for w in windows:
            rec = {
                "window_id": w["window_id"],
                "detections": [
                    {"rbox_xywha_rad": list(b[:5]), "score": b[5],
                     "category": "building"} for b in w.get("boxes", [])
                ],
            }
            f.write(json.dumps(rec) + "\n")
    return path


def plugin_dir() -> str:
    """插件包目录。

    开发布局是 ``qgis_plugin/vecrot_qgis/``，而 GitHub 仓库为了让用户
    `git clone` 后一眼看到插件，把它放在顶层 ``vecrot_qgis/``。
    两种都要支持，否则同一套测试在两种布局下结论不同。
    """
    nested = os.path.join(REPO, "qgis_plugin", "vecrot_qgis")
    if os.path.isdir(nested):
        return nested
    return os.path.join(REPO, "vecrot_qgis")


def plugin_parent() -> str:
    """``plugin_dir()`` 的父目录 —— 加进 sys.path 就能 ``import vecrot_qgis``。"""
    return os.path.dirname(plugin_dir())
