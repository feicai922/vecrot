"""构造 job.json（开发文档 §8.1）。纯逻辑，可在无 QGIS 环境单测。"""
from __future__ import annotations

import json
import os
import time

SCHEMA_VERSION = "1.0"


class JobSpecProblem(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def make_job_id() -> str:
    """job_id 会成为目录名：只用字母数字下划线连字符。"""
    return "vecrot_" + time.strftime("%Y%m%d_%H%M%S")


def build_job_payload(raster_path: str, aoi_geojson: dict | None, aoi_crs: str | None,
                      output_dir: str, bands_rgb=(1, 2, 3), score_threshold: float = 0.3,
                      vector_batch: int = 4, aoi_policy: str = "intersect_keep_complete",
                      crs_policy: str = "source", tile_size: int = 1024,
                      tile_overlap: int = 256, crop_expand: float = 1.45,
                      gpu_index: int = 0, job_id: str | None = None,
                      detector_profile: str | None = None,
                      vector_profile: str | None = None,
                      model_overrides: dict | None = None) -> dict:
    """组装 job.json 内容。

    校验在插件侧先做一遍，让用户在点"运行"前就拿到清晰错误，
    而不是等外部进程报错。真正的权威校验仍在 backend.contracts.jobspec。
    """
    if not raster_path or not os.path.isabs(raster_path):
        raise JobSpecProblem("RASTER_PATH_INVALID", "影像必须是绝对路径")
    if not os.path.exists(raster_path):
        raise JobSpecProblem("RASTER_MISSING", f"影像不存在：{raster_path}")
    if not output_dir or not os.path.isabs(output_dir):
        raise JobSpecProblem("OUTPUT_PATH_INVALID", "输出目录必须是绝对路径")

    if aoi_geojson is None:
        raise JobSpecProblem("AOI_MISSING", "请先在地图上框选范围")
    if aoi_geojson.get("type") not in ("Polygon", "MultiPolygon"):
        raise JobSpecProblem("AOI_INVALID", "AOI 必须是 Polygon 或 MultiPolygon")
    if aoi_crs is None:
        raise JobSpecProblem("AOI_CRS_MISSING",
                             "无法确定 AOI 的坐标系；请确认地图或影像 CRS")

    if not (0.0 <= score_threshold <= 1.0):
        raise JobSpecProblem("SCORE_INVALID", "检测分数阈值必须在 0 到 1 之间")
    if vector_batch < 1:
        raise JobSpecProblem("BATCH_INVALID", "生成批大小必须 ≥ 1")
    if aoi_policy not in ("intersect_keep_complete", "clip_strict"):
        raise JobSpecProblem("AOI_POLICY_INVALID", f"未知 AOI 策略：{aoi_policy}")
    if crs_policy != "source":
        if not crs_policy.upper().startswith("EPSG:"):
            raise JobSpecProblem("CRS_POLICY_INVALID",
                                 "输出 CRS 只接受 source 或 EPSG:<code>")
        try:
            int(crs_policy.split(":", 1)[1])
        except ValueError:
            raise JobSpecProblem("CRS_POLICY_INVALID", f"无法解析：{crs_policy}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id or make_job_id(),
        "input": {
            "raster_path": os.path.normpath(raster_path),
            "bands_rgb": [int(bands_rgb[0]), int(bands_rgb[1]), int(bands_rgb[2])],
            "aoi_geojson": aoi_geojson,
            "aoi_crs": aoi_crs,
        },
        "inference": {
            "mode": "sequential",
            "score_threshold": float(score_threshold),
            "tile_size": int(tile_size),
            "tile_overlap": int(tile_overlap),
            "crop_expand": float(crop_expand),
            "vector_batch": int(vector_batch),
            "gpu_index": int(gpu_index),
        },
        "output": {
            "directory": os.path.normpath(output_dir),
            "crs_policy": crs_policy,
            "aoi_policy": aoi_policy,
        },
    }
    if detector_profile:
        payload["inference"]["detector_profile"] = detector_profile
    if vector_profile:
        payload["inference"]["vector_profile"] = vector_profile

    # 用户自选的模型路径：只写非空项，让后端回退到 manifest 默认值
    for key, val in (model_overrides or {}).items():
        if val in (None, ""):
            continue
        if key in ("detector_weights", "detector_config", "vector_ckpt", "vector_base"):
            if not os.path.isabs(val):
                raise JobSpecProblem("MODEL_PATH_INVALID",
                                     f"{key} 必须是绝对路径：{val}")
        payload["inference"][key] = val
    return payload


def write_job_file(payload: dict, work_dir: str) -> str:
    """写到 work/jobs/<job_id>/job.json，返回路径。"""
    job_dir = os.path.join(work_dir, "jobs", payload["job_id"])
    os.makedirs(job_dir, exist_ok=True)
    path = os.path.join(job_dir, "job.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path
