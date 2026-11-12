"""job.json 校验与规范化（开发文档 §8.1）。

不用 jsonschema 库，手写校验：外部环境依赖越少越好，错误信息也能直接指到字段。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .version import SCHEMA_VERSION


class JobSpecError(ValueError):
    """job.json 不合法；code 供 UI 分类，message 给人看。"""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass
class InferenceParams:
    mode: str = "sequential"
    #: None = 交给部署 manifest 决定（UI 与 CLI 都不该写死模型后端）
    detector_profile: str | None = None
    vector_profile: str | None = None
    score_threshold: float = 0.3
    tile_size: int = 1024
    tile_overlap: int = 256
    crop_expand: float = 1.45
    vector_batch: int = 4
    gpu_index: int = 0
    det_dedup_iou: float = 0.5
    #: 生成阶段失败重试次数上限（OOM 降批也计入）
    max_retries: int = 2
    #: 用户自选的模型路径。为空则用 deployment manifest 里的默认值。
    #: 这三项让「同一套流水线换模型」不需要改 manifest。
    detector_weights: str | None = None
    detector_config: str | None = None
    detector_num_classes: int | None = None
    vector_ckpt: str | None = None
    vector_base: str | None = None
    #: 场景分类阈值：带 scene_head 的检测模型低于此值直接判「无建筑」并跳过。
    #: None = 用 manifest / 默认 0.5
    scene_threshold: float | None = None


@dataclass
class OutputParams:
    directory: str = ""
    crs_policy: str = "source"          # source | <EPSG 字符串>
    aoi_policy: str = "intersect_keep_complete"   # | clip_strict


@dataclass
class JobSpec:
    job_id: str
    raster_path: str
    bands_rgb: tuple[int, int, int]
    aoi_geojson: dict
    aoi_crs: str | None
    inference: InferenceParams = field(default_factory=InferenceParams)
    output: OutputParams = field(default_factory=OutputParams)
    schema_version: str = SCHEMA_VERSION

    # ---- 派生 ----
    @property
    def output_dir(self) -> str:
        return self.output.directory

    def to_manifest(self) -> dict:
        """写进 work/jobs/<id>/job.json 的规范化副本（不含路径以外的秘密）。"""
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "input": {
                "raster_path": self.raster_path,
                "bands_rgb": list(self.bands_rgb),
                "aoi_geojson": self.aoi_geojson,
                "aoi_crs": self.aoi_crs,
            },
            "inference": vars(self.inference),
            "output": vars(self.output),
        }


def _num(d: dict, key: str, default, cast, lo=None, hi=None, code="JOB_SPEC_INVALID"):
    if key not in d or d[key] is None:
        return default
    try:
        v = cast(d[key])
    except (TypeError, ValueError):
        raise JobSpecError(code, f"字段 {key} 不是合法数值：{d[key]!r}")
    if lo is not None and v < lo:
        raise JobSpecError(code, f"字段 {key} 必须 ≥ {lo}，实际 {v}")
    if hi is not None and v > hi:
        raise JobSpecError(code, f"字段 {key} 必须 ≤ {hi}，实际 {v}")
    return v


def _require_str(d: dict, key: str, where: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise JobSpecError("JOB_SPEC_INVALID", f"{where}.{key} 缺失或不是非空字符串")
    return v


def _check_aoi(aoi: Any) -> dict:
    if not isinstance(aoi, dict) or aoi.get("type") not in ("Polygon", "MultiPolygon"):
        raise JobSpecError("JOB_SPEC_INVALID", "input.aoi_geojson 必须是 Polygon 或 MultiPolygon")
    coords = aoi.get("coordinates")
    if not isinstance(coords, list) or not coords:
        raise JobSpecError("JOB_SPEC_INVALID", "input.aoi_geojson.coordinates 为空")

    polygons = [coords] if aoi["type"] == "Polygon" else coords
    for pi, poly in enumerate(polygons):
        if not isinstance(poly, list) or not poly:
            raise JobSpecError("JOB_SPEC_INVALID", f"AOI 第 {pi} 个环组为空")
        ring = poly[0]
        if not isinstance(ring, list) or len(ring) < 3:
            raise JobSpecError("JOB_SPEC_INVALID", f"AOI 第 {pi} 个外环顶点少于 3 个")
        for pt in ring:
            if (not isinstance(pt, (list, tuple)) or len(pt) < 2
                    or not all(isinstance(c, (int, float)) for c in pt[:2])):
                raise JobSpecError("JOB_SPEC_INVALID", f"AOI 第 {pi} 个环含非法坐标 {pt!r}")
    return aoi


def parse_job_spec(raw: Any) -> JobSpec:
    if not isinstance(raw, dict):
        raise JobSpecError("JOB_SPEC_INVALID", "job.json 顶层必须是对象")

    sv = raw.get("schema_version")
    if sv != SCHEMA_VERSION:
        raise JobSpecError("SCHEMA_UNSUPPORTED",
                           f"schema_version={sv!r} 不受支持（本版本只接受 {SCHEMA_VERSION}）")

    job_id = _require_str(raw, "job_id", "job")
    if not all(c.isalnum() or c in "_-" for c in job_id):
        raise JobSpecError("JOB_SPEC_INVALID",
                           "job_id 只能含字母、数字、下划线、连字符（它会成为目录名）")

    inp = raw.get("input")
    if not isinstance(inp, dict):
        raise JobSpecError("JOB_SPEC_INVALID", "缺少 input 对象")
    raster = _require_str(inp, "raster_path", "input")
    if not os.path.isabs(raster):
        raise JobSpecError("JOB_SPEC_INVALID", f"input.raster_path 必须是绝对路径：{raster}")

    bands = inp.get("bands_rgb", [1, 2, 3])
    if (not isinstance(bands, (list, tuple)) or len(bands) != 3
            or not all(isinstance(b, int) and b >= 1 for b in bands)):
        raise JobSpecError("JOB_SPEC_INVALID", "input.bands_rgb 必须是三个 ≥1 的整数波段号")

    aoi = _check_aoi(inp.get("aoi_geojson"))
    aoi_crs = inp.get("aoi_crs")
    if aoi_crs is not None and not isinstance(aoi_crs, str):
        raise JobSpecError("JOB_SPEC_INVALID", "input.aoi_crs 必须是字符串或 null")

    inf_raw = raw.get("inference") or {}
    if not isinstance(inf_raw, dict):
        raise JobSpecError("JOB_SPEC_INVALID", "inference 必须是对象")
    tile_size = _num(inf_raw, "tile_size", 1024, int, lo=64, hi=4096)
    overlap = _num(inf_raw, "tile_overlap", 256, int, lo=0)
    if overlap >= tile_size:
        raise JobSpecError("JOB_SPEC_INVALID",
                           f"tile_overlap({overlap}) 必须小于 tile_size({tile_size})")
    mode = inf_raw.get("mode", "sequential")
    if mode not in ("sequential", "pipelined"):
        raise JobSpecError("JOB_SPEC_INVALID", "inference.mode 只支持 sequential / pipelined")
    if mode == "pipelined":
        # 文档 §11：流水线是 M5 阶段能力，首版明确拒绝而不是悄悄退回串行
        raise JobSpecError("MODE_NOT_IMPLEMENTED",
                           "inference.mode=pipelined 属于流水线第二阶段，本版本未实现")

    def _opt_path(key: str) -> str | None:
        """模型路径必须是绝对路径；相对路径一律拒绝（manifest 的相对基准与 job.json 不同）。"""
        v = inf_raw.get(key)
        if v is None or v == "":
            return None
        if not isinstance(v, str) or not os.path.isabs(v):
            raise JobSpecError("JOB_SPEC_INVALID",
                               f"inference.{key} 必须是绝对路径：{v!r}")
        return v

    inference = InferenceParams(
        mode=mode,
        detector_profile=inf_raw.get("detector_profile"),
        vector_profile=inf_raw.get("vector_profile"),
        detector_weights=_opt_path("detector_weights"),
        detector_config=_opt_path("detector_config"),
        detector_num_classes=(
            int(inf_raw["detector_num_classes"])
            if inf_raw.get("detector_num_classes") not in (None, "") else None),
        vector_ckpt=_opt_path("vector_ckpt"),
        vector_base=_opt_path("vector_base"),
        scene_threshold=(
            _num(inf_raw, "scene_threshold", None, float, lo=0.0, hi=1.0)
            if inf_raw.get("scene_threshold") is not None else None),
        score_threshold=_num(inf_raw, "score_threshold", 0.3, float, lo=0.0, hi=1.0),
        tile_size=tile_size,
        tile_overlap=overlap,
        crop_expand=_num(inf_raw, "crop_expand", 1.45, float, lo=1.0, hi=4.0),
        vector_batch=_num(inf_raw, "vector_batch", 4, int, lo=1, hi=512),
        gpu_index=_num(inf_raw, "gpu_index", 0, int, lo=0),
        det_dedup_iou=_num(inf_raw, "det_dedup_iou", 0.5, float, lo=0.0, hi=1.0),
        max_retries=_num(inf_raw, "max_retries", 2, int, lo=0, hi=10),
    )

    out_raw = raw.get("output") or {}
    if not isinstance(out_raw, dict):
        raise JobSpecError("JOB_SPEC_INVALID", "output 必须是对象")
    out_dir = _require_str(out_raw, "directory", "output")
    if not os.path.isabs(out_dir):
        raise JobSpecError("JOB_SPEC_INVALID", f"output.directory 必须是绝对路径：{out_dir}")
    crs_policy = out_raw.get("crs_policy", "source")
    if crs_policy != "source":
        if not isinstance(crs_policy, str) or not crs_policy.upper().startswith("EPSG:"):
            raise JobSpecError("JOB_SPEC_INVALID",
                               "output.crs_policy 只接受 'source' 或 'EPSG:<code>'（不猜 EPSG）")
        try:
            int(crs_policy.split(":", 1)[1])
        except ValueError:
            raise JobSpecError("JOB_SPEC_INVALID", f"output.crs_policy 无法解析：{crs_policy}")
    aoi_policy = out_raw.get("aoi_policy", "intersect_keep_complete")
    if aoi_policy not in ("intersect_keep_complete", "clip_strict"):
        raise JobSpecError("JOB_SPEC_INVALID",
                           "output.aoi_policy 只接受 intersect_keep_complete / clip_strict")

    return JobSpec(
        job_id=job_id,
        raster_path=raster,
        bands_rgb=(bands[0], bands[1], bands[2]),
        aoi_geojson=aoi,
        aoi_crs=aoi_crs,
        inference=inference,
        output=OutputParams(directory=out_dir, crs_policy=crs_policy, aoi_policy=aoi_policy),
    )
