"""成果导出：staging 双 GeoPackage + 校验 + report.json（开发文档 §9）。

提交语义：两个文件无法做单次 rename 的原子提交，因此以「最后写入的完成 manifest」
作为作业提交标志。QGIS 只在 manifest 出现后才加载。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Sequence

from .boxes import rbox_corners

BOX_FIELDS = ["job_id", "det_id", "class_name", "det_score", "box_type",
              "angle_rad", "vector_status", "source_window", "edge_truncated", "clipped"]
BUILDING_FIELDS = ["job_id", "det_id", "det_score", "geom_status", "clipped",
                   "edge_truncated", "area_m2", "geom_reasons"]


class ExportError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass
class ExportResult:
    boxes_path: str
    buildings_path: str
    manifest_path: str
    report_path: str
    counts: dict = field(default_factory=dict)


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_layer(gdf, gpkg_path: str, layer: str, geometry_type: str) -> None:
    import pyogrio

    pyogrio.write_dataframe(gdf, gpkg_path, layer=layer, driver="GPKG",
                            geometry_type=geometry_type,
                            promote_to_multi=(geometry_type == "MultiPolygon"))


def _empty_frame(fields: list[str], crs: str, geom_type: str = "Polygon"):
    """构造带 CRS 与 schema 的空 GeoDataFrame（空结果也必须合法，§9.3）。"""
    import geopandas as gpd

    data = {f: [] for f in fields}
    data["geometry"] = gpd.GeoSeries([], crs=crs, dtype="geometry")
    return gpd.GeoDataFrame(data, geometry="geometry", crs=crs)


def export_outputs(job_id: str, detections: Sequence[dict], buildings: Sequence[dict],
                   crs: str, output_dir: str, src_crs: str,
                   aoi_policy: str, report: dict) -> ExportResult:
    """写 staging 目录 → 校验 → 原子提升到 output_dir。

    detections: 每个元素含 det_id/det_score/class_name/rbox/vector_status/
                source_window/edge_truncated/clipped 与源像素 corners。
    buildings:  每个元素含 det_id/det_score/geom_status/geom/polygon_source_px/
                clipped/edge_truncated。
    """
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon

    if not detections and not buildings:
        pass                                    # 空结果也要写合法空层

    os.makedirs(output_dir, exist_ok=True)
    staging = os.path.join(output_dir, ".staging")
    if os.path.exists(staging):
        shutil.rmtree(staging)
    os.makedirs(staging)

    boxes_path = os.path.join(staging, "boxes.gpkg")
    buildings_path = os.path.join(staging, "buildings.gpkg")

    # ---- boxes：几何 Polygon（OBB 的四角），字段按 §9.1
    box_rows = []
    box_geoms = []
    for d in detections:
        corners = d.get("corners")
        if corners is None:
            corners = [[float(x), float(y)] for x, y in rbox_corners(d["rbox"])]
        box_geoms.append(Polygon([(float(x), float(y)) for x, y in corners]))
        box_rows.append({
            "job_id": job_id,
            "det_id": d["det_id"],
            "class_name": d.get("class_name", "building"),
            "det_score": float(d["det_score"]),
            "box_type": d.get("box_type", "obb"),
            "angle_rad": d.get("angle_rad"),
            "vector_status": d.get("vector_status", "pending"),
            "source_window": int(d.get("source_window", -1)),
            "edge_truncated": bool(d.get("edge_truncated", False)),
            "clipped": bool(d.get("clipped", False)),
        })

    if box_rows:
        boxes_gdf = gpd.GeoDataFrame(box_rows, columns=BOX_FIELDS,
                                     geometry=box_geoms, crs=crs)
    else:
        boxes_gdf = _empty_frame(BOX_FIELDS, crs)
    _write_layer(boxes_gdf, boxes_path, "boxes", "Polygon")

    # ---- buildings：统一 MultiPolygon
    b_rows = []
    b_geoms = []
    pixel_area_scale = report.get("pixel_area_m2")

    for b in buildings:
        geom = b.get("geom")
        if geom is None:
            continue
        if geom.geom_type == "Polygon":
            geom = MultiPolygon([geom])
        if geom.is_empty:
            continue
        area_m2 = None
        if pixel_area_scale is not None and b.get("area_px") is not None:
            area_m2 = float(b["area_px"]) * float(pixel_area_scale)
        b_geoms.append(geom)
        b_rows.append({
            "job_id": job_id,
            "det_id": b["det_id"],
            "det_score": float(b.get("det_score", 0.0)),
            "geom_status": b.get("geom_status", "ok"),
            "clipped": bool(b.get("clipped", False)),
            "edge_truncated": bool(b.get("edge_truncated", False)),
            "area_m2": area_m2,
            "geom_reasons": ";".join(b.get("reasons") or []),
        })

    if b_rows:
        buildings_gdf = gpd.GeoDataFrame(b_rows, columns=BUILDING_FIELDS,
                                         geometry=b_geoms, crs=crs)
    else:
        buildings_gdf = _empty_frame(BUILDING_FIELDS, crs)
    _write_layer(buildings_gdf, buildings_path, "buildings", "MultiPolygon")

    # ---- 校验（§9.3）
    _validate_gpkg(boxes_path, "boxes", "Polygon", len(box_rows), crs, BOX_FIELDS)
    _validate_gpkg(buildings_path, "buildings", "MultiPolygon", len(b_rows), crs, BUILDING_FIELDS)

    det_ids = {r["det_id"] for r in box_rows}
    b_ids = {r["det_id"] for r in b_rows}
    orphan = b_ids - det_ids
    if orphan:
        raise ExportError("EXPORT_ORPHAN_BUILDING",
                          f"buildings 中有 {len(orphan)} 个 det_id 在 boxes 中不存在")

    # ---- 提升：先搬 boxes，再搬 buildings，最后写完成 manifest
    final_boxes = os.path.join(output_dir, "boxes.gpkg")
    final_buildings = os.path.join(output_dir, "buildings.gpkg")
    manifest_path = os.path.join(output_dir, "manifest.json")
    report_path = os.path.join(output_dir, "report.json")
    if os.path.exists(manifest_path):
        # 同名成果不覆盖：job_id 唯一由上层保证，这里再挡一次
        raise ExportError("EXPORT_EXISTS", f"输出目录已有完成成果：{output_dir}")

    shutil.move(boxes_path, final_boxes)
    shutil.move(buildings_path, final_buildings)

    manifest = {
        "schema_version": report.get("schema_version", "1.0"),
        "job_id": job_id,
        "completed_at_utc": report.get("completed_at_utc"),
        "crs": crs,
        "aoi_policy": aoi_policy,
        "counts": {
            "boxes": len(box_rows),
            "buildings": len(b_rows),
        },
        "files": {
            "boxes": {"name": "boxes.gpkg", "layer": "boxes", "sha256": _sha256(final_boxes),
                      "bytes": os.path.getsize(final_boxes)},
            "buildings": {"name": "buildings.gpkg", "layer": "buildings",
                          "sha256": _sha256(final_buildings),
                          "bytes": os.path.getsize(final_buildings)},
            "report": {"name": "report.json"},
        },
    }
    report["outputs"] = {
        "boxes": final_boxes,
        "buildings": final_buildings,
        "manifest": manifest_path,
        "report": report_path,
    }
    tmp = report_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(tmp, report_path)
    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, manifest_path)         # 最后一步 = 提交标志
    shutil.rmtree(staging, ignore_errors=True)

    return ExportResult(final_boxes, final_buildings, manifest_path, report_path,
                        counts=manifest["counts"])


def _is_projected(crs: str) -> bool:
    try:
        from pyproj import CRS

        c = CRS.from_user_input(crs)
        return bool(c.is_projected) and not bool(c.is_geographic)
    except Exception:                              # noqa: BLE001
        return False


def _validate_gpkg(path: str, layer: str, geom_type: str, expected_count: int,
                   crs: str, fields: list[str]) -> None:
    import pyogrio

    if not os.path.exists(path):
        raise ExportError("EXPORT_MISSING_FILE", f"未生成 {path}")
    layers = pyogrio.list_layers(path)
    names = [str(r[0]) for r in layers]
    if layer not in names:
        raise ExportError("EXPORT_BAD_LAYER", f"{os.path.basename(path)} 缺少图层 {layer}（实际 {names}）")
    info = pyogrio.read_info(path, layer=layer)
    got_crs = info.get("crs")
    if not got_crs:
        raise ExportError("EXPORT_NO_CRS", f"{layer} 图层缺少 CRS")
    got_count = int(info.get("features", -1))
    if got_count != expected_count:
        raise ExportError("EXPORT_COUNT_MISMATCH",
                          f"{layer} 要素数 {got_count} ≠ 预期 {expected_count}")
    got_geom = str(info.get("geometry_type", ""))
    if geom_type.lower() not in got_geom.lower():
        raise ExportError("EXPORT_BAD_GEOM",
                          f"{layer} 几何类型为 {got_geom}，预期 {geom_type}")
    # pyogrio ≥0.9 的 fields 可能是 numpy 数组，不能直接做真值判断
    got_fields = [str(f) for f in (info.get("fields") if info.get("fields") is not None else [])]
    missing = [f for f in fields if f not in got_fields]
    if missing:
        raise ExportError("EXPORT_MISSING_FIELDS", f"{layer} 缺字段：{missing}")


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
