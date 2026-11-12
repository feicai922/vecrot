"""多边形质检与修复（开发文档 §9.2）。

检查顺序严格按文档：有限坐标 → 顶点数 → 闭合环 → 去连续重复点 → 非空/非零面积
→ 自交检测 → 受控 make_valid → 检查修复结果。

make_valid 可能产生 GeometryCollection：只保留面部分并记录变化，绝不因此丢整栋建筑。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import explain_validity

#: geom_status 取值：UI 与 report 共用
STATUS_OK = "ok"
STATUS_REPAIRED = "repaired"
STATUS_DEGENERATE = "degenerate"
STATUS_SELF_INTERSECT = "self_intersect_unrepaired"
STATUS_INVALID = "invalid"


@dataclass
class PolygonCheck:
    geometry: MultiPolygon | None
    status: str
    reasons: list[str] = field(default_factory=list)
    area_before: float = 0.0
    area_after: float = 0.0
    dropped_parts: int = 0
    vertex_count: int = 0


def _finite(ring: Sequence[Sequence[float]]) -> bool:
    for p in ring:
        if len(p) < 2:
            return False
        x, y = float(p[0]), float(p[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
    return True


def _dedupe_consecutive(points: list[tuple[float, float]],
                        eps: float = 1e-12) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for p in points:
        if not out or abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
            out.append(p)
    if len(out) > 1 and abs(out[0][0] - out[-1][0]) <= eps and abs(out[0][1] - out[-1][1]) <= eps:
        out.pop()                     # 去掉闭合重复点，由 Polygon 自己闭合
    return out


def _failure_status(poly, area0: float) -> str:
    """修复失败后的分类。

    * 净面积为 0（共线环、"领结"被鞋带抵消到 0）→ degenerate
    * 有面积但几何无效 → self_intersect_unrepaired
    """
    if area0 <= 0.0 or not math.isfinite(area0):
        return STATUS_DEGENERATE
    return STATUS_DEGENERATE if poly.is_valid else STATUS_SELF_INTERSECT


def check_polygon(points: Sequence[Sequence[float]], min_vertices: int = 3,
                  min_area: float = 1e-6) -> PolygonCheck:
    """对模型输出的多边形做完整质检。

    points 为闭合或非闭合环（均可），坐标为源像素。

    注意 ``area_before`` 用鞋带公式计算：自交环的正负部分会相互抵消，
    因此"领结"形的 area_before 可能为 0 而实际覆盖面积不为 0。
    """
    reasons: list[str] = []
    if not _finite(points):
        return PolygonCheck(None, STATUS_INVALID, ["non_finite_coordinate"])

    pts = _dedupe_consecutive([(float(p[0]), float(p[1])) for p in points])
    if len(pts) < min_vertices:
        return PolygonCheck(None, STATUS_DEGENERATE,
                            [f"too_few_vertices:{len(pts)}"], vertex_count=len(pts))

    try:
        poly = Polygon(pts)
    except Exception as exc:                       # noqa: BLE001
        return PolygonCheck(None, STATUS_INVALID, [f"polygon_construct_failed:{exc}"])

    area0 = float(poly.area)
    if not math.isfinite(area0):
        return PolygonCheck(None, STATUS_DEGENERATE, ["non_finite_area"],
                            vertex_count=len(pts))

    if poly.is_valid and area0 > 0.0 and area0 >= min_area:
        return PolygonCheck(MultiPolygon([poly]), STATUS_OK, reasons,
                            area_before=area0, area_after=area0, vertex_count=len(pts))

    # 走到这里有两种情况，都要先尝试受控修复再决定丢弃：
    #   a) 自交（可能伴随鞋带面积为 0 的「领结」——按面积直接判死会丢掉整栋楼）
    #   b) 合法几何但面积过小
    if poly.is_valid:
        reasons.append(f"area_below_min:{area0:.3g}")
    else:
        reasons.append(explain_validity(poly))

    repaired = None
    try:
        from shapely import make_valid          # shapely ≥ 2.0
        repaired = make_valid(poly)
    except Exception:                              # noqa: BLE001 - 老版本或未预期失败
        try:
            repaired = poly.buffer(0)
        except Exception:                          # noqa: BLE001
            repaired = None

    if repaired is None or repaired.is_empty:
        return PolygonCheck(None, _failure_status(poly, area0), reasons + ["make_valid_empty"],
                            area_before=max(area0, 0.0), vertex_count=len(pts))

    polygons, dropped = _extract_polygons(repaired)
    if not polygons:
        return PolygonCheck(None, _failure_status(poly, area0), reasons + ["no_polygonal_part"],
                            area_before=max(area0, 0.0), vertex_count=len(pts))

    mp = MultiPolygon(polygons)
    area1 = float(mp.area)
    if area1 <= 0.0 or area1 < min_area:
        return PolygonCheck(None, STATUS_DEGENERATE, reasons + ["repaired_zero_area"],
                            area_before=max(area0, 0.0), area_after=max(area1, 0.0),
                            dropped_parts=dropped, vertex_count=len(pts))

    # 修复结果偏离过大 → 标记复核，但仍然保留（不默默丢弃整栋建筑）
    if area0 > 0 and (area1 / area0 > 1.5 or area1 / area0 < 0.5):
        reasons.append(f"area_changed_ratio:{area1/area0:.2f}")
    if dropped:
        reasons.append(f"dropped_non_polygonal_parts:{dropped}")

    return PolygonCheck(mp, STATUS_REPAIRED, reasons, area_before=max(area0, 0.0),
                        area_after=area1, dropped_parts=dropped, vertex_count=len(pts))


def _extract_polygons(geom) -> tuple[list[Polygon], int]:
    """从任意几何里只取面部分，其它部分计数丢弃。"""
    dropped = 0
    polys: list[Polygon] = []
    if geom.geom_type == "Polygon":
        if geom.area > 0:
            polys.append(geom)
        return polys, dropped
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for g in geom.geoms:
            if g.geom_type == "Polygon":
                if g.area > 0:
                    polys.append(g)
            elif g.geom_type in ("MultiPolygon", "GeometryCollection"):
                sub, d = _extract_polygons(g)
                polys.extend(sub)
                dropped += d
            elif not g.is_empty:
                dropped += 1
        return polys, dropped
    return polys, 1 if not geom.is_empty else 0


def clip_polygon_to_aoi(geom: MultiPolygon, aoi_geom) -> MultiPolygon:
    """严格裁剪模式：按同一 AOI 策略裁两个图层（§6.5）。"""
    clipped = geom.intersection(aoi_geom)
    polys, _ = _extract_polygons(clipped)
    return MultiPolygon(polys) if polys else MultiPolygon()


def aoi_geometry(aoi_geo: dict):
    """GeoJSON → shapely 几何（用于相交判断与严格裁剪）。"""
    from shapely.geometry import shape

    return shape(aoi_geo)
