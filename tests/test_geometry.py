"""几何：旋转 IoU、去重、多边形质检、HBB 适配。"""
from __future__ import annotations

import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from backend.geo import boxes as bgeo
from backend.geo import quality

# ---------------------------------------------------------------- 旋转 IoU


def test_rbox_iou_identical():
    r = (10.0, 20.0, 40.0, 20.0, math.radians(30))
    assert abs(bgeo.rbox_iou(r, r) - 1.0) < 1e-9


def test_rbox_iou_disjoint():
    a = (0.0, 0.0, 10.0, 10.0, 0.0)
    b = (100.0, 100.0, 10.0, 10.0, 0.0)
    assert bgeo.rbox_iou(a, b) == 0.0


def test_rbox_iou_half_overlap_axis_aligned():
    a = (0.0, 0.0, 10.0, 10.0, 0.0)
    b = (5.0, 0.0, 10.0, 10.0, 0.0)
    # 交集 5×10=50，并集 200-50=150
    assert abs(bgeo.rbox_iou(a, b) - 50.0 / 150.0) < 1e-9


def test_rbox_iou_rotation_aware():
    """斜置框与轴对齐框：包络相交但实体几乎不交，精确 IoU 必须反映这一点。"""
    a = (0.0, 0.0, 20.0, 2.0, 0.0)              # 很扁的水平框
    b = (0.0, 0.0, 20.0, 2.0, math.radians(90))  # 同中心竖框
    iou = bgeo.rbox_iou(a, b)
    # 两个 20×2 的框互相垂直交叉：交集 2×2=4，并集 80-4=76
    assert abs(iou - 4.0 / 76.0) < 1e-6, iou


def test_rbox_corners_roundtrip():
    for deg in (0, 20, 45, -60, 89):
        r = (123.0, 456.0, 50.0, 20.0, math.radians(deg))
        c = bgeo.rbox_corners(r)
        back = bgeo.corners_to_rbox(c)
        assert abs(back[0] - r[0]) < 1e-9
        assert abs(back[1] - r[1]) < 1e-9
        assert abs(back[2] - r[2]) < 1e-9
        assert abs(back[3] - r[3]) < 1e-9


def test_rbox_le90_normalization():
    """角度必须归一到 (-π/2, π/2] 且 w ≥ h。"""
    cx, cy, w, h, ang = bgeo.corners_to_rbox([[0, 0], [0, 10], [4, 10], [4, 0]])
    assert w >= h
    assert -math.pi / 2 < ang <= math.pi / 2
    assert abs(w - 10.0) < 1e-9 and abs(h - 4.0) < 1e-9


# ------------------------------------------------------------------ 去重


def _det(wid, score, rbox, edge=False):
    return bgeo.Detection(wid, score, rbox, edge_truncated=edge)


def test_dedup_keeps_higher_score():
    a = _det(0, 0.9, (100.0, 100.0, 40.0, 20.0, 0.0))
    b = _det(1, 0.7, (101.0, 100.0, 40.0, 20.0, 0.0))
    res = bgeo.dedup_detections([b, a], iou_threshold=0.5)
    assert len(res.kept) == 1
    assert res.kept[0].score == 0.9
    assert len(res.suppressed) == 1


def test_dedup_dense_row_houses_not_merged():
    """联排楼：相邻但不重叠的框不能因为挨着就被合并（§13 密集建筑行）。"""
    dets = [_det(0, 0.9, (i * 21.0, 100.0, 20.0, 12.0, 0.0)) for i in range(10)]
    res = bgeo.dedup_detections(dets, iou_threshold=0.5)
    assert len(res.kept) == 10, f"只保留 {len(res.kept)} 个，联排被误合并"


def test_dedup_rotated_neighbours_not_suppressed():
    """斜置联排：包络相交但实体不交，不能被抑制。"""
    ang = math.radians(35)
    dets = [_det(0, 0.9, (100.0, 100.0, 38.0, 14.0, ang)),
            _det(0, 0.9, (100.0 + 30.0 * math.cos(ang) + 12.0 * math.sin(ang),
                          100.0 + 30.0 * math.sin(ang) - 12.0 * math.cos(ang),
                          38.0, 14.0, ang))]
    assert bgeo.aabb_iou(bgeo.aabb_of(bgeo.rbox_corners(dets[0].rbox)),
                         bgeo.aabb_of(bgeo.rbox_corners(dets[1].rbox))) > 0
    res = bgeo.dedup_detections(dets, iou_threshold=0.5)
    assert len(res.kept) == 2, "旋转邻居被误抑制（说明用了包络 IoU 而非旋转 IoU）"


def test_dedup_cross_window_duplicate():
    """同一栋楼落在两个重叠窗口里：分数高的保留，关系可追溯。"""
    a = _det(0, 0.85, (500.0, 500.0, 30.0, 20.0, math.radians(10)))
    b = _det(4, 0.80, (501.0, 499.0, 30.0, 20.0, math.radians(10)))
    res = bgeo.dedup_detections([a, b], iou_threshold=0.5)
    assert len(res.kept) == 1
    assert res.kept[0].window_id == 0
    assert res.suppressed[0][0].window_id == 4


def test_dedup_deterministic_ids():
    dets = [_det(0, 0.9, (i * 21.0, 100.0, 20.0, 12.0, 0.0)) for i in range(5)]
    dets.append(_det(1, 0.9, (2 * 21.0, 100.0, 20.0, 12.0, 0.0)))   # 同分重复
    ids1 = [d["det_id"] for d in bgeo.assign_det_ids(bgeo.dedup_detections(dets).kept)]
    ids2 = [d["det_id"] for d in bgeo.assign_det_ids(bgeo.dedup_detections(dets).kept)]
    assert ids1 == ids2


def test_dedup_empty():
    res = bgeo.dedup_detections([], iou_threshold=0.5)
    assert res.kept == [] and res.suppressed == []


# ------------------------------------------------------------------ 质检


def test_polygon_valid_rectangle():
    chk = quality.check_polygon([[0, 0], [10, 0], [10, 5], [0, 5]])
    assert chk.status == quality.STATUS_OK
    assert chk.geometry is not None and abs(chk.area_after - 50.0) < 1e-9


def test_polygon_too_few_vertices():
    chk = quality.check_polygon([[0, 0], [10, 0]])
    assert chk.status == quality.STATUS_DEGENERATE
    assert chk.geometry is None


def test_polygon_zero_area():
    chk = quality.check_polygon([[0, 0], [10, 0], [20, 0], [0, 0]])
    assert chk.status == quality.STATUS_DEGENERATE


def test_polygon_non_finite():
    chk = quality.check_polygon([[0, 0], [float("nan"), 0], [10, 10]])
    assert chk.status == quality.STATUS_INVALID
    assert "non_finite_coordinate" in chk.reasons


def test_polygon_duplicate_vertices_removed():
    chk = quality.check_polygon([[0, 0], [0, 0], [10, 0], [10, 0], [10, 5], [0, 5], [0, 0]])
    assert chk.status == quality.STATUS_OK
    assert abs(chk.area_after - 50.0) < 1e-9


def test_polygon_self_intersection_repaired_not_dropped():
    """自交的「领结」多边形：受控 make_valid，保留面部分，不能整栋丢弃。

    领结的鞋带面积 = +50 与 -50 抵消成 0，所以 area_before 为 0 是正常的；
    关键是修复后必须拿到两块面积为 50 的面，而不是把整栋楼判死。
    """
    bowtie = [[0, 0], [10, 10], [10, 0], [0, 10]]
    chk = quality.check_polygon(bowtie)
    assert chk.status == quality.STATUS_REPAIRED, chk
    assert chk.geometry is not None
    assert abs(chk.area_after - 50.0) < 1e-6
    assert len(chk.geometry.geoms) == 2
    assert "Self-intersection" in " ".join(chk.reasons)


def test_polygon_area_below_min_dropped():
    chk = quality.check_polygon([[0, 0], [0.001, 0], [0.001, 0.001]], min_area=1e-3)
    assert chk.status == quality.STATUS_DEGENERATE


# --------------------------------------------------------------- HBB 适配


def test_hbb_to_obb_no_forced_swap():
    """HBB → OBB 不能为了满足 w≥h 无条件交换边（§7.1）。"""
    cx, cy, w, h, ang = bgeo.rbox_from_xyxy(0, 0, 10, 4)
    assert (cx, cy) == (5.0, 2.0)
    assert w == 10.0 and h == 4.0 and ang == 0.0
    # 转成 OBB 时角度必须同步（这里角度为 0，宽高按输入）
    corners = bgeo.rbox_corners((cx, cy, w, h, ang))
    x1, y1, x2, y2 = bgeo.aabb_of(corners)
    assert abs(x1 - 0.0) < 1e-9 and abs(y1 - 0.0) < 1e-9
    assert abs(x2 - 10.0) < 1e-9 and abs(y2 - 4.0) < 1e-9
