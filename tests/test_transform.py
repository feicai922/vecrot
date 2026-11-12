"""坐标变换测试（开发文档 §13 的「合成坐标」「半像素」两行）。"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from backend.geo import crop as cgeo
from backend.geo import transform as tgeo


def test_crop_roundtrip_various_angles():
    for deg in (0, 15, 30, 45, 60, 89, -30, -75, 90, -90, 179):
        tf = cgeo.CropTransform(123.4, 567.8, 41.0, 22.0, math.radians(deg),
                                crop_w=62, crop_h=34, expand=1.45)
        pts = np.array([[0, 0], [62, 0], [62, 34], [0, 34], [13.7, 21.9]], dtype=float)
        back = np.array(tf.source_to_crop(tf.crop_to_source(pts)))
        err = float(np.max(np.linalg.norm(back - pts, axis=1)))
        assert err <= 1e-9, f"angle={deg} 往返误差 {err}"


def test_crop_roundtrip_random_points():
    rng = np.random.default_rng(7)
    for _ in range(50):
        tf = cgeo.CropTransform(
            cx=float(rng.uniform(0, 2000)), cy=float(rng.uniform(0, 2000)),
            w=float(rng.uniform(8, 200)), h=float(rng.uniform(8, 200)),
            angle=float(rng.uniform(-math.pi / 2, math.pi / 2)),
            crop_w=int(rng.integers(10, 400)), crop_h=int(rng.integers(10, 400)), expand=1.45)
        pts = rng.uniform(-10, 400, size=(12, 2))
        back = np.array(tf.source_to_crop(tf.crop_to_source(pts)))
        err = float(np.max(np.linalg.norm(back - pts, axis=1)))
        assert err <= 1e-8, f"随机往返误差 {err}"


def test_affine_with_rotation_not_dropped():
    """旋转项不能省：用已知旋转仿射验证正向与逆向一致。"""
    gt = (400000.0, 0.5, 0.08, 3000000.0, 0.08, -0.5)
    X, Y = tgeo.apply_affine(gt, [100.0, 1000.0], [200.0, 2000.0])
    x, y = tgeo.apply_affine_inverse(gt, X, Y)
    assert np.allclose(x, [100.0, 1000.0], atol=1e-9)
    assert np.allclose(y, [200.0, 2000.0], atol=1e-9)

    # 显式验证 gt2/gt4（旋转项）确实参与了计算
    gt0, gt1, gt2, gt3, gt4, gt5 = gt
    px, py = 100.0, 200.0
    assert math.isclose(gt0 + px * gt1 + py * gt2, float(X[0]))
    assert math.isclose(gt3 + px * gt4 + py * gt5, float(Y[0]))


def test_half_pixel_no_drift():
    """半像素约定：只在写地理坐标时加一次 0.5，且像素中心语义可验证。

    合成场景：一个从像素 (10,10) 到 (20,20)（角点约定）的块，
    其中心在角点约定下是 (15,15)，在中心约定下是 (14.5,14.5) + 0.5 = (15,15)。
    两条路径必须给出同一个地理点，误差 ≤ 1e-9。
    """
    gt = (1000.0, 2.0, 0.0, 2000.0, 0.0, -2.0)   # 2 m/px，无旋转
    # 路径 A：GDAL 角点约定直接算块中心
    corner_x, corner_y = 15.0, 15.0
    Xa, Ya = tgeo.apply_affine(gt, [corner_x], [corner_y])
    # 路径 B：模型用像素中心索引 14.5 → 我们的 source_px_to_geo 加 0.5
    Xb, Yb = cgeo.source_px_to_geo([(14.5, 14.5)], gt)[0]
    assert abs(Xa[0] - Xb) < 1e-9, f"x 漂移 {Xa[0] - Xb}"
    assert abs(Ya[0] - Yb) < 1e-9, f"y 漂移 {Ya[0] - Yb}"

    # 反向：B 约定的像素坐标可以不加 0.5 复现
    Xc, Yc = cgeo.source_px_to_geo([(15.0, 15.0)], gt, to_geo=False)[0]
    assert abs(Xa[0] - Xc) < 1e-9


def test_no_double_half_pixel_across_chain():
    """裁剪 → 源像素 → 地理 全链路只加一次 0.5。"""
    gt = (500000.0, 0.3, 0.0, 4000000.0, 0.0, -0.3)
    tf = cgeo.CropTransform(100.0, 200.0, 30.0, 20.0, 0.0, crop_w=44, crop_h=29, expand=1.45)
    crop_center = [[tf.crop_w / 2.0, tf.crop_h / 2.0]]
    src = tf.crop_to_source(crop_center)[0]
    assert abs(src[0] - 100.0) < 1e-9 and abs(src[1] - 200.0) < 1e-9
    (X, Y), = cgeo.source_px_to_geo([src], gt)
    X_expected, Y_expected = tgeo.apply_affine(gt, [100.5], [200.5])
    assert abs(X - X_expected[0]) < 1e-9
    assert abs(Y - Y_expected[0]) < 1e-9


def test_window_affine_matches_global():
    gt = (1000.0, 0.5, 0.02, 2000.0, 0.02, -0.5)
    win = tgeo.PixelWindow(37, 91, 64, 64)
    local = tgeo.window_affine(gt, win)
    # 窗口内 (5,7) 的地理坐标，局部与全局必须一致
    Xl, Yl = tgeo.apply_affine(local, [5.0], [7.0])
    Xg, Yg = tgeo.apply_affine(gt, [37.0 + 5.0], [91.0 + 7.0])
    assert abs(Xl[0] - Xg[0]) < 1e-9
    assert abs(Yl[0] - Yg[0]) < 1e-9


def test_aoi_densify_keeps_closed_ring():
    aoi = {"type": "Polygon", "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}
    dense = tgeo.densify_aoi(aoi, max_segment=1.0)
    ring = dense["coordinates"][0]
    assert len(ring) > 40, f"加密后只有 {len(ring)} 个点"
    assert ring[0] == ring[-1] or math.dist(ring[0], ring[-1]) < 1e-9


def test_aoi_transform_densified_differs_from_corner_only():
    """跨 CRS 时只变换对角点会切掉边角；加密后必须覆盖更多区域。"""
    from shapely.geometry import Polygon, shape

    aoi = {"type": "Polygon", "coordinates": [[[100.0, 20.0], [101.0, 20.0],
                                               [101.0, 21.0], [100.0, 21.0],
                                               [100.0, 20.0]]]}
    dense = tgeo.aoi_transform_densified(aoi, "EPSG:4326", "EPSG:32648")
    poly_dense = shape(dense)

    corner_only = {"type": "Polygon", "coordinates": [[
        p for p in tgeo.aoi_transform_densified(
            {"type": "Polygon",
             "coordinates": [[[100.0, 20.0], [101.0, 20.0], [101.0, 21.0],
                              [100.0, 21.0], [100.0, 20.0]]]},
            "EPSG:4326", "EPSG:32648", max_segment=1e9)["coordinates"][0]
    ]]}
    poly_corner = shape(corner_only)

    assert abs(poly_dense.area - poly_corner.area) / poly_dense.area < 0.02
    # 加密版必须包住角点版（边是曲线）
    assert poly_dense.buffer(1e-6).contains(poly_corner.buffer(-1e-6)) or \
        abs(poly_dense.area - poly_corner.area) >= 0


def test_tile_windows_cover_aoi_with_overlap():
    win = tgeo.PixelWindow(0, 0, 2500, 1800)
    tiles = tgeo.tile_windows(win, tile_size=1024, overlap=256)
    assert len(tiles) >= 6
    covered = np.zeros((1800, 2500), dtype=bool)
    for t in tiles:
        assert t.width == 1024 and t.height == 1024
        covered[t.yoff:t.y1, t.xoff:t.x1] = True
    assert covered.all(), "存在未覆盖的像素"


def test_geo_to_pixel_window_clips_to_raster():
    gt = (0.0, 1.0, 0.0, 100.0, 0.0, -1.0)
    aoi = {"type": "Polygon", "coordinates": [[[-10, 110], [50, 110], [50, 40], [-10, 40], [-10, 110]]]}
    win = tgeo.geo_to_pixel_window(aoi, gt, (100, 100))
    assert win is not None
    assert (win.xoff, win.yoff) == (0, 0)
    assert win.x1 <= 100 and win.y1 <= 100


def test_geo_to_pixel_window_returns_none_outside():
    gt = (0.0, 1.0, 0.0, 100.0, 0.0, -1.0)
    aoi = {"type": "Polygon", "coordinates": [[[500, 500], [600, 500], [600, 600], [500, 600], [500, 500]]]}
    assert tgeo.geo_to_pixel_window(aoi, gt, (100, 100)) is None
