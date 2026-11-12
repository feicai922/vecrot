"""像素窗口、仿射与 CRS 变换（开发文档 §6.3 / §6.4）。

关键点：
* 六参数仿射不能省旋转项：``X = GT0 + x*GT1 + y*GT2``，``Y = GT3 + x*GT4 + y*GT5``。
* GDAL 像素原点在左上像素的左上角；像素中心是 (0.5, 0.5)。本模块的 ``PixelWindow``
  一律用「角点约定」的浮点像素边界；需要模型中心约定时由调用方显式 +0.5，
  不在全链路盲目加。
* AOI 跨 CRS 时沿边加密再变换，不只变换对角点。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PixelWindow:
    """源影像上的一个整像素窗口（左上角包含，右下角不含）。"""

    xoff: int
    yoff: int
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.xoff, self.yoff, self.width, self.height

    @property
    def x1(self) -> int:
        return self.xoff + self.width

    @property
    def y1(self) -> int:
        return self.yoff + self.height

    def intersect(self, other: "PixelWindow") -> "PixelWindow | None":
        x0 = max(self.xoff, other.xoff)
        y0 = max(self.yoff, other.yoff)
        x1 = min(self.x1, other.x1)
        y1 = min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return PixelWindow(x0, y0, x1 - x0, y1 - y0)

    def padded_to(self, width: int, height: int, mode: str = "edge") -> dict:
        """返回把本窗口补齐到 (width, height) 所需的填充记录。

        补齐只发生在源影像边界处；返回的 pad 是 (left, top, right, bottom)，
        必须在逆变换时扣掉，否则靠边的框会整体漂移。
        """
        if width < self.width or height < self.height:
            raise ValueError("目标尺寸小于窗口尺寸，无法补齐")
        extra_w, extra_h = width - self.width, height - self.height
        if mode == "center":
            left = extra_w // 2
            top = extra_h // 2
        else:                       # 对齐左上：窗口内容保持在 (0,0)
            left = top = 0
        return {
            "pad": (left, top, extra_w - left, extra_h - top),
            "source_window": self.as_tuple(),
        }


def window_affine(gt: Sequence[float], window: PixelWindow) -> tuple[float, ...]:
    """把整景仿射搬到窗口局部坐标系（局部 (0,0) = 窗口左上角）。

    这是模型坐标 ↔ 源像素坐标的第一段，必须保留旋转项。
    """
    gt0, gt1, gt2, gt3, gt4, gt5 = [float(v) for v in gt]
    return (
        gt0 + window.xoff * gt1 + window.yoff * gt2,
        gt1, gt2,
        gt3 + window.xoff * gt4 + window.yoff * gt5,
        gt4, gt5,
    )


def apply_affine(gt: Sequence[float], xs: Iterable[float], ys: Iterable[float]):
    """应用六参数仿射：返回 (X, Y) 数组。"""
    gt0, gt1, gt2, gt3, gt4, gt5 = [float(v) for v in gt]
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    return gt0 + x * gt1 + y * gt2, gt3 + x * gt4 + y * gt5


def apply_affine_inverse(gt: Sequence[float], X, Y):
    """仿射逆变换：地理 → 像素（角点约定）。"""
    gt0, gt1, gt2, gt3, gt4, gt5 = [float(v) for v in gt]
    det = gt1 * gt5 - gt2 * gt4
    if abs(det) < 1e-18:
        raise ValueError("仿射矩阵不可逆（可能缺少地理参考）")
    dx = np.asarray(X, dtype=np.float64) - gt0
    dy = np.asarray(Y, dtype=np.float64) - gt3
    x = (gt5 * dx - gt2 * dy) / det
    y = (-gt4 * dx + gt1 * dy) / det
    return x, y


# ------------------------------------------------------------------ AOI


def densify_ring(ring: Sequence[Sequence[float]], max_segment: float) -> list[tuple[float, float]]:
    """沿环边加密到每段不超过 max_segment（与 AOI 自身单位一致）。"""
    pts = [(float(p[0]), float(p[1])) for p in ring]
    if len(pts) < 2 or max_segment <= 0:
        return pts
    out: list[tuple[float, float]] = []
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        out.append((x0, y0))
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg > max_segment:
            n = int(math.ceil(seg / max_segment))
            for k in range(1, n):
                t = k / n
                out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return out


def densify_aoi(aoi: dict, max_segment: float) -> dict:
    """对 AOI 的每个环加密（Polygon / MultiPolygon 都支持）。"""
    aoi_type = aoi["type"]
    if aoi_type == "Polygon":
        polys = [aoi["coordinates"]]
    else:
        polys = aoi["coordinates"]
    new_polys = []
    for poly in polys:
        new_rings = [densify_ring(ring, max_segment) for ring in poly]
        new_polys.append([[list(p) for p in ring] for ring in new_rings])
    if aoi_type == "Polygon":
        return {"type": "Polygon", "coordinates": new_polys[0]}
    return {"type": "MultiPolygon", "coordinates": new_polys}


def aoi_transform_densified(aoi: dict, src_crs: str, dst_crs: str,
                            max_segment: float | None = None) -> dict:
    """AOI 从一个 CRS 转到另一个 CRS。

    先按源 CRS 的合理步长加密边，再逐点变换：矩形边在投影变形下是曲线，
    只变换对角点会切掉边角区域。`max_segment` 未给出时按 AOI 外接框对角线的 1/64 估计。
    """
    from pyproj import Transformer

    if src_crs == dst_crs:
        return aoi

    if max_segment is None:
        xs, ys = _all_coords(aoi)
        diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        max_segment = max(diag / 64.0, 1e-9)

    dense = densify_aoi(aoi, max_segment)
    tr = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    def xf(coords):
        xs = np.array([p[0] for p in coords], dtype=np.float64)
        ys = np.array([p[1] for p in coords], dtype=np.float64)
        ox, oy = tr.transform(xs, ys)
        return [list(v) for v in zip(np.asarray(ox).tolist(), np.asarray(oy).tolist())]

    if dense["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": [xf(r) for r in dense["coordinates"]]}
    return {"type": "MultiPolygon",
            "coordinates": [[xf(r) for r in poly] for poly in dense["coordinates"]]}


def _all_coords(aoi: dict) -> tuple[list[float], list[float]]:
    polys = [aoi["coordinates"]] if aoi["type"] == "Polygon" else aoi["coordinates"]
    xs, ys = [], []
    for poly in polys:
        for ring in poly:
            for p in ring:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
    return xs, ys


def geo_to_pixel_window(aoi_geo: dict, gt: Sequence[float], raster_shape: tuple[int, int],
                        valid_extent: tuple[int, int, int, int] | None = None) -> PixelWindow | None:
    """AOI（影像 CRS）→ 整像素窗口，返回与影像有效范围相交后的结果。

    `raster_shape` 是 (height, width)。返回 None 表示不相交。
    """
    xs, ys = _all_coords(aoi_geo)
    px, py = apply_affine_inverse(gt, xs, ys)
    x0 = int(math.floor(float(np.min(px))))
    y0 = int(math.floor(float(np.min(py))))
    x1 = int(math.ceil(float(np.max(px))))
    y1 = int(math.ceil(float(np.max(py))))

    h, w = raster_shape
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    win = PixelWindow(x0, y0, max(x1 - x0, 0), max(y1 - y0, 0))
    if win.width == 0 or win.height == 0:
        return None
    if valid_extent is not None:
        vx0, vy0, vx1, vy1 = valid_extent
        return win.intersect(PixelWindow(vx0, vy0, vx1 - vx0, vy1 - vy0))
    return win


def pixel_corners_to_geo(gt: Sequence[float], corners: Sequence[Sequence[float]],
                         src_crs: str, dst_crs: str | None = None) -> list[list[float]]:
    """源像素角点 → 地理坐标（可选换 CRS）。显式 x/y 顺序。"""
    xs = [float(p[0]) for p in corners]
    ys = [float(p[1]) for p in corners]
    X, Y = apply_affine(gt, xs, ys)
    if dst_crs and dst_crs != src_crs:
        from pyproj import Transformer

        tr = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        X, Y = tr.transform(X, Y)
    return [[float(a), float(b)] for a, b in zip(np.asarray(X).tolist(), np.asarray(Y).tolist())]


def tile_windows(win: PixelWindow, tile_size: int, overlap: int,
                 raster_w: int | None = None,
                 raster_h: int | None = None) -> list[PixelWindow]:
    """在 AOI 窗口内按 tile_size / overlap 铺检测窗口。

    ``raster_w`` / ``raster_h`` 是源影像尺寸，用于把不足一个 tile 的窗口居中时
    不越出影像边界；不传则按 AOI 自身范围夹取。
    """
    if tile_size <= 0:
        raise ValueError("tile_size 必须为正")
    step = tile_size - overlap
    if step <= 0:
        raise ValueError("tile_overlap 必须小于 tile_size")
    raster_w = raster_w if raster_w is not None else win.x1
    raster_h = raster_h if raster_h is not None else win.y1

    def starts(lo: int, hi: int, limit_hi: int) -> list[int]:
        """窗口起点。

        覆盖范围小于一个 tile 时，把窗口**居中**在 AOI 上而不是左上对齐：
        否则多出的部分全部落在 AOI 右/下侧，那里检出的目标 100% 会被 AOI 策略丢掉
        （实测：722×435 的 AOI 用 1024 tile、左上对齐 → 138 框里 85 个在 AOI 外）。
        居中后溢出均匀分布在两侧，边缘目标仍能被完整看到。
        """
        n = hi - lo
        if n <= tile_size:
            pad = (tile_size - n) // 2
            start = lo - pad
            # 不允许超出源影像边界
            start = max(0, min(start, limit_hi - tile_size))
            return [start]
        out = list(range(lo, hi - tile_size + 1, step))
        if out[-1] + tile_size < hi:
            out.append(hi - tile_size)
        return out

    out = []
    for y in starts(win.yoff, win.y1, raster_h):
        for x in starts(win.xoff, win.x1, raster_w):
            out.append(PixelWindow(x, y, tile_size, tile_size))
    return out
