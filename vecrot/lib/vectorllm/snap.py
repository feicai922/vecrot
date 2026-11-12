"""边证据引导的顶点吸附（P1.1，**零训练**）。

问题（本仓库实测）：模型在 128×128 的裁剪里出 4 个顶点，边界偏差中位 >2 个网格 cell；
`topo.refine()` 只能做"直角化 + 共墙吸附"这种**纯几何**修正，它不看图像，所以治不了
"整条边整体偏了 2~5px"这类错误（gzda_v0 test 上边缘距离均值 4~6px、B-F1 仅 0.33）。

本模块用**原分辨率影像的梯度脊线**把每条边吸附过去：

1. 在瓦片原图上取多边形外接框 + 8px 边距的窗口，算梯度幅值（Sobel + 高斯平滑）；
2. 每条边采样 K 个内部点，沿法向搜索 ±r 内的梯度极大值 → 得到该边的"脊线采样点"；
   用 (t, offset) 做鲁棒直线拟合（截断残差），避免个别点被邻楼/阴影带偏；
3. 相邻两条边的拟合直线求交 → 新顶点；求不出交/夹角过小时保留原顶点；
4. **门控**：只有当"边的平均梯度支撑"变好、且面积变化 ≤ 30%、几何有效时才接受，否则退回原多边形。
   门控不需要 GT（部署可用），也保证不会把已经很好的轮廓弄坏。

与 `topo.refine()` 的关系：`refine` 管"形状规则化"（直角/共墙），本模块管"贴合真实边缘"。
两者串行使用：先 snap（贴边）再 refine（规则化），或反过来——两种顺序都可以评估。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


# --------------------------------------------------------------------------- 图像侧


def _gray_patch(image, box: Tuple[float, float, float, float], pad: int = 8):
    """从整幅影像里取窗口灰度 + Sobel 梯度幅值（返回 patch, (x0, y0), grad）。"""
    from scipy import ndimage

    x0 = int(np.floor(box[0])) - pad
    y0 = int(np.floor(box[1])) - pad
    x1 = int(np.ceil(box[2])) + pad
    y1 = int(np.ceil(box[3])) + pad
    W, H = image.size
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    patch = np.asarray(image.crop((x0, y0, x1, y1)).convert("L"), dtype=np.float32)
    patch = ndimage.gaussian_filter(patch, 0.8)
    gx = ndimage.sobel(patch, axis=1)
    gy = ndimage.sobel(patch, axis=0)
    grad = np.hypot(gx, gy)
    return grad, (x0, y0)


def _bilinear(grad: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = grad.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    g00 = grad[y0, x0]
    g10 = grad[y0, np.minimum(x0 + 1, w - 1)]
    g01 = grad[np.minimum(y0 + 1, h - 1), x0]
    g11 = grad[np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)]
    return (g00 * (1 - fx) * (1 - fy) + g10 * fx * (1 - fy)
            + g01 * (1 - fx) * fy + g11 * fx * fy)


def _ring(points: Sequence[Point]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(arr) >= 2 and np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    return arr


def edge_support(image, ring: np.ndarray, n_samples: int = 24) -> float:
    """多边形的"边缘梯度支撑"：沿所有边均匀采样，取梯度幅值的平均（越高越贴合真实边缘）。"""
    if len(ring) < 3:
        return 0.0
    idx = np.arange(len(ring))
    pts = ring[idx]
    nxt = ring[(idx + 1) % len(ring)]
    grad, (x0, y0) = _gray_patch(image, (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
    total, count = 0.0, 0
    per = max(2, n_samples // len(ring))
    for a, b in zip(pts, nxt):
        t = np.linspace(0.0, 1.0, per)
        xs = a[0] + (b[0] - a[0]) * t - x0
        ys = a[1] + (b[1] - a[1]) * t - y0
        total += float(_bilinear(grad, xs, ys).mean())
        count += 1
    return total / max(count, 1)


# --------------------------------------------------------------------------- 边长拟合


def _fit_edge_line(grad, origin, a: np.ndarray, b: np.ndarray, radius: float, n_try: int = 9,
                   trim: float = 2.0):
    """对一条边做"法向搜索 + 直线拟合"。返回 (点, 方向) 与拟合质量；失败返回 None。"""
    d = b - a
    length = float(np.hypot(*d))
    if length < 4.0:
        return None
    u = d / length
    n = np.array([-u[1], u[0]])
    ts = np.linspace(0.12, 0.88, n_try)
    offsets = np.linspace(-radius, radius, int(radius * 4) + 1)
    keep_pts, keep_off, keep_t = [], [], []
    for t in ts:
        base = a + d * t
        px = base[0] - origin[0] + n[0] * offsets
        py = base[1] - origin[1] + n[1] * offsets
        vals = _bilinear(grad, px, py)
        k = int(np.argmax(vals))
        peak = float(vals[k])
        # 峰值必须明显高于该扫描线的中位数（否则这条边上没有可吸附的脊线）
        if peak <= float(np.median(vals)) * 1.15:
            continue
        keep_pts.append(base + n * offsets[k])
        keep_off.append(offsets[k])
        keep_t.append(t)
    if len(keep_pts) < 4:
        return None
    t_arr = np.asarray(keep_t)
    off = np.asarray(keep_off)
    # 截断最小二乘：off = c0 + c1 * t
    c1, c0 = np.polyfit(t_arr, off, 1)
    for _ in range(2):
        pred = c0 + c1 * t_arr
        resid = np.abs(off - pred)
        ok = resid <= max(trim, resid.mean() * 2.0)
        if ok.all() or ok.sum() < 3:
            break
        c1, c0 = np.polyfit(t_arr[ok], off[ok], 1)
    resid = np.abs(off - (c0 + c1 * t_arr))
    # c1 是"整条边上的偏移变化量"（t∈[0,1]），不需要再乘边长——早期写成 c1*length 会因量纲错误
    # 把所有边都判成失败（实测接受率为 0）。
    if resid.mean() > radius * 0.6 or np.abs(c0) > radius or np.abs(c1) > radius:
        return None
    p0 = a + n * c0
    p1 = (a + d * 1.0) + n * (c0 + c1)
    return p0, p1 - p0, float(resid.mean())


def _intersect(p1, d1, p2, d2) -> Optional[np.ndarray]:
    a = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    t = np.linalg.solve(a, p2 - p1)
    return p1 + d1 * t[0]


# --------------------------------------------------------------------------- 主入口


def snap_polygon(image, polygon: Sequence[Point], radius: float = 3.0,
                 min_gain: float = 0.05, max_area_change: float = 0.30,
                 max_vertices: int = 64) -> Tuple[List[Point], dict]:
    """把多边形吸附到影像上的真实边缘。返回 (新多边形, 诊断信息)。

    `image` 是**整幅瓦片**（PIL），`polygon` 是瓦片像素坐标。不做任何训练。
    """
    ring = _ring(polygon)
    info = {"changed": False, "gain": 0.0, "reason": ""}
    if len(ring) < 3 or len(ring) > max_vertices:
        info["reason"] = "too_few_or_many"
        return list(map(tuple, ring)), info

    box = (ring[:, 0].min(), ring[:, 1].min(), ring[:, 0].max(), ring[:, 1].max())
    grad, origin = _gray_patch(image, box, pad=int(radius) + 6)
    base_score = edge_support(image, ring)

    lines = []
    for i in range(len(ring)):
        fit = _fit_edge_line(grad, origin, ring[i], ring[(i + 1) % len(ring)], radius)
        lines.append(fit)
    if sum(1 for f in lines if f) < 2:
        info["reason"] = "no_lines"
        return list(map(tuple, ring)), info

    out = []
    for i in range(len(ring)):
        prev = lines[i - 1]
        cur = lines[i]
        if prev is None or cur is None:
            out.append(tuple(ring[i]))
            continue
        p = _intersect(prev[0], prev[1], cur[0], cur[1])
        if p is None or np.hypot(*(p - ring[i])) > radius * 2.0:
            out.append(tuple(ring[i]))
        else:
            out.append((float(p[0]), float(p[1])))

    new = np.asarray(out, dtype=np.float64)
    # 门控：几何有效 + 面积变化可控 + 边缘支撑变好
    from shapely.geometry import Polygon

    try:
        pg = Polygon(new)
        if not pg.is_valid or pg.area <= 0:
            info["reason"] = "invalid"
            return list(map(tuple, ring)), info
        area_ratio = pg.area / max(Polygon(ring).area, 1e-6)
        if not (1 - max_area_change <= area_ratio <= 1 + max_area_change):
            info["reason"] = f"area {area_ratio:.2f}"
            return list(map(tuple, ring)), info
    except Exception:
        info["reason"] = "shapely"
        return list(map(tuple, ring)), info

    new_score = edge_support(image, new)
    gain = new_score / max(base_score, 1e-6) - 1.0
    info["gain"] = float(gain)
    if gain < min_gain:
        info["reason"] = f"gain {gain:.3f}"
        return list(map(tuple, ring)), info
    info["changed"] = True
    return [(float(x), float(y)) for x, y in new], info
