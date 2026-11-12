"""拓扑精修（**纯几何，不需要掩膜**）：全局共线/共墙吸附 + 直角化。

灵感来自 GCP（Global Collinearity-aware Polygonizer, 2025）的"全局共线"和"直角损失"，
但我们**只吃 LLM 输出的多边形点序列**，不依赖任何实例掩膜：

1. `rectify_right_angles()` —— 单栋"直角化"：
   先估主方向（最小外接旋转矩形），把接近 0°/90° 的边**吸附成正交**。遮挡缺角、斜边抖动会被拉正。
2. `snap_global_collinear()` —— **跨建筑的全局共线吸附**（GCP 的 global collinearity 在多边形上的等价物）：
   把所有多边形的边按 (方向, 法向偏移) 聚类，同一簇（来自**不同**建筑）拟合一条公共直线，
   端点投影到该线 → 相邻建筑共享的墙变成**同一条线**（消双线/缝隙/错位）。
3. `interior_angles()` / `rectilinearity()` / `shared_wall_angle_gap()` —— 三个可直接上报的指标。

设计取舍：GCP 用"叠标签图再矢量化"实现全局一致，我们用"边聚类 + 公共直线拟合"实现同样效果，
好处是**不引入栅格化误差**（它们的 aggregate_mask 会连 4× 上采样后的阶梯边一起矢量化）。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Polygon = Sequence[Point]


# --------------------------------------------------------------------------- 指标


def _largest_polygon(shape):
    """buffer(0) 可能返回 MultiPolygon（自交修补后），统一取面积最大的那一个。

    踩过的坑：直接 `shape.exterior` 在 MultiPolygon 上会 AttributeError，
    再被 `getattr(..., None)` 吞成 None，最后 `.coords` 崩在几十行之外。
    """
    if shape is None or shape.is_empty:
        return None
    if shape.geom_type == "Polygon":
        return shape
    if shape.geom_type == "MultiPolygon":
        geoms = list(getattr(shape, "geoms", []))
        return max(geoms, key=lambda g: g.area) if geoms else None
    return None


def interior_angles(polygon: Polygon) -> List[float]:
    """各顶点的内角（度，0~180）。"""
    pts = list(polygon)
    n = len(pts)
    if n < 3:
        return []
    angs = []
    for i in range(n):
        a = np.asarray(pts[i - 1], dtype=float)
        b = np.asarray(pts[i], dtype=float)
        c = np.asarray(pts[(i + 1) % n], dtype=float)
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angs.append(math.degrees(math.acos(cos)))
    return angs


def rectilinearity(polygon: Polygon, tol_deg: float = 8.0) -> float:
    """"直角率"：内角落在 90°/180° 附近 tol 度内的比例（越接近 1 越规整）。"""
    angs = interior_angles(polygon)
    if not angs:
        return 0.0
    ok = sum(1 for a in angs if min(abs(a - 90.0), abs(a - 180.0), abs(a - 0.0)) <= tol_deg)
    return ok / len(angs)


def _edges(polygon: Polygon):
    n = len(polygon)
    for i in range(n):
        p, q = polygon[i], polygon[(i + 1) % n]
        dx, dy = q[0] - p[0], q[1] - p[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        # 方向归一到 [0, 180)，法向偏移 = 原点到直线的有符号距离
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        nx, ny = -dy / length, dx / length
        offset = nx * p[0] + ny * p[1]
        yield i, p, q, ang, offset, length


def _dir_normal(p: Point, q: Point) -> np.ndarray:
    """按**边的真实走向** p→q 取单位法向 `(-dy, dx)/len`。

    必须用走向而不是 `_edges()` 里折到 [0,180) 的角度：折角会丢方向，
    而 `off = nx*x + ny*y` 的符号依赖方向，两者混用会静默错（见 `snap_shared_walls_pairwise`）。
    """
    dx, dy = q[0] - p[0], q[1] - p[1]
    length = math.hypot(dx, dy) or 1.0
    return np.array([-dy / length, dx / length])


def shared_wall_angle_gap(polygons: Sequence[Polygon], tol_px: float = 3.0) -> Dict[str, float]:
    """相邻建筑"共享墙"的一致性：贴近的边对之间的**方向夹角**（度，越小越一致）+ 偏移差（px）。"""
    gaps, offsets = [], []
    polys = [list(p) for p in polygons if len(p) >= 3]
    for a in range(len(polys)):
        for b in range(a + 1, len(polys)):
            from shapely.geometry import Polygon as _P
            pa, pb = _P(polys[a]).buffer(0), _P(polys[b]).buffer(0)
            if pa.is_empty or pb.is_empty or pa.distance(pb) > tol_px:
                continue
            best = None
            for _, p1, q1, ang1, off1, _l1 in _edges(polys[a]):
                for _, p2, q2, ang2, off2, _l2 in _edges(polys[b]):
                    d_ang = min(abs(ang1 - ang2), 180 - abs(ang1 - ang2))
                    d_off = abs(abs(off1) - abs(off2))
                    # 只统计"近似平行且贴得很近"的边对
                    if d_ang <= 12.0 and d_off <= max(tol_px, 3.0):
                        if best is None or d_ang < best[0]:
                            best = (d_ang, d_off)
            if best is not None:
                gaps.append(best[0]); offsets.append(best[1])
    return {"pairs": len(gaps),
            "mean_angle_gap_deg": float(np.mean(gaps)) if gaps else 0.0,
            "mean_offset_px": float(np.mean(offsets)) if offsets else 0.0}


# --------------------------------------------------------------------------- 直角化


def rectify_right_angles(polygon: Polygon, angle_tol: float = 12.0, min_edge: float = 6.0) -> List[Point]:
    """把接近正交的边吸附成**严格直角**（先按主方向旋转到轴对齐，再拉正，最后转回去）。"""
    pts = [tuple(map(float, p)) for p in polygon]
    if len(pts) < 3:
        return pts
    try:
        from shapely.geometry import Polygon as _P
    except Exception:
        return pts
    shape = _largest_polygon(_P(pts).buffer(0))
    if shape is None or len(list(shape.exterior.coords)) < 4:
        return pts
    rect = shape.minimum_rotated_rectangle
    rcoords = list(rect.exterior.coords)[:4]
    # 主方向
    v = np.asarray(rcoords[1]) - np.asarray(rcoords[0])
    theta = math.degrees(math.atan2(v[1], v[0]))
    c, s = math.cos(math.radians(-theta)), math.sin(math.radians(-theta))

    def rot(p):
        return (c * p[0] - s * p[1], s * p[0] + c * p[1])

    def back(p):
        return (c * p[0] + s * p[1], -s * p[0] + c * p[1])

    rp = [rot(p) for p in pts]
    out = []
    n = len(rp)
    for i in range(n):
        p, q = rp[i], rp[(i + 1) % n]
        dx, dy = q[0] - p[0], q[1] - p[1]
        if math.hypot(dx, dy) < min_edge:
            out.append((p, q))
            continue
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        # 就近吸附到 0/90
        target = min((0.0, 90.0), key=lambda t: min(abs(ang - t), 180 - abs(ang - t)))
        if min(abs(ang - target), 180 - abs(ang - target)) <= angle_tol:
            if target == 0.0:
                y = (p[1] + q[1]) / 2.0
                p, q = (p[0], y), (q[0], y)
            else:
                x = (p[0] + q[0]) / 2.0
                p, q = (x, p[1]), (x, q[1])
        out.append((p, q))
    # 相邻边共享端点：用平均把两端对齐
    new = []
    for i in range(n):
        prev_q = out[i - 1][1]
        cur_p = out[i][0]
        new.append(((prev_q[0] + cur_p[0]) / 2.0, (prev_q[1] + cur_p[1]) / 2.0))
    new = [back(p) for p in new]
    try:
        fixed = _largest_polygon(_P(new).buffer(0))
        if fixed is None:
            return pts
        ring = list(fixed.exterior.coords)[:-1]
        return ring if len(ring) >= 3 else pts
    except Exception:
        return pts


# --------------------------------------------------------------------------- 全局共线吸附


def snap_global_collinear(polygons: Sequence[Polygon], angle_tol: float = 4.0, offset_tol: float = 3.0,
                          min_support: int = 2) -> List[List[Point]]:
    """**跨建筑的全局共线吸附**（GCP global collinearity 的多边形版，无需掩膜）。

    步骤：
      1. 收集所有多边形的边，记录 (方向, 法向偏移)；
      2. 按方向聚类（±angle_tol），簇内再按偏移聚类（±offset_tol）→ 得到若干"公共直线"候选；
      3. 只对**来自 ≥min_support 栋不同建筑**的簇，拟合一条公共直线（总最小二乘）；
      4. 把簇内所有端点投影到该直线上 → 相邻建筑的共享墙变成同一条线。

    返回新的多边形列表（顺序与输入一致）。
    """
    polys = [[tuple(map(float, p)) for p in poly] for poly in polygons]
    clusters: List[Dict] = []
    for pi, poly in enumerate(polys):
        if len(poly) < 3:
            continue
        for ei, p, q, ang, off, length in _edges(poly):
            if length < 6.0:          # 太短的边不参与，避免噪声
                continue
            placed = False
            for cl in clusters:
                d_ang = min(abs(cl["ang"] - ang), 180 - abs(cl["ang"] - ang))
                if d_ang <= angle_tol and abs(cl["off"] - off) <= offset_tol:
                    cl["edges"].append((pi, ei, p, q, ang, off))
                    # 用支持度加权更新簇中心
                    k = len(cl["edges"])
                    cl["ang"] = (cl["ang"] * (k - 1) + ang) / k
                    cl["off"] = (cl["off"] * (k - 1) + off) / k
                    placed = True
                    break
            if not placed:
                clusters.append({"ang": ang, "off": off, "edges": [(pi, ei, p, q, ang, off)]})

    new_polys = [list(p) for p in polys]
    snapped_clusters = 0
    for cl in clusters:
        members = {(e[0], e[1]) for e in cl["edges"]}
        n_buildings = len({e[0] for e in cl["edges"]})
        if n_buildings < min_support:
            continue
        # 拟一条公共直线：用两端点做总最小二乘（方向取簇均值方向，只解偏移）
        a = math.radians(cl["ang"])
        d = np.array([math.cos(a), math.sin(a)])
        n = np.array([-d[1], d[0]])
        pts = []
        for pi, ei, p, q, ang, off in cl["edges"]:
            pts.extend([p, q])
        offsets = [float(np.dot(np.asarray(pt), n)) for pt in pts]
        c = float(np.mean(offsets))                    # 公共直线的法向偏移
        for pi, ei, p, q, ang, off in cl["edges"]:
            for pt in (p, q):
                v = np.asarray(pt, dtype=float)
                proj = v - (np.dot(v, n) - c) * n       # 投影到公共线
                # 只把该边重新写回它所处的多边形的相邻两个顶点
                pass
            # 精确重写：把该边的两个端点都投影到公共线
            poly = new_polys[pi]
            pp = np.asarray(p, dtype=float); qq = np.asarray(q, dtype=float)
            poly[ei] = tuple(pp - (np.dot(pp, n) - c) * n)
            poly[(ei + 1) % len(poly)] = tuple(qq - (np.dot(qq, n) - c) * n)
        snapped_clusters += 1

    out = []
    for poly in new_polys:
        try:
            from shapely.geometry import Polygon as _P
            fixed = _largest_polygon(_P(poly).buffer(0))
            if fixed is None:
                out.append(poly)
                continue
            ring = list(fixed.exterior.coords)[:-1]
            out.append(ring if len(ring) >= 3 else poly)
        except Exception:
            out.append(poly)
    return out


def snap_shared_walls_pairwise(polygons: Sequence[Polygon], contact_tol: float = 3.0,
                               angle_tol: float = 10.0, offset_tol: float = 3.0,
                               max_shift: float = 2.5) -> List[List[Point]]:
    """**只对真正相邻的共享墙**做局部吸附（比 `snap_global_collinear` 保守得多）。

    为什么不用全局聚类：实测把"任意近共线的边"都拉到同一条线会把**不相邻**建筑的边也拉歪，
    IoU 从 0.825 掉到 0.696（−13pt）。正确做法是只看**贴在一起**的两栋：
      1. 两多边形距离 ≤ contact_tol 才处理；
      2. 在它们之间找"最贴近且近平行"的那一对边（共享墙）；
      3. 只把这一对边吸附到同一条公共线上，且**位移上限 max_shift 像素**，超了就放弃。
    """
    try:
        from shapely.geometry import Polygon as _P
    except Exception:
        return [list(p) for p in polygons]

    polys = [[tuple(map(float, pt)) for pt in poly] for poly in polygons]
    shapes = []
    for poly in polys:
        shape = _P(poly).buffer(0) if len(poly) >= 3 else None
        shapes.append(_largest_polygon(shape) if shape is not None else None)

    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            a, b = shapes[i], shapes[j]
            if a is None or b is None or a.distance(b) > contact_tol:
                continue
            best = None
            for ei, p1, q1, ang1, off1, _l1 in _edges(polys[i]):
                n1 = _dir_normal(p1, q1)        # 该边自身方向的法向（与 off1 同一口径）
                for ej, p2, q2, ang2, off2, _l2 in _edges(polys[j]):
                    d_ang = min(abs(ang1 - ang2), 180 - abs(ang1 - ang2))
                    if d_ang > angle_tol:
                        continue
                    m1 = np.array([(p1[0] + q1[0]) / 2.0, (p1[1] + q1[1]) / 2.0])
                    m2 = np.array([(p2[0] + q2[0]) / 2.0, (p2[1] + q2[1]) / 2.0])
                    dist = float(np.linalg.norm(m1 - m2))
                    # ⚠️（2026-10-31 修）不能直接比两条边的有符号 `off`：
                    #    `off` 的符号取决于**边的走向**（p→q），而 `ang` 被折到 [0,180) 丢了方向，
                    #    两条共墙的边走向相反时 `off` 差 2×|off| → 永远配不上、静默不吸附
                    #    （我方用例：1.6px 缝隙完全没被吸，但指标看着"跑了"）。
                    #    正确做法：把第二条边的中点投到**第一条边的法向**上，得到两直线的垂直距离。
                    off2_in_n1 = float(np.dot(n1, m2))
                    d_off = abs(off1 - off2_in_n1)
                    if d_off > offset_tol:
                        continue
                    score = dist + 5.0 * d_ang
                    if best is None or score < best[0]:
                        best = (score, ei, ej, n1, (off1 + off2_in_n1) / 2.0)
            if best is None:
                continue
            _, ei, ej, normal, off_mean = best
            for poly, e in ((polys[i], ei), (polys[j], ej)):
                for vid in (e, (e + 1) % len(poly)):
                    v = np.asarray(poly[vid], dtype=float)
                    delta = (np.dot(v, normal) - off_mean)
                    if abs(delta) <= max_shift:          # 位移有界，避免拉歪
                        poly[vid] = tuple(v - delta * normal)

    out = []
    for poly in polys:
        try:
            from shapely.geometry import Polygon as _P
            fixed = _largest_polygon(_P(poly).buffer(0))
            if fixed is None:
                out.append(poly); continue
            ring = list(fixed.exterior.coords)[:-1]
            out.append(ring if len(ring) >= 3 else poly)
        except Exception:
            out.append(poly)
    return out


def refine(polygons: Sequence[Polygon], do_rectify: bool = True, do_global: bool = True,
           angle_tol: float = 12.0, global_angle_tol: float = 4.0, global_offset_tol: float = 3.0
           ) -> List[List[Point]]:
    """一键精修：直角化 → 全局共线吸附（顺序有讲究：先各自拉正，再跨建筑对齐公共墙）。"""
    polys = [list(map(tuple, p)) for p in polygons]
    if do_rectify:
        polys = [rectify_right_angles(p, angle_tol=angle_tol) for p in polys]
    if do_global:
        # 默认走**保守的成对共享墙吸附**（全局聚类版 snap_global_collinear 会误伤，见其 docstring）
        polys = snap_shared_walls_pairwise(polys)
    return polys
