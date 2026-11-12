"""密集场景的实例分离后处理：把"一个轮廓吞掉几栋楼"拆回每栋各自的多边形。

为什么需要（实测依据，见 `docs/exp_box_tightness_and_abstention.md`）：
VectorLLM 的输出尺寸是**锚在裁剪框**上的（预测占裁面积恒定 0.27~0.39，与楼在框里多大无关），
所以当裁剪框偏大/偏松时，它会画出一个把目标+邻居一起包住的大轮廓——
外扩 2.0 时"画膨胀"占 25.6%。而它几乎不会"选错楼"（0/4231）。

对策（本模块，全部**无需训练**）：
1. `resolve_overlaps()`：像素级"争地仲裁"。每栋楼先算自己的**核心区**（内缩 r 像素的那部分，
   这块必定属于自己），重叠区里的像素判给"离哪个核心更近"的那一栋。
   效果是吞并的大轮廓被邻居按距离切开——同时**间接纠正了尺度偏差**。
2. `snap_shared_edges()`：贴在一起的两栋楼，把彼此的顶点吸附到对方边界上，
   让共享墙成为**同一条线**（消掉缝隙/双线）。
3. `mask_to_polygon()`：仲裁后的掩膜重新矢量化回多边形（Douglas-Peucker 简化）。

两种用法：
- **离线后处理**（不重训，立刻可用）：对已有预测做 `resolve_overlaps` 再评估；
- **训练/流程内**：把它当"解码后的几何一致性层"，或把"互斥损失"加进训练（见 docs）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Polygon = Sequence[Point]


# --------------------------------------------------------------------------- 栅格化


def _rasterize_masks(polygons: Sequence[Polygon], size: Tuple[int, int]) -> np.ndarray:
    """逐实例独立栅格化，返回 (n, H, W) bool。

    ⚠️ 不要用 `geometry.rasterize()` 的整数标签图来算重叠：一张图只能存一个标签，
    后画的多边形会覆盖先画的，重叠区信息直接丢失（曾经因为这个把 IoU 算成 0）。
    """
    from vectorllm.geometry import rasterize_masks

    return rasterize_masks(polygons, size)


def _distance_to_core(mask: np.ndarray, erode_px: int) -> np.ndarray:
    """某实例"离核心多远"的距离场（核心内为 0，向外单调增大）。"""
    mask_u8 = (mask > 0).astype(np.uint8)
    try:
        import cv2
    except Exception:  # pragma: no cover
        cv2 = None
    if cv2 is not None:
        kernel = np.ones((max(1, 2 * erode_px + 1),) * 2, np.uint8)
        core = cv2.erode(mask_u8, kernel) if erode_px > 0 else mask_u8
        if core.sum() == 0:  # 细长目标一腐蚀就没了，退化为原掩膜
            core = mask_u8
        return cv2.distanceTransform(1 - core, cv2.DIST_L2, 5)
    from scipy import ndimage  # pragma: no cover

    core = ndimage.binary_erosion(mask_u8, iterations=erode_px) if erode_px > 0 else mask_u8.astype(bool)
    if core.sum() == 0:
        core = mask_u8.astype(bool)
    return ndimage.distance_transform_edt(~core)


# --------------------------------------------------------------------------- 主逻辑


def resolve_overlaps(
    polygons: Sequence[Polygon],
    size: Tuple[int, int],
    erode_frac: float = 0.15,
    min_erode_px: int = 2,
    mode: str = "reassign",
    ref_polygons: Optional[Sequence[Polygon]] = None,
) -> Dict[str, object]:
    """把互相重叠的多边形切成互不重叠的一组（"离谁的核心近就归谁"）。

    Args:
        polygons: 同一张图里的若干预测多边形（像素坐标）。
        size: 图尺寸 (W, H)。
        erode_frac: 核心区按"该实例 bbox 长边的这个比例"内缩；越大越保守（保留更多争地区）。
        mode:
            - `"reassign"`（保守，默认）：**只重新分配重叠区**的像素，其余原样保留。
              保证"互不重叠"，但不会改变各自"独占"的那部分（对画太大的框作用有限）。
            - `"partition"`（激进）：把**所有实例的并集**按"离谁核心近"重新划分。
              能吃掉"画大了一圈、伸进邻居地盘"的那部分，但对伸进**空地**的超额无能为力。
        ref_polygons: 可选参考多边形（如 GT 或更可信的预测），预留给半监督仲裁。

    Returns:
        dict，含：
        - `masks`: 每实例仲裁后的掩膜（bool 数组的 list）
        - `polygons`: 仲裁后重新矢量化的多边形（保留不了原来顶点的原样，但保证互斥）
        - `stats`: 争地区像素数、被改判的实例数、面积变化等
    """
    n = len(polygons)
    if n == 0:
        return {"masks": [], "polygons": [], "stats": {"n": 0}}

    # 只在"所有实例的联合外接框"里算（整图 1024² 上逐实例做距离变换太慢）
    xs = [p[0] for poly in polygons for p in poly]
    ys = [p[1] for poly in polygons for p in poly]
    x0, y0 = max(0, int(np.floor(min(xs))) - 2), max(0, int(np.floor(min(ys))) - 2)
    x1, y1 = min(size[0], int(np.ceil(max(xs))) + 3), min(size[1], int(np.ceil(max(ys))) + 3)
    window = (x1 - x0, y1 - y0)
    shifted = [[(x - x0, y - y0) for x, y in poly] for poly in polygons]
    members = _rasterize_masks(shifted, window)
    covered = members.any(axis=0)
    contested = members.sum(axis=0) >= 2

    fields = []
    areas_before = []
    for index, polygon in enumerate(polygons):
        mask = members[index]
        areas_before.append(int(mask.sum()))
        longest = 1.0
        if len(polygon) >= 3:
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            longest = max(max(xs) - min(xs), max(ys) - min(ys))
        erode_px = max(min_erode_px, int(round(erode_frac * longest)))
        fields.append(_distance_to_core(mask, erode_px))
    fields = np.stack(fields)  # [n, h, w]

    # 仲裁：判给距离场最小的实例
    best = np.argmin(fields, axis=0)
    assigned = np.zeros((n, window[1], window[0]), dtype=bool)
    if mode == "partition":
        region = members.any(axis=0)             # 所有实例的并集
        for index in range(n):
            assigned[index] = region & (best == index)
    else:  # reassign：只动重叠区
        for index in range(n):
            own = members[index]
            assigned[index] = (own & (~contested)) | (contested & (best == index))

    # 掩膜补回整图坐标系（调用方拿到的永远是和图同尺寸的掩膜，不用关心内部窗口）
    full_masks = []
    for index in range(n):
        canvas = np.zeros((size[1], size[0]), dtype=bool)
        canvas[y0 : y0 + window[1], x0 : x0 + window[0]] = assigned[index]
        full_masks.append(canvas)
    polygons_after = [
        [(x + x0, y + y0) for x, y in mask_to_polygon(assigned[i])] for i in range(n)
    ]
    areas_after = [int(assigned[i].sum()) for i in range(n)]
    changed = sum(1 for a, b in zip(areas_before, areas_after) if a != b)
    stats = {
        "n": n,
        "mode": mode,
        "contested_px": int(contested.sum()),
        "contested_frac_of_union": float(contested.sum() / max(int(covered.sum()), 1)),
        "instances_changed": int(changed),
        "area_before": areas_before,
        "area_after": areas_after,
        "area_shrink_ratio": [
            (b / a) if a else 1.0 for a, b in zip(areas_before, areas_after)
        ],
    }
    return {"masks": full_masks, "polygons": polygons_after, "window": (x0, y0, x1, y1), "stats": stats}


def mask_to_polygon(mask: np.ndarray, epsilon_frac: float = 0.01) -> List[Point]:
    """掩膜 → 多边形（取最大外轮廓 + Douglas-Peucker 简化）。"""
    mask_u8 = (mask > 0).astype(np.uint8)
    try:
        import cv2
    except Exception:  # pragma: no cover
        cv2 = None
    if cv2 is not None:
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, max(epsilon_frac * peri, 0.5), True)
        return [(float(p[0][0]), float(p[0][1])) for p in approx]

    # 无 cv2：用 marching squares 退化实现（足够做指标，不追求最优简化）
    from skimage import measure  # pragma: no cover

    contours = measure.find_contours(mask_u8, 0.5)
    if not contours:
        return []
    contour = max(contours, key=len)  # (row, col)
    return [(float(c), float(r)) for r, c in contour]  # pragma: no cover


def snap_shared_edges(
    poly_a: Sequence[Point], poly_b: Sequence[Point], tol: float = 2.0
) -> Tuple[List[Point], List[Point]]:
    """把贴近（≤tol 像素）的顶点吸附到对方边界上，让"共享墙"成为同一条线。

    只动顶点、不改点数，适合"相邻但没重叠"的贴合建筑对（共享墙场景）。
    """
    try:
        from shapely.geometry import LineString, Polygon
    except Exception:
        return list(poly_a), list(poly_b)

    if len(poly_a) < 3 or len(poly_b) < 3:
        return list(poly_a), list(poly_b)
    ring_a, ring_b = Polygon(poly_a).exterior, Polygon(poly_b).exterior
    if not ring_a.is_valid or not ring_b.is_valid:
        return list(poly_a), list(poly_b)

    def snap(points, target):
        out = []
        for x, y in points:
            nearest = target.interpolate(target.project(__import__("shapely").geometry.Point(x, y)))
            if ((nearest.x - x) ** 2 + (nearest.y - y) ** 2) ** 0.5 <= tol:
                out.append((nearest.x, nearest.y))
            else:
                out.append((x, y))
        return out

    return snap(poly_a, ring_b), snap(poly_b, ring_a)


# --------------------------------------------------------------------------- 指标


def pairwise_overlap_area(polygons: Sequence[Polygon], size: Tuple[int, int]) -> Dict[str, float]:
    """重叠/缝隙统计（用于评估分离前后）。"""
    if not polygons:
        return {"pairs_overlapping": 0, "overlap_area": 0, "overlap_frac_of_union": 0.0}
    members = _rasterize_masks(polygons, size)
    counts = members.sum(axis=0)
    contested = counts >= 2
    union = counts >= 1
    pairs = 0
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            if np.logical_and(members[i], members[j]).any():
                pairs += 1
    return {
        "pairs_overlapping": int(pairs),
        "overlap_area": int(contested.sum()),
        "overlap_frac_of_union": float(contested.sum() / max(int(union.sum()), 1)),
    }
