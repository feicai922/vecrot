"""旋转框（OBB）几何与跨窗口去重（开发文档 §7.1 / §7.2）。

坐标约定沿用 vecrot：``rbox_xywha`` = (cx, cy, w, h, angle_rad)，le90，w ≥ h，
图像坐标 x 右 y 下，正角顺时针。所有计算在**源像素平面**进行。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


# ----------------------------------------------------------------- OBB 基本


def rbox_corners(rbox: Sequence[float]) -> np.ndarray:
    """(cx, cy, w, h, ang) → 4×2 角点，顺序为「沿 w 方向正负」的四个角。"""
    cx, cy, w, h, ang = [float(v) for v in rbox]
    c, s = math.cos(ang), math.sin(ang)
    dx, dy = w / 2.0, h / 2.0
    # 局部坐标 (±dx, ±dy) 经 R(ang) 旋转到图像坐标
    local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float64)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ R.T + np.array([cx, cy], dtype=np.float64)


def corners_to_rbox(corners: Sequence[Sequence[float]]) -> tuple[float, float, float, float, float]:
    """4 角点 → (cx, cy, w, h, ang)，归一化到 le90 且 w ≥ h。"""
    pts = np.asarray(corners, dtype=np.float64)[:4]
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    # 用两条相邻边确定 w / h 方向
    e1 = pts[1] - pts[0]
    e2 = pts[3] - pts[0]
    w = float(np.linalg.norm(e1))
    h = float(np.linalg.norm(e2))
    ang = math.atan2(float(e1[1]), float(e1[0]))
    if w < h:
        w, h = h, w
        ang += math.pi / 2
    while ang <= -math.pi / 2:
        ang += math.pi
    while ang > math.pi / 2:
        ang -= math.pi
    return cx, cy, w, h, ang


def aabb_of(corners: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    pts = np.asarray(corners, dtype=np.float64)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def rbox_from_xyxy(x1, y1, x2, y2) -> tuple[float, float, float, float, float]:
    """HBB → OBB（角度 0）。宽高不为了满足 w≥h 无条件交换（§7.1）。"""
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, abs(x2 - x1), abs(y2 - y1), 0.0)


def aabb_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])   # noqa: E731
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def polygon_area(pts: np.ndarray) -> float:
    """鞋带公式（有向面积取绝对值）。"""
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def convex_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman：用凸多边形 clip 裁剪 subject（两者都需凸）。"""
    out = [tuple(p) for p in subject]

    def inside(p, a, b, ccw):
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        return cross >= -1e-12 if ccw else cross <= 1e-12

    def intersect(p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-18:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    signed = polygon_signed_area(clip)
    ccw = signed > 0
    clip_pts = [tuple(p) for p in clip]
    for i in range(len(clip_pts)):
        a, b = clip_pts[i], clip_pts[(i + 1) % len(clip_pts)]
        if not out:
            return np.empty((0, 2))
        inp = out
        out = []
        for j in range(len(inp)):
            cur = inp[j]
            prev = inp[j - 1]
            if inside(cur, a, b, ccw):
                if not inside(prev, a, b, ccw):
                    out.append(intersect(prev, cur, a, b))
                out.append(cur)
            elif inside(prev, a, b, ccw):
                out.append(intersect(prev, cur, a, b))
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def polygon_signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def rbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """两个旋转框的 IoU（凸多边形精确裁剪，不是包络近似）。

    开发文档 §7.2 要求「计算旋转框/多边形 IoU」，用包络会把斜置的相邻楼判成重叠。
    """
    pa = rbox_corners(a)
    pb = rbox_corners(b)
    inter_poly = convex_clip(pa, pb)
    inter = polygon_area(inter_poly) if len(inter_poly) >= 3 else 0.0
    if inter <= 0:
        return 0.0
    area_a = abs(float(a[2]) * float(a[3]))
    area_b = abs(float(b[2]) * float(b[3]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------- 候选与去重


@dataclass
class Detection:
    """一个检测候选；det_id 只在去重后分配。"""

    window_id: int
    score: float
    rbox: tuple[float, float, float, float, float]
    class_name: str = "building"
    edge_truncated: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def corners(self) -> np.ndarray:
        return rbox_corners(self.rbox)

    @property
    def aabb(self) -> tuple[float, float, float, float]:
        return aabb_of(self.corners)


@dataclass
class DedupResult:
    kept: list[Detection]
    suppressed: list[tuple[Detection, int]]     # (被抑制候选, 保留它的候选的 kept 下标)


class _GridIndex:
    """均匀网格空间索引：只比较同格及邻格的候选，避免 O(N²)。"""

    def __init__(self, cell: float):
        self.cell = max(float(cell), 1e-6)
        self._cells: dict[tuple[int, int], list[int]] = {}

    def _range(self, aabb):
        x1, y1, x2, y2 = aabb
        c = self.cell
        return (int(math.floor(x1 / c)), int(math.floor(y1 / c)),
                int(math.floor(x2 / c)), int(math.floor(y2 / c)))

    def insert(self, idx: int, aabb) -> None:
        cx0, cy0, cx1, cy1 = self._range(aabb)
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                self._cells.setdefault((cx, cy), []).append(idx)

    def query(self, aabb) -> set[int]:
        cx0, cy0, cx1, cy1 = self._range(aabb)
        found: set[int] = set()
        for cy in range(cy0, cy1 + 1):
            for cx in range(cx0, cx1 + 1):
                found.update(self._cells.get((cx, cy), ()))
        return found


def dedup_detections(dets: Iterable[Detection], iou_threshold: float = 0.5,
                     prefer_edge_complete: bool = True) -> DedupResult:
    """跨窗口去重。

    * 排序确定性：score 降序，其次「边缘完整优先」，最后窗口号/坐标做稳定 tie-break。
    * 用旋转框精确 IoU，而不是包络：紧邻的斜置建筑不能因为包络相交被误抑制。
    * 不合并几何（§7.2）：去重只决定「保留哪个候选」，不 union。
    """
    items = list(dets)
    if not items:
        return DedupResult([], [])

    def sort_key(d: Detection):
        edge_rank = 0 if (prefer_edge_complete and not d.edge_truncated) else 1
        r = d.rbox
        return (-d.score, edge_rank, d.window_id, round(r[0], 6), round(r[1], 6))

    ordered = sorted(items, key=sort_key)
    max_side = max(max(d.rbox[2], d.rbox[3]) for d in ordered)
    index = _GridIndex(cell=max(max_side, 8.0) * 1.5)

    kept: list[Detection] = []
    suppressed: list[tuple[Detection, int]] = []
    kept_aabbs: list[tuple[float, float, float, float]] = []

    for det in ordered:
        aabb = det.aabb
        neighbours = index.query(aabb)
        hit = -1
        for k in neighbours:
            if aabb_iou(aabb, kept_aabbs[k]) <= 0.0:
                continue            # 包络不交，旋转框必然不交，省一次精确计算
            if rbox_iou(det.rbox, kept[k].rbox) >= iou_threshold:
                hit = k
                break
        if hit >= 0:
            suppressed.append((det, hit))
        else:
            idx = len(kept)
            kept.append(det)
            kept_aabbs.append(aabb)
            index.insert(idx, aabb)

    return DedupResult(kept, suppressed)


def assign_det_ids(kept: Sequence[Detection], prefix: str = "d") -> list[dict]:
    """给保留框分配稳定 det_id。

    ID 只由排序位置决定（同分由 sort_key 固定），因此同一输入重跑得到相同 ID，
    满足 §7.2「同分用固定规则」与恢复时的幂等要求。
    """
    out = []
    for i, det in enumerate(kept):
        out.append({
            "det_id": f"{prefix}{i:06d}",
            "det_score": round(float(det.score), 6),
            "class_name": det.class_name,
            "box_type": "obb",
            "angle_rad": round(float(det.rbox[4]), 9),
            "rbox": [round(float(v), 6) for v in det.rbox],
            "corners": [[round(float(x), 6), round(float(y), 6)] for x, y in det.corners],
            "source_window": int(det.window_id),
            "edge_truncated": bool(det.edge_truncated),
        })
    return out
