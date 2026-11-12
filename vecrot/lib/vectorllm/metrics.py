"""边界级 / 顶点级指标：BIoU、Boundary F1、顶点距离、顶点数偏差。

为什么需要（见 `docs/exp_box_tightness_and_abstention.md` 的结论）：
现有体系只有 IoU / pass@t。但 IoU 对两类完全不同的错误给出的分数可以一样 ——
**整体偏移 2px 的轮廓**（边界问题，可用吸附/精修救）和**少画一个角/多画一个凹口**
（结构问题，只能靠顶点数/拓扑救）。要区分它们、并验收"边界精修"与"顶点数预测"，
必须有边界带与顶点级指标。

口径约定（全部在**同一瓦片像素坐标系**上算，与本仓库 `geometry.py` 一致）：
- `iou`：栅格化 IoU（用于和现有表格对拍）；
- `biou(δ)`：边界带 IoU（Cheng et al., CVPR'21 的简化版）。带 = 掩膜膨胀 δ 减去腐蚀 δ，
  `BIoU = |Bp ∩ Bg| / |Bp ∪ Bg|`。δ 用**像素**（默认 2）：本数据建筑中位长边只有 ~64px，
  按"图像对角线 2%"取 δ 会得到 ~29px，等于把整栋楼都算成边界，指标失效；
- `boundary_f1(tol)`：1px 内边缘的精确率/召回率（F1 用距离变换算，tol 单位像素）；
- 顶点指标：顶点数 pred/gt、`|Δn|`、双向最近顶点距离（px，并用 sqrt(面积) 归一化）。
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


# --------------------------------------------------------------------------- 基础工具


def _ring(points: Sequence[Point]) -> np.ndarray:
    """去掉闭合重复点，返回 (n, 2) float 数组。"""
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(arr) >= 2 and np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    return arr


def _window(pred: Sequence[Point], gt: Sequence[Point], pad: int) -> Tuple[int, int, int, int]:
    pts = np.vstack([_ring(pred), _ring(gt)])
    x0 = int(np.floor(pts[:, 0].min())) - pad
    y0 = int(np.floor(pts[:, 1].min())) - pad
    x1 = int(np.ceil(pts[:, 0].max())) + pad
    y1 = int(np.ceil(pts[:, 1].max())) + pad
    return x0, y0, max(x1 - x0, 2), max(y1 - y0, 2)


def _mask(points: Sequence[Point], window: Tuple[int, int, int, int]) -> np.ndarray:
    """在给定窗口里栅格化一个多边形（窗口外自然被裁掉）。"""
    from PIL import Image, ImageDraw

    x0, y0, w, h = window
    ring = _ring(points)
    canvas = Image.new("1", (w, h), 0)
    if len(ring) >= 3:
        ImageDraw.Draw(canvas).polygon([(float(x) - x0, float(y) - y0) for x, y in ring], fill=1)
    return np.array(canvas, dtype=bool)


def _band(mask: np.ndarray, delta: int) -> np.ndarray:
    """边界带：膨胀 δ 减腐蚀 δ。δ 太大时腐蚀会吃掉整个小建筑，此时退化为"整块掩膜"。"""
    from scipy import ndimage

    if not mask.any():
        return np.zeros_like(mask)
    dil = ndimage.binary_dilation(mask, iterations=delta)
    ero = ndimage.binary_erosion(mask, iterations=delta)
    band = dil & ~ero
    if not band.any():
        band = mask
    return band


def _edges(mask: np.ndarray) -> np.ndarray:
    """1px 内边缘。"""
    from scipy import ndimage

    if not mask.any():
        return np.zeros_like(mask)
    return mask & ~ndimage.binary_erosion(mask, iterations=1)


# --------------------------------------------------------------------------- 单对指标


def pair_iou(pred: Sequence[Point], gt: Sequence[Point]) -> float:
    if len(_ring(pred)) < 3 or len(_ring(gt)) < 3:
        return 0.0
    win = _window(pred, gt, 1)
    mp, mg = _mask(pred, win), _mask(gt, win)
    union = int((mp | mg).sum())
    return float((mp & mg).sum()) / union if union else 0.0


def pair_biou(pred: Sequence[Point], gt: Sequence[Point], delta: int = 2) -> float:
    if len(_ring(pred)) < 3 or len(_ring(gt)) < 3:
        return 0.0
    win = _window(pred, gt, delta + 1)
    bp, bg = _band(_mask(pred, win), delta), _band(_mask(gt, win), delta)
    union = int((bp | bg).sum())
    return float((bp & bg).sum()) / union if union else 0.0


def pair_boundary_f1(pred: Sequence[Point], gt: Sequence[Point], tol: float = 1.5) -> Tuple[float, float]:
    """返回 (F1, 平均边缘距离px)。边缘像素在 tol 内即算命中（距离变换实现）。"""
    from scipy import ndimage

    if len(_ring(pred)) < 3 or len(_ring(gt)) < 3:
        return 0.0, float("nan")
    win = _window(pred, gt, int(np.ceil(tol)) + 2)
    ep, eg = _edges(_mask(pred, win)), _edges(_mask(gt, win))
    if not ep.any() or not eg.any():
        return 0.0, float("nan")
    d_to_g = ndimage.distance_transform_edt(~eg)
    d_to_p = ndimage.distance_transform_edt(~ep)
    precision = float((d_to_g[ep] <= tol).mean())
    recall = float((d_to_p[eg] <= tol).mean())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_dist = float(0.5 * (d_to_g[ep].mean() + d_to_p[eg].mean()))
    return f1, mean_dist


def vertex_stats(pred: Sequence[Point], gt: Sequence[Point]) -> dict:
    """顶点数偏差 + 双向最近顶点距离（px 与按 sqrt(面积) 归一化）。"""
    rp, rg = _ring(pred), _ring(gt)
    out = {"n_pred": int(len(rp)), "n_gt": int(len(rg)), "dn": int(len(rp) - len(rg))}
    if len(rp) == 0 or len(rg) == 0:
        out.update(vdist_mean=float("nan"), vdist_med=float("nan"), vdist_max=float("nan"))
        return out

    def _nn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        return d.min(axis=1)

    d_p2g, d_g2p = _nn(rp, rg), _nn(rg, rp)
    both = np.concatenate([d_p2g, d_g2p])
    scale = max(float(np.sqrt(max(_ring_area(rg), 1.0))), 1.0)
    out.update(
        vdist_mean=float(both.mean()),
        vdist_med=float(np.median(both)),
        vdist_max=float(both.max()),
        vdist_norm_mean=float(both.mean() / scale),
    )
    return out


def _ring_area(ring: np.ndarray) -> float:
    """鞋带公式面积（像素²）。"""
    if len(ring) < 3:
        return 0.0
    x, y = ring[:, 0], ring[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_valid(pred: Sequence[Point]) -> Tuple[bool, bool]:
    """返回 (几何有效, 自交)。用 shapely；不可用时按 (True, False) 放过。"""
    ring = _ring(pred)
    if len(ring) < 3:
        return False, False
    try:
        from shapely.geometry import Polygon
    except Exception:
        return True, False
    poly = Polygon(ring)
    # 自交 = 环不简单（"8" 字形/蝴蝶结）；它与"有效"高度重合但不是同义：重复顶点也可能有效但退化。
    return bool(poly.is_valid), bool(not poly.is_simple)


def eval_pair(pred: Sequence[Point], gt: Sequence[Point], biou_deltas=(1, 2, 4), f1_tol: float = 1.5) -> dict:
    """一次算齐一对 (pred, gt) 的全部边界/顶点指标。"""
    row = {"iou": pair_iou(pred, gt)}
    for d in biou_deltas:
        row[f"biou@{d}"] = pair_biou(pred, gt, d)
    f1, dist = pair_boundary_f1(pred, gt, f1_tol)
    row["bf1"] = f1
    row["bdist"] = dist
    row.update(vertex_stats(pred, gt))
    valid, selfx = polygon_valid(pred)
    row["valid"], row["selfx"] = valid, selfx
    return row
