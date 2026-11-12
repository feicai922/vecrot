"""多边形几何工具：坐标序列解析、网格 ↔ 像素映射、栅格化与 IoU。

约定（与 v1 及训练保持一致，改动会静默改变精度，勿随手改）：
- 模型输出的离散网格是 **128×128**（`<x0..x127><y0..y127>`），
- 网格坐标 x 映射到像素的约定是**格心**：`px = (x + 0.5) / 128 * W`，
- 于是单点量化误差上界 = 0.5 格 = 裁剪边长的 0.391%。
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

import numpy as np

COORD_PATTERN = re.compile(r"<([xy])(\d+)>")
GRID_SIZE = 128

Point = Tuple[float, float]


def parse_polygon(text: str) -> List[Tuple[int, int]]:
    """从模型输出文本里抽出 (x, y) 网格点序列。

    只认成对的 `<xN><yN>`；末尾落单的 `<xN>` 会被丢弃（v1 行为，保持不变）。
    """
    points: List[Tuple[int, int]] = []
    pending_x = None
    for axis, raw_value in COORD_PATTERN.findall(text):
        value = int(raw_value)
        if axis == "x":
            pending_x = value
        elif pending_x is not None:
            points.append((pending_x, value))
            pending_x = None
    return points


def grid_to_pixel(
    points: Sequence[Tuple[int, int]],
    size: Tuple[int, int],
    grid_size: int = GRID_SIZE,
    offset: Tuple[float, float] = (0.0, 0.0),
) -> List[Point]:
    """网格点 → 像素坐标（`size` 是裁剪图/整图的 (W, H)；`offset` 用于贴回整图）。"""
    width, height = size
    ox, oy = offset
    return [
        ((x + 0.5) / grid_size * width + ox, (y + 0.5) / grid_size * height + oy)
        for x, y in points
    ]


def pixel_to_grid(points: Iterable[Point], size: Tuple[int, int], grid_size: int = GRID_SIZE):
    """像素坐标 → 网格点（四舍五入并夹到 [0, grid_size-1]），构造训练标签时用。"""
    width, height = size
    out = []
    for x, y in points:
        gx = int(round(x / width * grid_size - 0.5))
        gy = int(round(y / height * grid_size - 0.5))
        out.append((min(max(gx, 0), grid_size - 1), min(max(gy, 0), grid_size - 1)))
    return out


def clip_polygon(points: Iterable[Point], size: Tuple[int, int]) -> List[Point]:
    width, height = size
    return [(min(max(x, 0.0), width), min(max(y, 0.0), height)) for x, y in points]


def rasterize(polygons: Sequence[Sequence[Point]], size: Tuple[int, int], scale: int = 1) -> np.ndarray:
    """把一组多边形栅格化成 int32 标签图（0 = 背景，i+1 = 第 i 个多边形）。

    `scale` 只用于加速（>=1 时先缩小再画），做 IoU 时请保持 scale=1。
    """
    from PIL import Image, ImageDraw

    width, height = size
    if scale != 1:
        width, height = max(1, width // scale), max(1, height // scale)
    canvas = Image.new("I", (width, height), 0)
    drawer = ImageDraw.Draw(canvas)
    for index, polygon in enumerate(polygons):
        if len(polygon) < 3:
            continue
        coords = [(float(x) / scale, float(y) / scale) for x, y in polygon]
        drawer.polygon(coords, fill=index + 1)
    return np.array(canvas, dtype=np.int32)


def rasterize_masks(
    polygons: Sequence[Sequence[Point]], size: Tuple[int, int]
) -> np.ndarray:
    """每个多边形一张**独立的** bool 掩膜，返回 (n, H, W)。

    为什么不用 `rasterize()` 的标签图算 IoU：标签图里后画的多边形会**覆盖**先画的
    （一张图只能存一个整数标签），当两个多边形重叠时，先画的那个会丢掉重叠区，
    算出来的 IoU 会偏小、"粘连/吞并"也会被低估。这里逐个实例独立栅格化，避免该问题。
    """
    from PIL import Image, ImageDraw

    width, height = size
    masks = np.zeros((len(polygons), height, width), dtype=bool)
    for index, polygon in enumerate(polygons):
        if len(polygon) < 3:
            continue
        canvas = Image.new("1", (width, height), 0)
        ImageDraw.Draw(canvas).polygon([(float(x), float(y)) for x, y in polygon], fill=1)
        masks[index] = np.array(canvas, dtype=bool)
    return masks


def iou_matrix_masks(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    """由两组独立掩膜算 pred×gt 的 IoU 矩阵（不会被"后画覆盖先画"污染）。

    实现上用"按 GT 的像素索引做批量求和"，单图 40×40 个实例也只需要几毫秒；
    早期的双重 for + 逐像素 `logical_and` 在 1024² 瓦片上会慢上千倍。
    """
    n_pred = int(pred_masks.shape[0])
    n_gt = int(gt_masks.shape[0])
    iou = np.zeros((n_gt, n_pred), dtype=np.float64)
    if n_pred == 0 or n_gt == 0:
        return iou
    flat_pred = pred_masks.reshape(n_pred, -1)
    flat_gt = gt_masks.reshape(n_gt, -1)
    pred_area = flat_pred.sum(axis=1).astype(np.float64)
    gt_area = flat_gt.sum(axis=1).astype(np.float64)
    for gi in range(n_gt):
        if gt_area[gi] == 0:
            continue
        selector = flat_gt[gi]
        inter = flat_pred[:, selector].sum(axis=1).astype(np.float64)
        union = gt_area[gi] + pred_area - inter
        iou[gi] = np.where(union > 0, inter / np.maximum(union, 1.0), 0.0)
    return iou


def iou_matrix(
    pred_labels: np.ndarray, gt_labels: np.ndarray, n_pred: int, n_gt: int
) -> np.ndarray:
    """由两张标签图算出 pred×gt 的 IoU 矩阵（O(像素数)，无逐对重画）。"""
    if pred_labels.shape != gt_labels.shape:
        raise ValueError("label maps must share shape")
    pair = gt_labels.astype(np.int64) * (n_pred + 1) + pred_labels.astype(np.int64)
    counts = np.bincount(pair.ravel(), minlength=(n_gt + 1) * (n_pred + 1))
    inter = counts.reshape(n_gt + 1, n_pred + 1)[1:, 1:]
    pred_area = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)[1:]
    gt_area = np.bincount(gt_labels.ravel(), minlength=n_gt + 1)[1:]
    union = pred_area[None, :] + gt_area[:, None] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return iou


def match_greedy(iou: np.ndarray, threshold: float = 0.0) -> List[Tuple[int, int]]:
    """按 IoU 从大到小贪心一对一匹配，返回 [(gt_idx, pred_idx), ...]。"""
    matches: List[Tuple[int, int]] = []
    if iou.size == 0:
        return matches
    work = iou.copy()
    while True:
        flat = int(np.argmax(work))
        gi, pi = divmod(flat, work.shape[1])
        if work[gi, pi] <= threshold:
            break
        matches.append((gi, pi))
        work[gi, :] = -1.0
        work[:, pi] = -1.0
    return matches


def polygon_iou(poly_a: Sequence[Point], poly_b: Sequence[Point]) -> float:
    """单对多边形的 rasterized IoU（按两多边形联合外接框裁窗口，避免整图栅格化）。"""
    if len(poly_a) < 3 or len(poly_b) < 3:
        return 0.0
    xs = [p[0] for p in poly_a] + [p[0] for p in poly_b]
    ys = [p[1] for p in poly_a] + [p[1] for p in poly_b]
    x0, y0 = float(np.floor(min(xs))), float(np.floor(min(ys)))
    x1, y1 = float(np.ceil(max(xs))), float(np.ceil(max(ys)))
    size = (int(max(x1 - x0, 1)) + 1, int(max(y1 - y0, 1)) + 1)
    shift = lambda poly: [(x - x0, y - y0) for x, y in poly]  # noqa: E731
    masks = rasterize_masks([shift(poly_a), shift(poly_b)], size)
    area_a, area_b = masks[0], masks[1]
    union = int((area_a | area_b).sum())
    if union == 0:
        return 0.0
    return float((area_a & area_b).sum()) / union

# --------------------------------------------------------------------------- 自适应裁剪外扩

ADAPTIVE_EXPAND_TABLE = (
    # (短边上限 px, 外扩比)  —— 实测标定，见 tools/eval_adaptive_expand.py
    (32.0, 1.2),    # 极小楼：放大反而被邻居干扰
    (96.0, 1.4),    # 中等尺寸：1.4 最稳（48-64px 桶 +6.3pt pass@0.75）
    (float("inf"), 1.1),  # 大楼：收紧，1.6 时 p@0.75 会掉到 0.22
)


def adaptive_expand_ratio(short_side_px: float) -> float:
    """按建筑短边（像素）选裁剪外扩比。

    为什么固定 1.2 不是最优：模型输出尺寸**锚在裁剪框**上（预测占裁面积恒 0.27~0.39，
    与楼在框里多大无关），所以"框松紧"直接决定画多大。WHU val 240 栋实测（pass@0.75）：

        | 短边   | 1.1  | 1.2  | 1.4  | 1.6  | 最优 |
        | <32px  | .650 | .730 | .730 | .590 | 1.2 |
        | 32-48  | .385 | .615 | .692 | .462 | 1.4 |
        | 48-64  | .812 | .854 | .917 | .792 | 1.4 |
        | 64-96  | .791 | .814 | .837 | .698 | 1.4 |
        | >96px  | .652 | .652 | .609 | .217 | 1.1 |

    整体：固定 1.2 → p@0.75 0.750；按本表自适应 → **0.775（+2.5pt）**，且零训练成本。
    """
    for upper, ratio in ADAPTIVE_EXPAND_TABLE:
        if short_side_px < upper:
            return ratio
    return ADAPTIVE_EXPAND_TABLE[-1][1]


def polygon_short_side(points) -> float:
    """多边形的短边（用外接框短边近似，像素）。"""
    if not points:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return float(min(max(xs) - min(xs), max(ys) - min(ys)))
