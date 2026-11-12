"""弃权（abstention）：让流水线知道"这张裁剪里可能没有建筑 / 这次生成不可信"。

实测动机（见 `docs/exp_box_tightness_and_abstention.md`）：
给 300 张**确认没有建筑**的空瓦片裁剪，当前 checkpoint **100% 仍然画出一个完整多边形**，
面积占画面 0.350（真实建筑是 0.364），顶点数还更多（7 vs 5）。
也就是说：**模型本身没有弃权能力**，检测阶段的误检框会被 100% 变成"凭空捏造的建筑"。

两条路，本模块都提供：

1. **零训练代理信号**（今天就能用）：`polygon_signals()` 从输出几何里算可解释特征，
   `abstain_score()` 合成一个分数，`best_threshold()` 用带标注的正/负样本拟合阈值。
   最强的一路通常是 **K 次采样的自一致性**（`consistency_from_samples()`）：
   真建筑上多样本应收敛到同一条轮廓，空地上会乱飘。
2. **训练出真正的弃权 token**（需要训练回路）：输出语法扩展一个 `<no_object>`，
   `tools/build_abstain_dataset.py` 已经能直接把训练数据造好（负样本零标注成本）。
   推理端用 `parse_output()` 识别它。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import parse_polygon, rasterize

Point = Tuple[float, float]
Polygon = Sequence[Point]

NO_OBJECT_TOKEN = "<no_object>"

# 实测先验：模型输出面积占裁剪面积的"塌缩值"（1.2 外扩下中位数 0.364，跨 4 档外扩都在 0.27~0.39）
DEFAULT_FILL_PRIOR = 0.36


def polygon_area(points: Polygon) -> float:
    if len(points) < 3:
        return 0.0
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    return float(abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))) / 2.0)


def is_self_intersecting(points: Polygon) -> bool:
    try:
        from shapely.geometry import LinearRing
    except Exception:  # pragma: no cover
        return False
    if len(points) < 4:
        return False
    try:
        return not LinearRing(points).is_simple
    except Exception:
        return True


def polygon_signals(
    polygon: Polygon,
    crop_size: Tuple[int, int],
    prior_fill: float = DEFAULT_FILL_PRIOR,
) -> Dict[str, float]:
    """单个输出多边形的"可信度"特征（全部无需标签）。

    | 特征 | 含义 / 为什么有用 |
    |---|---|
    | `fill_ratio` | 多边形面积 / 裁剪面积。模型会塌缩到 ~0.36，若明显偏离说明它"没在对的地方画" |
    | `fill_dev` | 与塌缩先验的偏离（归一化）。太大 ⇒ 可能吞了邻居或画小了 |
    | `n_vertices` | 顶点数。空地上反而更多（7 vs 5），是个弱但有方向性的信号 |
    | `compactness` | 面积 / 外接框面积。过低 ⇒ 形状碎裂/毛刺 |
    | `self_intersect` | 自交（拓扑无效） |
    | `aspect` | 外接框长宽比。远大于 GT 分布（本数据集 >3 只占 1.7%）时可疑 |
    """
    width, height = crop_size
    area = polygon_area(polygon)
    fill = area / max(width * height, 1.0)
    if len(polygon) >= 3:
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        bw, bh = max(xs) - min(xs), max(ys) - min(ys)
        compactness = area / max(bw * bh, 1.0)
        aspect = max(bw, bh) / max(min(bw, bh), 1e-6)
    else:
        compactness, aspect = 0.0, 1.0
    return {
        "fill_ratio": float(fill),
        "fill_dev": float(abs(fill - prior_fill) / prior_fill),
        "n_vertices": float(len(polygon)),
        "compactness": float(compactness),
        "self_intersect": float(bool(is_self_intersecting(polygon))),
        "aspect": float(aspect),
        "degenerate": float(len(polygon) < 3 or area <= 1.0),
    }


def consistency_from_samples(
    sample_polygons: Sequence[Polygon], size: Tuple[int, int]
) -> Dict[str, float]:
    """K 次采样（温度采样）的两两 IoU 均值——最有效的零训练不确定性信号。

    真建筑：多样本收敛到同一条轮廓（一致性高）；
    空地/遮挡严重：每次画的地方都不一样（一致性低）。
    """
    valid = [poly for poly in sample_polygons if len(poly) >= 3]
    if len(valid) < 2:
        return {"consistency": 0.0, "n_samples": float(len(sample_polygons)), "fill_std": 0.0}
    labels = rasterize(valid, size)
    n = len(valid)
    pair_ious: List[float] = []
    areas = []
    for i in range(n):
        mask_i = labels == i + 1
        areas.append(mask_i.sum())
        for j in range(i + 1, n):
            mask_j = labels == j + 1
            inter = np.logical_and(mask_i, mask_j).sum()
            union = np.logical_or(mask_i, mask_j).sum()
            pair_ious.append(float(inter / max(union, 1)))
    fill = np.array(areas, dtype=np.float64) / max(size[0] * size[1], 1.0)
    return {
        "consistency": float(np.mean(pair_ious)),
        "n_samples": float(len(sample_polygons)),
        "fill_std": float(fill.std()),
    }


# 默认权重：先验塌缩 + 自一致性为主，其余作辅助。
DEFAULT_WEIGHTS = {
    "fill_dev": 1.0,
    "self_intersect": 2.0,
    "degenerate": 5.0,
    "consistency": -1.5,   # 一致性越高越可信 ⇒ 负权重
    "vertex_excess": 1.0,
}


def abstain_score(signals: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """把特征合成为"越像非建筑/越不可信，分数越高"的单一分数（>阈值即弃权）。"""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    vertex_excess = max(0.0, signals.get("n_vertices", 5.0) - 5.0) / 5.0
    features = {
        "fill_dev": signals.get("fill_dev", 0.0),
        "self_intersect": signals.get("self_intersect", 0.0),
        "degenerate": signals.get("degenerate", 0.0),
        "consistency": signals.get("consistency", 1.0),
        "vertex_excess": vertex_excess,
    }
    return float(sum(weights[name] * value for name, value in features.items()))


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC（labels=1 表示"应弃权"的负样本/不可信样本）。用秩公式，无第三方依赖。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 处理并列：取平均秩
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.mean(np.arange(i + 1, j + 2))
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def best_threshold(scores: Sequence[float], labels: Sequence[int]) -> Dict[str, float]:
    """用 Youden's J 选阈值，返回阈值 + 该点的 TPR/FPR + AUC。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    candidates = np.unique(scores)
    best = {"threshold": float("nan"), "tpr": 0.0, "fpr": 0.0, "j": -1.0, "auroc": auroc(scores, labels)}
    for thr in candidates:
        pred = scores >= thr
        tpr = float((pred & (labels == 1)).sum() / max((labels == 1).sum(), 1))
        fpr = float((pred & (labels == 0)).sum() / max((labels == 0).sum(), 1))
        j = tpr - fpr
        if j > best["j"]:
            best = {"threshold": float(thr), "tpr": tpr, "fpr": fpr, "j": j, "auroc": best["auroc"]}
    return best


def parse_output(text: str, grid_size: int = 128) -> Dict[str, object]:
    """解析模型输出：弃权 token / 有效多边形 / 无效。

    当前 checkpoint **不会**输出 `<no_object>`（训练后才会有）；
    在这个假设落地前，`kind` 只可能是 `"polygon"` 或 `"invalid"`。
    """
    if NO_OBJECT_TOKEN in text:
        return {"kind": "no_object", "grid_polygon": [], "n_vertices": 0}
    grid_polygon = parse_polygon(text)
    if len(grid_polygon) < 3:
        return {"kind": "invalid", "grid_polygon": grid_polygon, "n_vertices": len(grid_polygon)}
    return {"kind": "polygon", "grid_polygon": grid_polygon, "n_vertices": len(grid_polygon)}
