"""邻居轮廓文本条件（NeighborContourPrompt，docs/21 方向 A）。

把已完成邻居的多边形翻译成**当前裁剪框 128 网格坐标 token 序列**，拼进文本 prompt。
核心原则（GF 三轮失败的教训）：上下文不用语义特征、不用注入层——用模型自己的
坐标语言（与输出同词表、同坐标系），借 Qwen 预训练的 in-context 序列能力进入。

几何链路：
  邻居多边形(瓦片坐标) --tile_to_crop--> 裁剪框坐标 --clip_poly--> 框内部分
  --/W*128--> 128 网格 --> `<xN><yN>` token 序列 --> prompt
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from .rbox import tile_to_crop

GRID = 128


def clip_poly_to_box(poly: Sequence[Tuple[float, float]],
                     w: float, h: float) -> List[Tuple[float, float]]:
    """Sutherland–Hodgman 多边形-矩形裁剪：只留框内部分（框外顶点丢弃）。"""
    def inside(p, edge):
        x, y = p
        if edge == 0: return x >= 0
        if edge == 1: return x <= w
        if edge == 2: return y >= 0
        return y <= h
    def intersect(p1, p2, edge):
        (x1, y1), (x2, y2) = p1, p2
        if edge == 0:
            t = (0 - x1) / (x2 - x1) if x2 != x1 else 0
            return (0.0, y1 + t * (y2 - y1))
        if edge == 1:
            t = (w - x1) / (x2 - x1) if x2 != x1 else 0
            return (w, y1 + t * (y2 - y1))
        if edge == 2:
            t = (0 - y1) / (y2 - y1) if y2 != y1 else 0
            return (x1 + t * (x2 - x1), 0.0)
        t = (h - y1) / (y2 - y1) if y2 != y1 else 0
        return (x1 + t * (x2 - x1), h)

    pts = list(poly)
    if len(pts) < 3:
        return []
    for edge in range(4):
        out = []
        for i in range(len(pts)):
            cur, prev = pts[i], pts[i - 1]
            if inside(cur, edge):
                if not inside(prev, edge):
                    out.append(intersect(prev, cur, edge))
                out.append(cur)
            elif inside(prev, edge):
                out.append(intersect(prev, cur, edge))
        pts = out
        if not pts:
            return []
    return pts


def neighbor_grid_tokens(neighbor_poly: Sequence[Tuple[float, float]],
                         transform: dict, max_side: int = 8) -> str:
    """邻居多边形 → 当前裁剪框 128 网格 token 串。框外/退化返回 ''（该邻居跳过）。"""
    if not transform.get("ok") or len(neighbor_poly) < 3:
        return ""
    W, H = transform["size"]
    crop = tile_to_crop(neighbor_poly, transform)
    clipped = clip_poly_to_box(crop, W, H)
    if len(clipped) < 3:
        return ""
    # 降采样顶点（邻居只需要形状级精度，省 token）
    if len(clipped) > max_side:
        step = len(clipped) / max_side
        clipped = [clipped[int(i * step)] for i in range(max_side)]
    toks = []
    for u, v in clipped:
        x = min(max(int(round(u / W * GRID)), 0), GRID - 1)
        y = min(max(int(round(v / H * GRID)), 0), GRID - 1)
        toks.append(f"<x{x}><y{y}>")
    return "".join(toks)


def neighbors_prompt_block(neighbors: Sequence[str]) -> str:
    """K 个邻居 token 串 → prompt 文本块。空列表返回 ''（保持原 prompt 不变）。"""
    valid = [t for t in neighbors if t]
    if not valid:
        return ""
    lines = ["Adjacent buildings already extracted:"]
    for i, t in enumerate(valid, 1):
        lines.append(f"{i}: {t}")
    return "\n".join(lines) + "\n"


def build_neighbor_context(done_polys: Sequence[Sequence[Tuple[float, float]]],
                           transform: dict, max_neighbors: int = 3,
                           max_side: int = 8) -> str:
    """已完成多边形（瓦片坐标）→ 拼好的 prompt 上下文块。"""
    if not done_polys or not transform.get("ok"):
        return ""
    toks = [neighbor_grid_tokens(p, transform, max_side=max_side)
            for p in done_polys[:max_neighbors]]
    return neighbors_prompt_block(toks)
