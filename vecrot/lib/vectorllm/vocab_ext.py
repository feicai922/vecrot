"""词表扩展：顶点数 token `<nv3>`…`<nv64>`（"`<n=K>`"）。

动机（本仓库实测）：
- 模型输出顶点数**几乎恒定**（gzda_v0 上 pred 中位 4，GT 中位 6~15；27% 的 GT 超过 120 点）；
- `|Δn|` 与 IoU 强相关：Δn=0 → 0.72，|Δn|≥4 → 0.60；
- 自回归长度没有"显式意图"：模型只能靠隐式的停止时机决定画几个角，误差会累积。

做法：目标序列改成 `<nvK><x..><y..>…`（K = 顶点数）。这样
1. 顶点数变成**可监督、可评测**的量（一条 token 的 CE）；
2. 解码期可以**按 K 约束**（`hub.VertexCountProcessor`）：满 2K 个坐标就强制结束、不足就禁止结束；
3. 我们可以在不同 K 上做受控实验（同一条裁片强制 4/8/16 个顶点），把"顶点数"与"边界精度"解耦。

注意：这些 token 必须与 `<no_object>` 一样作为 **special token** 加进 tokenizer，
并且 `resize_token_embeddings` **之后**重绑 tied `lm_head`（见 `tools/train_sft.py`）。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

MIN_VERTICES = 3
MAX_VERTICES = 64

NVERTEX_TOKENS: List[str] = [f"<nv{k}>" for k in range(MIN_VERTICES, MAX_VERTICES + 1)]
NVERTEX_TOKEN_SET = set(NVERTEX_TOKENS)
_NV_PATTERN = re.compile(r"<nv(\d+)>")


def vertex_token(k: int) -> str:
    """顶点数 → token；超出 [MIN, MAX] 时夹到边界。"""
    k = int(min(max(k, MIN_VERTICES), MAX_VERTICES))
    return f"<nv{k}>"


def parse_vertex_count(text: str) -> Optional[int]:
    """从生成文本里读出 `<nvK>`（没有则 None）。"""
    m = _NV_PATTERN.search(text or "")
    return int(m.group(1)) if m else None


def strip_vertex_token(text: str) -> str:
    return _NV_PATTERN.sub("", text or "")


def format_target(points: Sequence[Tuple[int, int]], with_count: bool = True) -> str:
    """网格点序列 → 目标串（`<nvK>` + 逐点 `<xN><yN>`）。"""
    body = "".join(f"<x{gx}><y{gy}>" for gx, gy in points)
    return f"{vertex_token(len(points))}{body}" if with_count else body


def ensure_vertex_tokens(tokenizer, model=None) -> Dict[str, int]:
    """幂等地把 `<nv*>` 加进 tokenizer（以及 model 的 embedding / tied lm_head）。

    返回 {"added": n_new_tokens}。与 `train_sft.py` 里 `<no_object>` 的处理保持同一套约定：
    新增 embedding 用已有特殊 token 的均值热启动（均值初始化 + 大量步数学不会，见 HANDOFF §3.3）。
    """
    import torch

    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [t for t in NVERTEX_TOKENS if t not in existing]
    if not new_tokens:
        return {"added": 0}
    tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    if model is None:
        return {"added": len(new_tokens)}

    old_vocab = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    emb = model.get_input_embeddings().weight
    new_ids = [tokenizer.convert_tokens_to_ids(t) for t in new_tokens]
    with torch.no_grad():
        # 热启动：用"坐标 token + 结束 token"的均值，让新 token 一开始就有合理的量级
        ref_ids = []
        for v in (0, 1, 64, 127):
            for t in (f"<x{v}>", f"<y{v}>"):
                i = tokenizer.convert_tokens_to_ids(t)
                if isinstance(i, int) and 0 <= i < old_vocab:
                    ref_ids.append(i)
        for t in ("<|im_end|>", "<no_object>"):
            i = tokenizer.convert_tokens_to_ids(t)
            if isinstance(i, int) and 0 <= i < old_vocab:
                ref_ids.append(i)
        vec = emb[ref_ids].mean(dim=0) if ref_ids else emb[:old_vocab].mean(dim=0)
        for i in new_ids:
            emb[i] = vec
    _rebind_tied_lm_head(model)
    return {"added": len(new_tokens)}


def _rebind_tied_lm_head(model) -> None:
    """`resize_token_embeddings` 会新建 nn.Embedding，tie 必须重做（否则 lm_head 还是旧的、尺寸不匹配）。"""
    try:
        out_emb = model.get_output_embeddings()
        in_emb = model.get_input_embeddings()
        if out_emb is not None:
            out_emb.weight = in_emb.weight
    except Exception:
        pass
