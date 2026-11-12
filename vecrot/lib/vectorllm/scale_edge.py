"""**尺度条件注入（小目标/尺寸自适应）+ 边界&角点引导增强 + 多尺度放大融合**。

三个模块都只作用在**视觉 token 序列 `[B,64,1024]`** 上，输出 token 数与维度不变 →
projector / ParaX / Qwen3 / 坐标词表**一行都不用改**，与既有 ckpt 完全兼容（默认全关）。

方向 → 模块 → 针对的实测病灶（`docs/exp_boundary_metrics_and_label_quality.md`、
`docs/exp_box_tightness_and_abstention.md`）：

| 方向 | 模块 | 病灶（本仓库实测） |
|---|---|---|
| 小目标增强 / 尺寸自适应 | `ScaleCond`（FiLM） | 预测面积占裁面积恒定 0.27~0.39，**与楼在框里多大无关** → 小楼画大、大楼画小；短边 <32px 的桶 p@0.75 只有 0.66 |
| 边界增强 | `EdgeCornerGuide` | 128² 输入把屋顶高频抹掉；**顶点**是预测目标却从来没有直接监督 |
| 多尺度 / 动态融合 | `ZoomFusion` | 主裁剪必须留邻居上下文（收紧到 1.1 极小楼 p@0.75 从 .730 掉到 .650），所以分辨率不能靠"收紧主裁剪"拿，只能另开一条放大分支并**门控**融回来 |

一个关键的工程改进（相对 `global_fusion.py` 里 ls=0 的写法）
-------------------------------------------------------------
旧做法把 `layer_scale` 零初始化 → step0 与基线逐位一致，但 **`∂L/∂(q,k,v,gate) ≡ 0`**，
分支要等 ls 自己长大才开始学（文档已记这个坑：step200 前后几乎没动）。

本文件一律改成 **"零初始化分支的输出投影"**（`out.weight = 0` / FiLM 头零初始化）：

    y = x + W_out(...)          W_out 零初始化

于是 step0 仍是恒等（`y ≡ x`，与基线逐位一致），但 `∂L/∂W_out ≠ 0` → **第一步就在学**，
`ls` 直接取 1.0。这是 ViT-Adapter / DiT / Flamingo 里"zero-init 残差输出"的标准做法。

用法（训练侧）::

    --scale-cond --edge-corner          # 本文件 §1 + §2
    --zoom-fusion                       # 本文件 §3（多一次视觉塔前向）

（`global_fusion.py` 里的 `GlobalFusion` / `HRBoundaryEnhance` 保持原样，老 ckpt 仍可加载。）
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .global_fusion import (
    EDGE_SIZE,
    HR_SIZE,
    POS_DIM,
    BoundaryTokenSampler,
    HRBoundaryEnhance,
    HREdgeEncoder,
    hr_tensor_from_crop,
)

N_TOKENS = 64          # RADIO 8×8 patch token
GRID_TOKENS = 8


# ===========================================================================
# 尺度编码（ScaleCond / ZoomFusion 共用）
# ===========================================================================


def density_code(obj_box: Sequence[float], neighbor_boxes: Sequence[Sequence[float]],
                 dim_extra: int = 16) -> torch.Tensor:
    """邻域布局密度编码（LANet layout density prior 移植，docs/21 方向 D）。

    两个标量（都在 0~1）：
        overlap  邻居框与本目标外接框的平均 IoU（密集联排 → 高）
        near     最近邻中心距 / 本目标框对角线（贴墙 → 低）

    `neighbor_boxes`: 瓦片坐标 [x1,y1,x2,y2] 列表（推理期用同图检测框即可，不依赖 GT）。
    返回 `[dim_extra]` Fourier 编码（2 标量 × 4 频率 × sin/cos）。
    """
    ox1, oy1, ox2, oy2 = [float(v) for v in obj_box]
    oa = max((ox2 - ox1) * (oy2 - oy1), 1e-6)
    diag = math.hypot(ox2 - ox1, oy2 - oy1)
    ious, dists = [], []
    for nb in neighbor_boxes:
        nx1, ny1, nx2, ny2 = [float(v) for v in nb[:4]]
        ix1, iy1 = max(ox1, nx1), max(oy1, ny1)
        ix2, iy2 = min(ox2, nx2), min(oy2, ny2)
        iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
        inter = iw * ih
        if inter > 0:
            ious.append(inter / (oa + (nx2 - nx1) * (ny2 - ny1) - inter))
        cx, cy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
        dists.append(math.hypot(cx - (ox1 + ox2) / 2, cy - (oy1 + oy2) / 2))
    overlap = sum(ious) / len(ious) if ious else 0.0
    near = min((d / diag for d in dists), default=2.0) / 2.0
    vals = [min(overlap, 1.0), min(near, 1.0)]
    n_freq = max(1, dim_extra // (2 * len(vals)))
    out = []
    for v in vals:
        for k in range(n_freq):
            f = 2.0 ** k * math.pi
            out += [torch.sin(torch.tensor(f * v)), torch.cos(torch.tensor(f * v))]
    code = torch.stack(out)
    if code.numel() < dim_extra:
        code = F.pad(code, (0, dim_extra - code.numel()))
    return code[:dim_extra]


def scale_code(obj_box: Sequence[float], crop_box: Sequence[float],
               dim: int = POS_DIM) -> torch.Tensor:
    """目标框 vs 裁剪框 → `[dim]` 的 Fourier 尺度编码。

    四个标量（都在 0~1，且**都是"目标相对裁剪框"的量**，正是"输出锚在裁剪框"要的东西）：

        w_c/cw     目标宽 / 裁剪宽
        h_c/ch     目标高 / 裁剪高
        area_ratio 目标面积 / 裁剪面积   ← 直接对应"预测面积占画幅比例"
        aspect     min(w,h)/max(w,h)   ← 细长楼（>3:1）p@0.75 只有 0.47

    每个标量做 8 个频率的 sin/cos → 4 × 8 × 2 = 64 维。
    """
    ox1, oy1, ox2, oy2 = [float(v) for v in obj_box]
    cx1, cy1, cx2, cy2 = [float(v) for v in crop_box]
    cw = max(cx2 - cx1, 1e-6)
    ch = max(cy2 - cy1, 1e-6)
    ow = max(ox2 - ox1, 1e-6)
    oh = max(oy2 - oy1, 1e-6)
    vals = [min(ow / cw, 1.0), min(oh / ch, 1.0),
            min(ow * oh / (cw * ch), 1.0),
            min(ow, oh) / max(ow, oh)]
    n_freq = max(1, dim // (2 * len(vals)))
    out = []
    for v in vals:
        for k in range(n_freq):
            f = 2.0 ** k * math.pi
            out += [torch.sin(torch.tensor(f * v)), torch.cos(torch.tensor(f * v))]
    code = torch.stack(out)                    # 4*2*n_freq
    if code.numel() < dim:
        code = F.pad(code, (0, dim - code.numel()))
    return code[:dim]


def bounded_ls(raw: torch.Tensor, ls_max: Optional[float]) -> torch.Tensor:
    """**有界 layer-scale**：`ls = ls_max · σ(raw)`（`ls_max=None` 时退回旧的 raw 值）。

    为什么需要：纯模块微调时 `ls` 一旦自己长大，模块就能把 token 任意推离冻结主干熟悉的分布
    —— 实测（`docs/SUMMARY_SCALE_EDGE_2026-11-03.md` §5.1）会让 20% 的样本退化成
    "同点重复、不发 EOS"。给 `ls` 一个上界是最直接的结构性保险。
    """
    if ls_max is None:
        return raw
    return float(ls_max) * torch.sigmoid(raw)


def cap_residual(res: torch.Tensor, x: torch.Tensor, cap: Optional[float]) -> torch.Tensor:
    """**逐 token 残差范数上限**：保证 `‖res_j‖ ≤ cap · ‖x_j‖`（`cap=None` 或 ≤0 时不变）。

    与 `bounded_ls` 互补：ls 只控整体幅度，这里控"每个 token 最多被推动多少"，
    对少数异常 token（正是会诱发退化的那些）额外有效。乘的系数 ≤1，梯度照常回传。
    """
    if cap is None or float(cap) <= 0:
        return res
    rn = res.norm(dim=-1, keepdim=True)
    xn = x.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    factor = (float(cap) * xn / rn.clamp(min=1e-6)).clamp(max=1.0)
    return res * factor


def object_box_of(polygon: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """多边形 → 外接框（像素）。"""
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


# ===========================================================================
# ① 尺度条件注入 ScaleCond（小目标增强 / 尺寸自适应）
# ===========================================================================


class ScaleCond(nn.Module):
    """把"目标在裁剪框里有多大"用 FiLM 注入到全部视觉 token。

        γ, β = MLP(scale_code)                     # [B,dim] × 2，**零初始化** → γ=β=0
        x'   = x + ls · (LN(x) ⊙ γ + β)            # ls = 1.0，step0 恒等但立刻可学

    为什么不是"再拼一个 token"：视觉占位符必须恒为 64 个（processor / 坐标词表 / 既有 ckpt
    都按 64 写死），所以只能**调制**这 64 个 token，不能增删。

    参数量 ≈ 2 × 1024 × 256 = 0.53M（hidden=256）。实测病灶：模型输出尺寸锚在裁剪框上，
    这个模块就是显式把"应该多大"喂给它。
    """

    def __init__(self, dim: int = 1024, code_dim: int = POS_DIM, hidden: int = 256,
                 ls_init: float = 1.0, dropout: float = 0.0,
                 ls_max: Optional[float] = None, res_cap: Optional[float] = None):
        super().__init__()
        self.ls_max, self.res_cap = ls_max, res_cap
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(code_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * dim))
        # ⚠️ 零初始化**输出层**（不是 ls）：step0 恒等，且第一步就有梯度
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.drop = nn.Dropout(dropout)
        self.ls = nn.Parameter(torch.full((1,), float(ls_init)))

    def forward(self, x: torch.Tensor, code: Optional[torch.Tensor] = None,
                aux_out: Optional[dict] = None,
                gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`gate`：**样本级尺寸门控** `[B]` 或 `[B,1]`，∈[0,1]；1=用模块，0=完全退回官方路径。

        离线实验（`tools/_tmp/analyze_size_gate.py`）显示：只在**短边 ≤32~48 px** 的楼上启用
        尺度条件，gzda_v0 val 上 IoU 0.7484→0.7522~0.7532、p@0.75 0.6021→0.6165~0.6190；
        中/大目标保持官方模型不变。门控值来自**目标框短边**（部署时 = 检测框，不需要 GT）。
        """
        if code is None:
            return x
        g, b = self.mlp(self.drop(code.to(x.dtype))).chunk(2, dim=-1)     # [B,dim] ×2
        y = self.norm(x) * g.unsqueeze(1) + b.unsqueeze(1)
        res = bounded_ls(self.ls, self.ls_max) * y
        if gate is not None:
            w = gate.to(res.dtype).reshape(-1, 1, 1)
            res = res * w
        res = cap_residual(res, x, self.res_cap)
        if aux_out is not None:
            aux_out["scale_gamma_abs"] = float(g.abs().mean().detach())
            aux_out["arch_residual_scale"] = res.pow(2).mean()   # 逐模块记录（训练期正则用）
        return x + res


# ===========================================================================
# ② 边界 + 角点引导增强 EdgeCornerGuide
# ===========================================================================


class EdgeCornerGuide(HRBoundaryEnhance):
    """在 `HRBoundaryEnhance` 的基础上加两条通道（形状/权重全部继承，老 ckpt 仍能加载）：

    **(a) 稠密边界注入**（新增）—— 把 256² 分支的 `edge_feat[B,C,64,64]` 池化到 8×8，
        `Linear(C→dim)` 后**加到每一个视觉 token** 上。旧版只把边界信息塞进 32~36 个
        boundary token 里做 K/V，主 token 本身拿不到"我这一格有没有边界"这个信号。

    **(b) 角点热图监督**（新增）—— `corner_head: Conv1×1 → [B,1,64,64]`，
        目标是 GT 顶点的高斯热图。**LLM 生成的正是顶点**，这条监督是唯一直接压在
        "顶点该在哪"上的辅助信号（旧版只有边界线监督）。

    残差写法同 `ScaleCond`：`out.weight` 与 `dense_proj` 都零初始化 → step0 恒等，第一步可学。
    """

    def __init__(self, dim: int = 1024, hr_size: int = HR_SIZE, edge_size: int = EDGE_SIZE,
                 hr_channels: int = 64, heads: int = 8, n_tokens: int = 32, n_uniform: int = 8,
                 pos_dim: int = POS_DIM, ls_init: float = 1.0, gate_init: float = 0.05,
                 dropout: float = 0.0, use_dense: bool = True,
                 ls_max: Optional[float] = None, res_cap: Optional[float] = None):
        super().__init__(dim, hr_size=hr_size, edge_size=edge_size, hr_channels=hr_channels,
                         heads=heads, n_tokens=n_tokens, n_uniform=n_uniform, pos_dim=pos_dim,
                         ls_init=ls_init, gate_init=gate_init, dropout=dropout)
        self.ls_max, self.res_cap = ls_max, res_cap
        self.use_dense = bool(use_dense)
        self.corner_head = nn.Conv2d(hr_channels, 1, 1)
        self.dense_proj = nn.Sequential(nn.LayerNorm(hr_channels), nn.Linear(hr_channels, dim))
        # 零初始化：不给 step0 引入任何扰动（父类里 ls 也是 1.0，靠这里保持恒等）
        nn.init.zeros_(self.dense_proj[-1].weight)
        nn.init.zeros_(self.dense_proj[-1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, hr_pixel_values: Optional[torch.Tensor] = None,
                aux_out: Optional[dict] = None) -> torch.Tensor:
        if hr_pixel_values is None:          # 没有 HR 输入 → 纯恒等（不崩、不改语义）
            return x
        edge_feat, edge_logits = self.encoder(hr_pixel_values)
        corner_logits = self.corner_head(edge_feat)
        dense = None
        if self.use_dense:
            pooled = F.adaptive_avg_pool2d(edge_feat, GRID_TOKENS)          # [B,C,8,8]
            pooled = pooled.flatten(2).transpose(1, 2)                      # [B,64,C]
            if pooled.shape[1] != x.shape[1]:                               # 兜底：对齐 token 数
                pooled = F.adaptive_avg_pool1d(pooled.transpose(1, 2), x.shape[1]).transpose(1, 2)
            dense = self.dense_proj(pooled)                                 # [B,64,dim]
        tokens, prob = self.sampler(edge_feat, edge_logits)
        ctx = self.drop(self._mha(self.norm_q(x), self.norm_kv(tokens)))
        g = torch.sigmoid(self.gate(torch.cat([x, ctx], dim=-1)))
        res = cap_residual(bounded_ls(self.ls, self.ls_max)
                           * (g * ctx + (dense if dense is not None else 0.0)), x, self.res_cap)
        out = x + res
        if aux_out is not None:
            aux_out["edge_logits"] = edge_logits
            aux_out["corner_logits"] = corner_logits
            aux_out["edge_prob_mean"] = float(torch.sigmoid(edge_logits).mean().detach())
            aux_out["boundary_tokens"] = int(tokens.shape[1])
            aux_out["arch_residual_edge"] = res.pow(2).mean()
        return out


def corner_label_from_polygon(polygon: Sequence[Sequence[float]], crop_size: Tuple[float, float],
                              edge_size: int = EDGE_SIZE, sigma: float = 1.2,
                              max_vertices: int = 64) -> torch.Tensor:
    """crop 像素坐标的多边形 → `[1,edge_size,edge_size]` 的**顶点高斯热图**（训练期辅助监督）。

    与 `edge_label_from_polygon`（画线）互补：线告诉模型"边界在哪"，点告诉它"拐角在哪"。
    标签只由 GT 生成；推理路径完全不读标签。
    """
    import numpy as np

    canvas = np.zeros((edge_size, edge_size), dtype="float32")
    if not polygon:
        return torch.from_numpy(canvas).unsqueeze(0)
    sx = edge_size / float(max(crop_size[0], 1))
    sy = edge_size / float(max(crop_size[1], 1))
    r = max(1, int(math.ceil(3.0 * sigma)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    kern = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2)).astype("float32")
    for pt in list(polygon)[:max_vertices]:
        ix = int(round(float(pt[0]) * sx))
        iy = int(round(float(pt[1]) * sy))
        if not (0 <= ix < edge_size and 0 <= iy < edge_size):
            continue                       # 顶点落在裁剪外（框抖动/外扩时可能出现）
        x0, x1 = max(0, ix - r), min(edge_size, ix + r + 1)
        y0, y1 = max(0, iy - r), min(edge_size, iy + r + 1)
        canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1],
                                          kern[y0 - (iy - r):y1 - (iy - r), x0 - (ix - r):x1 - (ix - r)])
    return torch.from_numpy(canvas).unsqueeze(0).contiguous()


def corner_loss(logits: torch.Tensor, target: torch.Tensor,
                pos_weight: float = 20.0) -> Tuple[torch.Tensor, float]:
    """角点热图损失 = 加权 BCE + Dice（与 `edge_loss` 同构，目标值是 0~1 软标签）。"""
    target = target.to(logits.dtype)
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target, size=logits.shape[-2:], mode="bilinear", align_corners=False)
    pos = float(target.sum())
    neg = float(target.numel()) - pos
    pw = torch.tensor(1.0 if pos <= 0 else min(max(pos_weight, neg / max(pos, 1.0)), 2000.0),
                      device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    p = torch.sigmoid(logits)
    num = 2.0 * (p * target).sum(dim=(1, 2, 3)) + 1.0
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
    dice = (1.0 - num / den).mean()
    total = bce + dice
    return total, float(total.detach())


# ===========================================================================
# ③ 多尺度放大分支融合 ZoomFusion（小目标 + 动态特征融合）
# ===========================================================================

ZOOM_RATIO_MIN = 1.06     # 再紧就会撞"框紧度悬崖"（外扩 1.0 时 p@0.75 只有 0.196）


def adaptive_zoom_ratio(r_main: float) -> float:
    """主裁剪比 → 放大分支裁剪比。**总是比主裁剪紧**，但留 6% 边距避免轮廓贴边。"""
    return float(min(max(r_main * 0.85, ZOOM_RATIO_MIN), max(r_main, ZOOM_RATIO_MIN)))


def zoom_crop_box(obj_box: Sequence[float], ratio: float,
                  image_size: Sequence[int]) -> Tuple[int, int, int, int]:
    """目标框按 `ratio` 外扩、并夹到图像内 → 整数裁剪框。"""
    x1, y1, x2, y2 = [float(v) for v in obj_box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = max(1.0, x2 - x1) * ratio, max(1.0, y2 - y1) * ratio
    W, H = float(image_size[0]), float(image_size[1])
    bx1 = int(max(0, min(W - 2, round(cx - w / 2.0))))
    by1 = int(max(0, min(H - 2, round(cy - h / 2.0))))
    bx2 = int(max(bx1 + 2, min(W, round(cx + w / 2.0))))
    by2 = int(max(by1 + 2, min(H, round(cy + h / 2.0))))
    return bx1, by1, bx2, by2


def build_zoom_inputs(image, obj_box: Sequence[float], image_processor,
                      r_main: Optional[float] = None, r_zoom: Optional[float] = None,
                      code_dim: int = POS_DIM,
                      crop_box: Optional[Sequence[float]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """整幅瓦片 + 目标框 → `(zoom_pixel_values [3,128,128], scale_code [64])`。

    推理期**不读任何标签**：`obj_box` 就是检测框（或评测用的 GT 框）。
    """
    W, H = image.size
    x1, y1, x2, y2 = [float(v) for v in obj_box]
    if r_main is None:
        from .geometry import adaptive_expand_ratio
        r_main = adaptive_expand_ratio(min(max(x2 - x1, 1.0), max(y2 - y1, 1.0)))
    if r_zoom is None:
        r_zoom = adaptive_zoom_ratio(r_main)
    box = zoom_crop_box(obj_box, r_zoom, (W, H))
    crop = image.crop(box)
    pv = image_processor(crop, return_tensors="pt")["pixel_values"][0]
    # ⚠️ 2026-11-04 修（审查发现）：训练用的 scale_code 是"目标框 vs **实际裁剪框**"，
    #    而这里原来用 r_main 重新拼了一个**居中**框 → 边界被裁/非对称外扩时两边不一致。
    #    现在优先直接吃调用方传进来的真实裁剪框。
    main_box = tuple(float(v) for v in crop_box) if crop_box is not None \
        else zoom_crop_box(obj_box, r_main, (W, H))
    return pv, scale_code(obj_box, main_box, dim=code_dim)


class ZoomFusion(nn.Module):
    """放大分支（同一目标、更紧的外扩 → 等效线放大 ~1.15×）与主分支的**动态门控融合**。

        Q   = Wq·LN(local) + MLP(scale_code)      # "我在框里占多大"
        ctx = MHA(Q, Wk·LN(zoom), Wv·LN(zoom))    # 只在放大分支里取细节
        G   = σ(MLP([local, ctx]))                # 逐通道门控：压掉无关细节
        λ   = σ(MLP([pool(local), pool(zoom), pool(local⊙ctx), scale_code]))   # 样本级强度
        out = local + ls · λ · G ⊙ ctx

    **λ 是"动态特征融合"的落点**：小目标样本（`scale_code` 里 w/cw 小）自动把放大分支的
    权重抬高；大楼样本 λ→0，等于不挂这条分支。旧 `GlobalFusion` 只有一个全局标量 ls。

    没有 FFN（省 4.2M），参数量 ≈ 5.4M。视觉塔冻结时这只是一次额外前向。
    """

    def __init__(self, dim: int = 1024, heads: int = 8, code_dim: int = POS_DIM,
                 ls_init: float = 1.0, gate_init: float = 0.1, router_init: float = 0.1,
                 use_dynamic: bool = True, dropout: float = 0.0,
                 ls_max: Optional[float] = None, res_cap: Optional[float] = None):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.ls_max, self.res_cap = ls_max, res_cap
        self.use_dynamic = bool(use_dynamic)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.scale_proj = nn.Sequential(nn.Linear(code_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)          # 零初始化输出投影 → step0 恒等、第一步可学
        nn.init.zeros_(self.out.bias)
        hid = max(dim // 4, 32)
        self.gate = nn.Sequential(nn.Linear(2 * dim, hid), nn.GELU(), nn.Linear(hid, dim))
        with torch.no_grad():
            g = min(max(float(gate_init), 1e-4), 1 - 1e-4)
            self.gate[-1].bias.fill_(math.log(g / (1 - g)))
            self.gate[-1].weight.mul_(0.1)
        if self.use_dynamic:
            # 输入 = pool(local) ⊕ pool(zoom) ⊕ pool(local⊙ctx) ⊕ scale_code
            self.router = nn.Sequential(nn.Linear(3 * dim + code_dim, hid), nn.GELU(),
                                        nn.Linear(hid, dim))
            with torch.no_grad():
                r = min(max(float(router_init), 1e-4), 1 - 1e-4)
                self.router[-1].bias.fill_(math.log(r / (1 - r)))
                self.router[-1].weight.mul_(0.1)
        self.drop = nn.Dropout(dropout)
        self.ls = nn.Parameter(torch.full((1,), float(ls_init)))

    def _mha(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        b, n, c = q.shape
        hd = c // self.heads
        q = self.q(q).view(b, n, self.heads, hd).transpose(1, 2)
        k = self.k(kv).view(b, kv.shape[1], self.heads, hd).transpose(1, 2)
        v = self.v(kv).view(b, kv.shape[1], self.heads, hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.out(o.transpose(1, 2).reshape(b, n, c))

    def forward(self, local: torch.Tensor, zoom: Optional[torch.Tensor] = None,
                code: Optional[torch.Tensor] = None,
                aux_out: Optional[dict] = None) -> torch.Tensor:
        if zoom is None:
            return local
        q = self.norm_q(local)
        if code is not None:
            q = q + self.scale_proj(code.to(q.dtype)).unsqueeze(1)
        ctx = self.drop(self._mha(q, self.norm_kv(zoom)))
        g = torch.sigmoid(self.gate(torch.cat([local, ctx], dim=-1)))
        if self.use_dynamic:
            pooled = torch.cat([local.mean(1), zoom.mean(1), (local * ctx).mean(1)], dim=-1)
            if code is not None:
                pooled = torch.cat([pooled, code.to(pooled.dtype)], dim=-1)
            lam = torch.sigmoid(self.router(pooled)).unsqueeze(1)        # [B,1,D]
            if aux_out is not None:
                aux_out["zoom_lambda_mean"] = float(lam.mean().detach())
                aux_out["zoom_gate_mean"] = float(g.mean().detach())
            ctx = lam * g * ctx
        else:
            ctx = g * ctx
        res = cap_residual(bounded_ls(self.ls, self.ls_max) * ctx, local, self.res_cap)
        if aux_out is not None:
            aux_out["arch_residual_zoom"] = res.pow(2).mean()
        return local + res


# ===========================================================================
# 挂载 / ckpt 兼容
# ===========================================================================


def attach_scale_edge(model, dim: int = 1024, heads: int = 8, scale_cond: bool = True,
                      edge_corner: bool = False, zoom: bool = False, code_dim: int = POS_DIM,
                      scale_cond_hidden: int = 256, hr_edge: bool = True,
                      hr_size: int = HR_SIZE, edge_size: int = EDGE_SIZE, hr_channels: int = 64,
                      boundary_tokens: int = 32, boundary_uniform: int = 8,
                      gate_init: float = 0.05, ls_init: float = 1.0,
                      edge_use_dense: bool = True, zoom_dynamic: bool = True,
                      ls_max: Optional[float] = None, res_cap: Optional[float] = None,
                      verbose: bool = True) -> Dict:
    """把 ①ScaleCond / ②EdgeCornerGuide / ③ZoomFusion 挂到 `VectorLLMForCausalLM` 上。

    * `edge_corner=True` 会**替换** `global_fusion` 挂的 `boundary_enhance`（同为 HR 分支，
      新版多两条通道）。两者同时为真时以本函数为准，避免两个 256² 分支重复算。
    * 全部零初始化输出层 → step0 与"不挂模块"逐位一致。
    """
    inner = getattr(model, "module", model)
    if scale_cond:
        inner.scale_cond = ScaleCond(dim, code_dim=code_dim, hidden=scale_cond_hidden,
                                     ls_init=1.0, ls_max=ls_max, res_cap=res_cap)
    if zoom:
        inner.zoom_fusion = ZoomFusion(dim, heads=heads, code_dim=code_dim,
                                       ls_init=1.0, use_dynamic=zoom_dynamic,
                                       ls_max=ls_max, res_cap=res_cap)
    if edge_corner:
        inner.boundary_enhance = EdgeCornerGuide(
            dim, hr_size=hr_size, edge_size=edge_size, hr_channels=hr_channels, heads=heads,
            n_tokens=boundary_tokens, n_uniform=boundary_uniform, ls_init=1.0,
            gate_init=gate_init, use_dense=edge_use_dense, ls_max=ls_max, res_cap=res_cap)
        inner._edge_corner = True
    # dtype/device 对齐（视觉塔可能是 bf16、projector 是 fp32 之类）
    ref = next(inner.projector.parameters())
    for name in ("scale_cond", "zoom_fusion", "boundary_enhance"):
        mod = getattr(inner, name, None)
        if mod is not None:
            mod.to(device=ref.device, dtype=ref.dtype)

    n = 0
    for name in ("scale_cond", "zoom_fusion", "boundary_enhance"):
        mod = getattr(inner, name, None)
        if mod is not None:
            n += sum(p.numel() for p in mod.parameters())
    info = {"scale_cond": bool(scale_cond), "scale_cond_hidden": int(scale_cond_hidden),
            "zoom": bool(zoom), "zoom_dynamic": bool(zoom_dynamic),
            "edge_corner": bool(edge_corner), "edge_use_dense": bool(edge_use_dense),
            "code_dim": int(code_dim), "hr_edge": bool(hr_edge and edge_corner),
            "hr_size": int(hr_size), "edge_size": int(edge_size), "hr_channels": int(hr_channels),
            "boundary_tokens": int(boundary_tokens), "boundary_uniform": int(boundary_uniform),
            "dim": int(dim), "heads": int(heads), "scale_edge_params": int(n),
            "ls_max": ls_max, "res_cap": res_cap}
    if verbose:
        def _p(name):
            mod = getattr(inner, name, None)
            return 0 if mod is None else sum(p.numel() for p in mod.parameters())
        print(f"[scale_edge] scale_cond={scale_cond}({_p('scale_cond')/1e6:.2f}M) "
              f"zoom={zoom}({_p('zoom_fusion')/1e6:.2f}M) "
              f"edge_corner={edge_corner}({(sum(p.numel() for p in inner.boundary_enhance.parameters())/1e6) if (edge_corner and getattr(inner, 'boundary_enhance', None) is not None) else 0:.2f}M)"
              f" → **{n/1e6:.2f}M**（零初始化输出投影：step0 与基线逐位一致，第一步即可学）",
              flush=True)
    return info


def maybe_attach_from_ckpt(model, ckpt_dir, verbose: bool = True) -> Optional[Dict]:
    """评测/续训端：`arch.json` 里有本文件的开关就按它挂回去（否则返回 None）。

    ⚠️ 必须在 `load_state_dict` **之前**调用（否则新模块权重会被当 unexpected key 丢掉）。
    """
    import json
    from pathlib import Path

    path = Path(ckpt_dir) / "arch.json"
    if not path.exists():
        return None
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not (cfg.get("scale_cond") or cfg.get("zoom") or cfg.get("edge_corner")):
        return None
    return attach_scale_edge(
        model, dim=int(cfg.get("dim", 1024)), heads=int(cfg.get("heads", 8)),
        scale_cond=bool(cfg.get("scale_cond", False)),
        edge_corner=bool(cfg.get("edge_corner", False)),
        zoom=bool(cfg.get("zoom", False)),
        code_dim=int(cfg.get("code_dim", POS_DIM)),
        scale_cond_hidden=int(cfg.get("scale_cond_hidden", 256)),
        hr_edge=bool(cfg.get("hr_edge", True)),
        hr_size=int(cfg.get("hr_size", HR_SIZE)), edge_size=int(cfg.get("edge_size", EDGE_SIZE)),
        hr_channels=int(cfg.get("hr_channels", 64)),
        boundary_tokens=int(cfg.get("boundary_tokens", 32)),
        boundary_uniform=int(cfg.get("boundary_uniform", 8)),
        edge_use_dense=bool(cfg.get("edge_use_dense", True)),
        zoom_dynamic=bool(cfg.get("zoom_dynamic", True)),
        ls_max=cfg.get("ls_max"), res_cap=cfg.get("res_cap"),
        verbose=verbose,
    )
