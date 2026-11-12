"""**目标感知全局上下文融合 + 高分辨率边界引导增强**（VectorLLM 视觉 token 级改进）。

流水线位置（不动 RADIO / projector / ParaX / Qwen3）：

    裁剪图 ──────────► RADIO ──► local tokens [B,64,1024] ─┐
    整幅瓦片 ────────► RADIO ──► global tokens [B,64,1024] ─┤
    裁剪框 box ──► Fourier pos ─────────────────────────┘
                                    │
                       ① TargetAwareGlobalFusion（局部作 Q，全局作 K/V，门控残差）
                                    │
    原始高分辨率 crop ──► HREdgeEncoder ──► 边界图 + EdgeHead
                                    │            │
                       ② BoundaryGuidedEnhance（边界引导稀疏采样 → 交叉注意力 → 门控残差）
                                    │
                              projector(+ParaX) ──► Qwen3 ──► 多边形

设计约束（与既有训练/推理逻辑兼容）：
1. 两个模块输出**始终是 `[B, Nl, D]`**，token 数与维度不变 → 下游 projector/Qwen3 完全不用改。
2. 全部**零初始化残差**（`ls=0`）→ 关掉或 step0 时与原模型**逐位一致**（可热启动、可公平对比）。
3. 关掉开关时**连计算都不发生**（`global_fusion=None` / `boundary_enhance=None`）。
4. 边界监督（EdgeHead）**只在训练期**由 GT 多边形栅格化生成，推理路径不读任何标签。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

POS_DIM = 64          # 4 个标量 × 8 频率 × (sin,cos)
POS_DIM_ANGLE = 96    # 6 个标量（+sin2θ,cos2θ）× 8 频率 × (sin,cos) —— OBB 条件版
HR_SIZE = 256         # 高分辨率边界分支输入边长
EDGE_SIZE = 64        # 边界概率图/监督边长（= HR_SIZE / 4）


# --------------------------------------------------------------------------- 位置编码


def pos_box_from_box(box, image_size) -> torch.Tensor:
    """(x1,y1,x2,y2) + 瓦片尺寸 → `[4]` 的**原始几何量** `(cx/W, cy/H, w/W, h/H)`。

    ⚠️ 2026-11-04 修（审查发现）：`pos_code_from_box` 返回的是 **Fourier 编码**，
    它的前 4 项是"中心 x 的 sin/cos"（两个频率），**不是** (cx, cy, w, h)。
    空间偏置必须吃这个原始量，否则 `gx - cx` 里的 cx 其实是个 sin 值，几何意义完全错。
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    W, H = float(image_size[0]), float(image_size[1])
    return torch.tensor([(x1 + x2) / 2.0 / W, (y1 + y2) / 2.0 / H,
                         max(0.0, x2 - x1) / W, max(0.0, y2 - y1) / H], dtype=torch.float32)


def pos_code_from_box(box, image_size, dim: int = POS_DIM,
                      angle: Optional[float] = None) -> torch.Tensor:
    """(x1,y1,x2,y2) + 瓦片尺寸 → (dim,) 的 Fourier 位置编码（归一化 + 多频 sin/cos）。

    表达"这个局部裁剪框在整幅瓦片的哪里、多大"，注入到全局融合的 Query 上。

    `angle`（弧度）：**OBB 条件版**（2026-11-06 新增）。转正裁剪时"裁剪框在整幅图里的朝向"
    也是上下文的一部分——只用外接框的 (cx,cy,w,h) 丢掉 θ，会让位置编码与真实裁剪几何对不上。
    给 `angle` 时编码变成 6 个标量 `(cx/W, cy/H, w/W, h/H, sin2θ, cos2θ)`，维度应为 96
    （`POS_DIM_ANGLE`），用双角编码避免 ±π 歧义。
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    W, H = float(image_size[0]), float(image_size[1])
    _vals = [(x1 + x2) / 2.0 / W, (y1 + y2) / 2.0 / H,
             max(0.0, x2 - x1) / W, max(0.0, y2 - y1) / H]
    if angle is not None:
        _vals += [math.sin(2.0 * float(angle)), math.cos(2.0 * float(angle))]
    vals = torch.tensor(_vals, dtype=torch.float32)
    n_freq = max(1, dim // (2 * len(_vals)))
    out = []
    for v in vals:
        for k in range(n_freq):
            f = 2.0 ** k * math.pi
            out += [torch.sin(f * v), torch.cos(f * v)]
    code = torch.stack(out)                     # 4*2*n_freq
    if code.numel() < dim:
        code = F.pad(code, (0, dim - code.numel()))
    return code[:dim]


def fourier_2d(xy: torch.Tensor, n_freq: int = 8) -> torch.Tensor:
    """归一化坐标 `xy[..., 2] ∈ [0,1]` → `[..., 4*n_freq]` 多频 sin/cos（边界 token 的位置编码）。"""
    out = []
    for k in range(n_freq):
        f = 2.0 ** k * math.pi
        out += [torch.sin(f * xy), torch.cos(f * xy)]        # 各 2 维
    return torch.cat(out, dim=-1)                            # 4*n_freq


# ------------------------------------------------------------------- ⓿ 检测器多层全局 token 适配器


class DetGlobalAdapter(nn.Module):
    """检测器 FPN 的多层池化特征 → GlobalFusion 的 K/V token（2026-11-07 新增）。

    输入：`[B, N, in_dim]`，`N = grid²`（默认 8×8=64），`in_dim = 256 × len(levels)`（P3/P4/P5 → 768）。
    输出：`[B, N, dim] = [B, 64, 1024]`，与 RADIO 局部 token 同维度，可直接喂 `GlobalFusion`。

    为什么需要它：原来的 GlobalFusion 用「整幅瓦片再跑一次 RADIO」当 K/V，
    那条通路**没有检测监督**、且等效 stride 很大（1024 瓦片缩到 128 再 patch16）；
    用检测器自己的 FPN 特征（stride 8/16/32，已被建筑监督训练过）作 K/V 才是"目标感知"的语义。
    顺带省掉一次视觉塔前向。
    """

    def __init__(self, in_dim: int = 768, dim: int = 1024, zero_init: bool = False):
        super().__init__()
        self.in_dim, self.out_dim = int(in_dim), int(dim)
        self.zero_init = bool(zero_init)
        self.norm = nn.LayerNorm(self.in_dim)
        self.proj = nn.Linear(self.in_dim, self.out_dim)
        # 常规版用于复现旧 ckpt。zero_init 版不能直接把 proj.weight 全置零：
        # 那会让 norm/proj 的梯度也为零。改为小投影 + 零初始化输出缩放：
        # step0 严格退回 SC-only，且 proj 与 scale 都能立即学习。
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        self.out_scale = nn.Parameter(torch.zeros(self.out_dim) if self.zero_init
                                      else torch.ones(self.out_dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # 与挂载后的 LayerNorm 参数 dtype 保持一致（bf16 模型不能用 float32 输入）。
        x = self.norm(tokens.to(self.norm.weight.dtype))
        x = x.to(self.proj.weight.dtype)
        return self.proj(x) * self.out_scale


def attach_det_global(model, dim: int = 1024, in_dim: int = 768, zero_init: bool = False) -> dict:
    """把 `DetGlobalAdapter` 挂到模型上（trainable），并返回写进 arch.json 的配置。"""
    inner = getattr(model, "module", model)
    mod = DetGlobalAdapter(in_dim=in_dim, dim=dim, zero_init=zero_init)
    ref = next(inner.projector.parameters())
    inner.det_global_adapter = mod.to(device=ref.device, dtype=ref.dtype)
    n = sum(p.numel() for p in mod.parameters())
    print(f"[det_global] 多层全局 token 适配器挂载：{in_dim}→{dim}，{n/1e6:.2f}M 参数（零初始化）", flush=True)
    return {"det_global": True, "det_global_in": int(in_dim), "det_global_zero_init": bool(zero_init),
            "det_global_params": int(n)}


# --------------------------------------------------------------------------- ① 目标感知全局融合


class GlobalFusion(nn.Module):
    """目标感知交叉注意力：**局部 token 作 Q，整幅瓦片 token 作 K/V**，门控残差注入。

    相对"直接相加/拼接"的改进（对应任务书 2.1~2.4）：
    * Q/K/V 各自 LayerNorm，避免尺度不一致；
    * `pos_code`（裁剪框在整幅图里的归一化位置/尺寸）经 MLP 注入 **Query**，
      让局部特征知道"我在场景的哪里、我多大"；
    * **逐通道门控** `G = σ(MLP([F_l, F_c]))`，`F_out = F_l + ls·(G ⊙ F_c)`，
      抑制与目标无关的背景上下文（门控初值偏小，`gate_init`）；
    * `ls` 零初始化 → step0 与"不挂模块"逐位一致，可安全热启动。
    """

    def __init__(self, dim: int, heads: int = 8, pos_dim: int = POS_DIM,
                 ls_init: float = 0.0, dropout: float = 0.0,
                 gate_init: float = 0.05, use_gate: bool = True,
                 use_spatial_bias: bool = False, spatial_heads: int = 8,
                 ls_max: Optional[float] = None, res_cap: Optional[float] = None):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.pos_dim = int(pos_dim)      # 64=AABB 版；96=OBB 条件版（含 sin2θ/cos2θ）
        # **残差预算**（2026-11-04 补）：以前本模块不写 arch_residual，正则/限幅都不生效
        self.ls_max, self.res_cap = ls_max, res_cap
        self.use_gate = use_gate
        self.use_spatial_bias = use_spatial_bias
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.pos_mlp = nn.Sequential(nn.Linear(pos_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        # 逐通道门控（瓶颈结构，参数少）
        hid = max(dim // 4, 32)
        self.gate = nn.Sequential(nn.Linear(2 * dim, hid), nn.GELU(), nn.Linear(hid, dim))
        if gate_init is not None:
            # sigmoid 偏置初始化为 logit(gate_init) → 初始注入强度小，不破坏 RADIO 特征
            g = min(max(float(gate_init), 1e-4), 1 - 1e-4)
            with torch.no_grad():
                self.gate[-1].bias.fill_(math.log(g / (1 - g)))
                self.gate[-1].weight.mul_(0.1)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
                                 nn.Linear(dim * 2, dim))
        self.drop = nn.Dropout(dropout)
        self.ls1 = nn.Parameter(torch.full((1,), float(ls_init)))
        self.ls2 = nn.Parameter(torch.full((1,), float(ls_init)))
        # 可选消融：目标框与全局 token 的相对几何 → 注意力偏置（默认关闭）
        if use_spatial_bias:
            self.spatial_mlp = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, spatial_heads))
            self.spatial_heads = spatial_heads

    def _mha(self, q, kv, attn_bias: Optional[torch.Tensor] = None):
        b, n, c = q.shape
        hd = c // self.heads
        q = self.q(q).view(b, n, self.heads, hd).transpose(1, 2)               # (B,h,Nl,hd)
        k = self.k(kv).view(b, kv.shape[1], self.heads, hd).transpose(1, 2)    # (B,h,Ng,hd)
        v = self.v(kv).view(b, kv.shape[1], self.heads, hd).transpose(1, 2)
        mask = None
        if attn_bias is not None:
            mask = attn_bias[:, :, None, :].expand(-1, -1, n, -1)              # (B,h,Nl,Ng)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(o.transpose(1, 2).reshape(b, n, c))

    def forward(self, local: torch.Tensor, glob: torch.Tensor,
                pos_code: Optional[torch.Tensor] = None,
                aux_out: Optional[dict] = None,
                pos_box: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.norm_q(local)
        if pos_code is not None:
            q = q + self.pos_mlp(pos_code.to(q.dtype)).unsqueeze(1)   # "我在全局的哪里"
        bias = None
        if self.use_spatial_bias and self.spatial_heads == self.heads:
            # ⚠️ 必须用**原始归一化框**(cx,cy,w,h)，不能用 pos_code（那是 Fourier 编码）
            bias = self._spatial_bias(pos_box, glob.shape[1], local.dtype)
        ctx = self.drop(self._mha(q, self.norm_kv(glob), bias))        # 全局上下文 [B,Nl,D]
        if self.use_gate:
            g = torch.sigmoid(self.gate(torch.cat([local, ctx], dim=-1)))
            ctx = g * ctx
            if aux_out is not None:
                aux_out["fusion_gate_mean"] = float(g.mean().detach())
        from .scale_edge import bounded_ls, cap_residual

        res = bounded_ls(self.ls1, self.ls_max) * ctx
        res = res + bounded_ls(self.ls2, self.ls_max) * self.drop(self.ffn(local))
        res = cap_residual(res, local, self.res_cap)
        if aux_out is not None:
            # 训练期仪表盘：>0.02 就要出事（见 HANDOFF §5.2），压到 ~0.006 安全
            aux_out["arch_residual_global"] = res.pow(2).mean()
        return local + res

    def _spatial_bias(self, pos_box, n_global: int, dtype):
        """目标框 ↔ 全局 token 的**相对几何** → 每个 head 一个加性注意力偏置 `[B,h,Ng]`。

        ⚠️ 2026-11-04 修（审查发现的两个问题）：
          (a) 旧实现把 `pos_code[:4]` 当成 (cx,cy,w,h)，但 `pos_code_from_box` 返回的是 **Fourier 编码**
              （前 4 项是"中心 x 的 sin/cos"两个频率）→ 几何意义完全错。现在改用
              `pos_box_from_box()` 得到的**原始归一化框**；
          (b) 上一版虽然改成了逐 token，但仍然吃的是 Fourier 编码 → 同样错。两者一起修。
        现在按全局 token 的**真实格心坐标**算：

            gx, gy = (c+0.5)/S, (r+0.5)/S        # 全局 8×8 网格 → 瓦片归一化坐标
            dx, dy = gx-cx, gy-cy                # 相对目标中心（归一化）
            nx, ny = dx/(w/2), dy/(h/2)          # 按目标尺寸归一化（±1 = 目标框边，0 = 中心）

        于是"离这栋楼多远的上下文"直接进 attention logits：贴邻/沿街的结构被显式拉近，
        远处无关地物被压低。参数形状不变（`Linear(4,64)+Linear(64,h)`），老 ckpt 照常加载。
        """
        if pos_box is None or n_global <= 0:
            return None
        n_side = int(round(math.sqrt(n_global)))
        if n_side * n_side != n_global:          # 非方形 token 网格 → 不做空间偏置
            return None
        b = pos_box.shape[0]
        cx, cy, w, h = (pos_box[:, i] for i in range(4))   # 原始归一化框，不是 Fourier 编码
        idx = torch.arange(n_side, device=pos_box.device, dtype=torch.float32)
        gx = ((idx + 0.5) / n_side)[None, None, :].expand(b, n_side, n_side)   # 列 → x
        gy = gx.transpose(1, 2)                                               # 行 → y
        cxb, cyb = cx[:, None, None], cy[:, None, None]
        wb = (0.5 * w)[:, None, None].clamp(min=1e-4)
        hb = (0.5 * h)[:, None, None].clamp(min=1e-4)
        feat = torch.stack([gx - cxb, gy - cyb, (gx - cxb) / wb, (gy - cyb) / hb], dim=-1)
        # 小目标时 dx/(w/2) 可以很大（1/小框）→ **夹到 [-8, 8]**，保证偏置量级稳定
        feat = feat.clamp(-8.0, 8.0).reshape(b, n_global, 4)
        return self.spatial_mlp(feat.to(dtype)).transpose(1, 2)               # [B,h,Ng]


# --------------------------------------------------------------------------- ② HR 边界增强


class HREdgeEncoder(nn.Module):
    """轻量高分辨率边界编码器：`[B,3,256,256] → [B,C,64,64]`。

    只补 RADIO 在 128×128 输入上丢掉的**高频几何**（屋顶边缘/角点），不换主干。
    参数量 ≈ 7.5 万（C=64），算力约 0.35 GFLOPs/样本（256² 输入）。
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        c = channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, 3, stride=2, padding=1), nn.BatchNorm2d(c), nn.GELU())      # 128
        self.block1 = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=2, padding=1), nn.BatchNorm2d(c), nn.GELU())      # 64
        self.block2 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.GELU())                # 64
        self.edge_head = nn.Conv2d(c, 1, 1)

    def forward(self, hr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.block2(self.block1(self.stem(hr.to(self.stem[0].weight.dtype))))
        return h, self.edge_head(h)          # (feat[B,C,64,64], logits[B,1,64,64])


class BoundaryTokenSampler(nn.Module):
    """边界引导稀疏采样：从边界概率图上取 Top-K 峰值位置，再取该位置的高频特征做 token。

    * 峰值抑制：先做 `max_pool` 局部极大值筛选，避免 K 个点全挤在同一条边上；
    * 稳定性：额外拼 `n_uniform` 个均匀网格点（防止边界预测早期不稳定时没有可用 token）；
    * `grid_sample` 取样 → 特征本身可反传；**Top-K 的位置索引本身不可微**，
      因此采样的"该采哪里"由 EdgeHead 的辅助损失（BCE+Dice）来监督，而不是靠 index 求导。
    """

    def __init__(self, in_ch: int, dim: int, n_tokens: int = 32, n_uniform: int = 8,
                 pos_dim: int = POS_DIM, peak_kernel: int = 5):
        super().__init__()
        self.n_tokens, self.n_uniform, self.peak_kernel = n_tokens, n_uniform, peak_kernel
        self.proj = nn.Linear(in_ch, dim)
        self.pos_proj = nn.Linear(4 * 8, dim)      # Fourier(8 频) → dim
        self.norm = nn.LayerNorm(dim)

    def forward(self, edge_feat: torch.Tensor, edge_logits: torch.Tensor):
        b, c, h, w = edge_feat.shape
        prob = torch.sigmoid(edge_logits)                                   # (B,1,h,w)
        with torch.no_grad():
            peak = F.max_pool2d(prob, self.peak_kernel, stride=1, padding=self.peak_kernel // 2)
            is_peak = (prob >= peak - 1e-6).flatten(2)                      # (B,1,hw)
            flat = prob.flatten(2)
            # 峰值优先，不足时用全局 top-k 兜底（保证长度恒为 n_tokens）
            mask_val = torch.where(is_peak.bool(), flat, torch.full_like(flat, -1.0)).squeeze(1)  # (B,hw)
            flat1 = flat.squeeze(1)                                                    # (B,hw)
            k1 = min(self.n_tokens, h * w)
            idx_peak = mask_val.topk(k1, dim=-1).indices                    # (B,k1) 峰值位置
            idx_all = flat1.topk(k1, dim=-1).indices                        # (B,k1) 全局 top-k 兜底
            valid = torch.gather(mask_val, 1, idx_peak) > -0.5              # 该峰值是否有效
            idx = torch.where(valid, idx_peak, idx_all)                     # 峰值不足处用 top-k
            if self.n_uniform > 0:
                g = int(math.sqrt(self.n_uniform)) or 1
                ys, xs = torch.meshgrid(torch.arange(g, device=prob.device),
                                        torch.arange(g, device=prob.device), indexing="ij")
                gy = ((ys.flatten().float() + 0.5) / g * h).long().clamp(0, h - 1)
                gx = ((xs.flatten().float() + 0.5) / g * w).long().clamp(0, w - 1)
                uidx = (gy * w + gx)[None, :].expand(b, -1)
                idx = torch.cat([idx, uidx], dim=1)
            xi = (idx % w).float()
            yi = torch.div(idx, w, rounding_mode="floor").float()
            # 归一化到 [-1,1] 供 grid_sample 使用（按像素中心）
            gx_n = (xi + 0.5) / w * 2 - 1
            gy_n = (yi + 0.5) / h * 2 - 1
            grid = torch.stack([gx_n, gy_n], dim=-1).unsqueeze(1)           # (B,1,K,2)
            xy_norm = torch.stack([(xi + 0.5) / w, (yi + 0.5) / h], dim=-1)  # (B,K,2)
        feat = F.grid_sample(edge_feat, grid.to(edge_feat.dtype), mode="bilinear",
                             align_corners=True)                            # (B,C,1,K)
        feat = feat.squeeze(2).transpose(1, 2)                              # (B,K,C) —— 可微
        pos = self.pos_proj(fourier_2d(xy_norm.to(edge_feat.dtype)))        # (B,K,dim)
        return self.norm(self.proj(feat) + pos), prob


class HRBoundaryEnhance(nn.Module):
    """高分辨率边界引导增强：`context tokens [B,Nl,D]` → `enhanced tokens [B,Nl,D]`。

    流程（对应任务书 3.2~3.5）：
        HR crop → HREdgeEncoder → (边界特征, 边界 logits)
                                   │            └─ EdgeHead 辅助损失（仅训练期）
                                   ▼
                        BoundaryTokenSampler（Top-K 峰值 + 均匀兜底 + 位置编码）
                                   ▼
        context tokens 作 Q，边界 token 作 K/V → MHA → 门控残差 → 输出 [B,Nl,D]

    `ls` 零初始化 → 初始化时恒等；`edge_head` 的辅助损失让"该采哪里"被显式监督。
    """

    def __init__(self, dim: int, hr_size: int = HR_SIZE, edge_size: int = EDGE_SIZE,
                 hr_channels: int = 64, heads: int = 8, n_tokens: int = 32, n_uniform: int = 8,
                 pos_dim: int = POS_DIM, ls_init: float = 0.0, gate_init: float = 0.05,
                 dropout: float = 0.0):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.hr_size, self.edge_size = hr_size, edge_size
        self.encoder = HREdgeEncoder(hr_channels)
        self.sampler = BoundaryTokenSampler(hr_channels, dim, n_tokens=n_tokens,
                                            n_uniform=n_uniform, pos_dim=pos_dim)
        hid = max(dim // 4, 32)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.gate = nn.Sequential(nn.Linear(2 * dim, hid), nn.GELU(), nn.Linear(hid, dim))
        g = min(max(float(gate_init), 1e-4), 1 - 1e-4)
        with torch.no_grad():
            self.gate[-1].bias.fill_(math.log(g / (1 - g)))
            self.gate[-1].weight.mul_(0.1)
        self.drop = nn.Dropout(dropout)
        self.ls = nn.Parameter(torch.full((1,), float(ls_init)))

    def _mha(self, q, kv):
        b, n, c = q.shape
        hd = c // self.heads
        q = self.q(q).view(b, n, self.heads, hd).transpose(1, 2)
        k = self.k(kv).view(b, kv.shape[1], self.heads, hd).transpose(1, 2)
        v = self.v(kv).view(b, kv.shape[1], self.heads, hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.out(o.transpose(1, 2).reshape(b, n, c))

    def forward(self, x: torch.Tensor, hr_pixel_values: Optional[torch.Tensor] = None,
                aux_out: Optional[dict] = None) -> torch.Tensor:
        if hr_pixel_values is None:                 # 没有 HR 输入 → 纯恒等（不崩、不改语义）
            return x
        edge_feat, edge_logits = self.encoder(hr_pixel_values)
        tokens, prob = self.sampler(edge_feat, edge_logits)
        if aux_out is not None:
            aux_out["edge_logits"] = edge_logits
            aux_out["boundary_tokens"] = int(tokens.shape[1])
        ctx = self.drop(self._mha(self.norm_q(x), self.norm_kv(tokens)))
        g = torch.sigmoid(self.gate(torch.cat([x, ctx], dim=-1)))
        return x + self.ls * (g * ctx)


class BoundaryEnhance(nn.Module):
    """**旧版**边界增强：视觉 token 网格上的 depthwise 3×3 差分 + sigmoid 门控 + 零初始化残差。

    保留它只为**兼容 2026-11-02 那批 ckpt**（`arch.json` 里没有 `hr_edge` 字段的那批），
    新训练一律用 `HRBoundaryEnhance`（真正补高频信息的那版）。
    """

    def __init__(self, dim: int, hidden: Optional[int] = None, ls_init: float = 0.0):
        super().__init__()
        hidden = hidden or max(dim // 8, 32)
        self.norm = nn.LayerNorm(dim)
        self.edge = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)     # 逐通道局部差分
        self.gate = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.GELU(),
                                  nn.Conv2d(hidden, dim, 1), nn.Sigmoid())
        self.proj = nn.Conv2d(dim, dim, 1)
        self.ls = nn.Parameter(torch.full((1,), float(ls_init)))

    def forward(self, x: torch.Tensor, hr_pixel_values=None, aux_out=None) -> torch.Tensor:
        b, n, c = x.shape
        g = int(round(n ** 0.5))
        h = self.norm(x).transpose(1, 2).reshape(b, c, g, g)
        y = self.proj(self.edge(h) * self.gate(h))
        return x + self.ls * y.flatten(2).transpose(1, 2)


# --------------------------------------------------------------------------- 挂载 / 接线


def attach_global_fusion(model, dim: int = 1024, heads: int = 8, boundary: bool = True,
                         glob: bool = True, pos_dim: int = POS_DIM, ls_init: float = 0.0, verbose: bool = True,
                         gate_init: float = 0.05, use_gate: bool = True,
                         use_spatial_bias: bool = False,
                         hr_edge: bool = False, hr_size: int = HR_SIZE,
                         edge_size: int = EDGE_SIZE, hr_channels: int = 64,
                         boundary_tokens: int = 32, boundary_uniform: int = 8,
                         ls_max: Optional[float] = None, res_cap: Optional[float] = None) -> dict:
    """给 `VectorLLMForCausalLM` 挂上全局融合 /（HR）边界增强。全部零初始化 → step0 与基线一致。

    `hr_edge=True` 时边界增强走**高分辨率边界引导**版本（需要数据侧提供 `hr_pixel_values`）；
    为兼容旧 ckpt，`hr_edge=False` 时保持旧行为（只用视觉 token 的局部差分，不再作为主路径）。
    """
    inner = getattr(model, "module", model)
    if glob:
        inner.global_fusion = GlobalFusion(dim, heads=heads, pos_dim=pos_dim, ls_init=ls_init,
                                           gate_init=gate_init, use_gate=use_gate,
                                           use_spatial_bias=use_spatial_bias,
                                           ls_max=ls_max, res_cap=res_cap)
    if boundary:
        if hr_edge:
            inner.boundary_enhance = HRBoundaryEnhance(
                dim, hr_size=hr_size, edge_size=edge_size, hr_channels=hr_channels, heads=heads,
                n_tokens=boundary_tokens, n_uniform=boundary_uniform, ls_init=ls_init,
                gate_init=gate_init)
        else:
            # 兼容旧 ckpt：token 网格差分版（arch.json 里没有 hr_edge 的都是这一批）
            inner.boundary_enhance = BoundaryEnhance(dim, ls_init=ls_init)
    ref = next(inner.projector.parameters())
    for name in ("global_fusion", "boundary_enhance"):
        mod = getattr(inner, name, None)
        if mod is not None:
            setattr(inner, name, mod.to(device=ref.device, dtype=ref.dtype))
    n = sum(p.numel() for mod in (getattr(inner, "global_fusion", None),
                                  getattr(inner, "boundary_enhance", None))
            if mod is not None for p in mod.parameters())
    info = {"global_fusion": bool(glob), "boundary_enhance": bool(boundary),
            "dim": dim, "heads": heads, "pos_dim": int(pos_dim),
            "pos_angle": bool(int(pos_dim) == POS_DIM_ANGLE),
            "ls_init": ls_init, "gate_init": gate_init,
            "use_gate": bool(use_gate), "use_spatial_bias": bool(use_spatial_bias),
            "hr_edge": bool(hr_edge), "hr_size": hr_size, "edge_size": edge_size,
            "hr_channels": hr_channels, "boundary_tokens": boundary_tokens,
            "boundary_uniform": boundary_uniform,
            "params": int(n)}
    if verbose:
        g_n = sum(p.numel() for p in inner.global_fusion.parameters()) if glob else 0
        b_n = sum(p.numel() for p in inner.boundary_enhance.parameters()) if boundary else 0
        print(f"[fusion] 挂载完成：global={glob}({g_n/1e6:.2f}M) "
              f"boundary={boundary}({b_n/1e6:.2f}M) → **{n/1e6:.2f}M 参数**"
              f"（零初始化，step0 与基线逐位一致）", flush=True)
    return info


def maybe_attach_from_ckpt(model, ckpt_dir, verbose: bool = True) -> Optional[dict]:
    """评测端：ckpt 目录里有 `arch.json` 就按里面的配置把模块挂回去（否则返回 None）。

    ⚠️ 必须在 `load_state_dict` **之前**调用，否则 ckpt 里的模块权重会被当成 unexpected key 丢掉。
    """
    import json
    from pathlib import Path

    path = Path(ckpt_dir) / "arch.json"
    if not path.exists():
        return None
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if bool(cfg.get("det_global", False)):
        attach_det_global(model, dim=int(cfg.get("dim", 1024)),
                          in_dim=int(cfg.get("det_global_in", 768)),
                          zero_init=bool(cfg.get("det_global_zero_init", False)))
    return attach_global_fusion(
        model, dim=int(cfg.get("dim", 1024)), heads=int(cfg.get("heads", 8)),
        boundary=bool(cfg.get("boundary_enhance")), glob=bool(cfg.get("global_fusion")),
        ls_init=float(cfg.get("ls_init", 0.0)), verbose=verbose,
        gate_init=float(cfg.get("gate_init", 0.05)), use_gate=bool(cfg.get("use_gate", True)),
        use_spatial_bias=bool(cfg.get("use_spatial_bias", False)),
        pos_dim=int(cfg.get("pos_dim", POS_DIM)),
        hr_edge=bool(cfg.get("hr_edge", False)), hr_size=int(cfg.get("hr_size", HR_SIZE)),
        edge_size=int(cfg.get("edge_size", EDGE_SIZE)),
        hr_channels=int(cfg.get("hr_channels", 64)),
        boundary_tokens=int(cfg.get("boundary_tokens", 32)),
        boundary_uniform=int(cfg.get("boundary_uniform", 8)))


def build_fusion_inputs(image, box, image_processor, angle=None, dim: int = POS_DIM):
    """给定整幅瓦片 + 局部框 → (global_pixel_values[1,3,H,W], pos_code[1,dim])。

    `angle` 非 None 时编码里带 (sin2θ, cos2θ)（OBB 条件版，`dim` 应为 96）。
    """
    g = image_processor(image, return_tensors="pt")["pixel_values"]
    p = pos_code_from_box(box, image.size, dim=dim, angle=angle).unsqueeze(0)
    return g, p


# --------------------------------------------------------------------------- HR 预处理 / 边界标签


def hr_tensor_from_crop(crop, hr_size: int = HR_SIZE) -> torch.Tensor:
    """原始分辨率 crop → `[3,hr_size,hr_size]` 的 0~1 张量（与 RADIO 侧一致：只 rescale 不 normalize）。

    ⚠️ 必须从**原始 crop** 缩放而来；不能拿已经缩到 128 的 `pixel_values` 上采样
    （那样没有任何新增高频信息）。
    """
    from PIL import Image
    import numpy as np

    if not isinstance(crop, Image.Image):
        raise TypeError("hr_tensor_from_crop 需要 PIL.Image")
    im = crop.convert("RGB")
    if im.size != (hr_size, hr_size):
        im = im.resize((hr_size, hr_size), Image.BILINEAR)
    arr = np.asarray(im, dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def edge_label_from_polygon(polygon, crop_size, edge_size: int = EDGE_SIZE,
                            width: int = 2) -> torch.Tensor:
    """crop 像素坐标下的多边形 → `[1,edge_size,edge_size]` 的 0/1 边界图（训练期辅助监督用）。"""
    from PIL import Image, ImageDraw
    import numpy as np

    canvas = Image.new("L", (edge_size, edge_size), 0)
    if polygon and len(polygon) >= 2:
        sx = edge_size / float(max(crop_size[0], 1))
        sy = edge_size / float(max(crop_size[1], 1))
        pts = [(float(x) * sx, float(y) * sy) for x, y in polygon]
        draw = ImageDraw.Draw(canvas)
        draw.line(pts + [pts[0]], fill=255, width=int(max(1, width)))
    arr = np.asarray(canvas, dtype="float32") / 255.0
    return torch.from_numpy(arr).unsqueeze(0).contiguous()
