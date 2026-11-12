"""GlobalFusion v2：修复三处结构缺陷的加性门控融合（2026-11-08 深夜）。

针对全量证据的三个诊断（docs/17 §补充、docs/18 §0）：
1. ls1 慢启动：0.001 起步 + cap_residual 双重钳制 → 600 步只到 0.002~0.007，attention 通路从未打开；
2. ls2(FFN) 与 ls1(attention) 竞争：梯度流向更好学的 FFN，attention 被抢信号；
3. cross-attention 输出与局部特征近正交（cos=0.14）：K/V 特征分布与 local 不兼容，
   ls1 学不起来是模型在主动抑制"噪声注入"。

修复（参考 sota_idea PlugNPlay 的 CGA/UAFF 门控融合范式）：
A. **零初始化 out 投影替代 ls1 慢启动**（scale_edge.py 已验证的模式）：
   `ctx = self.out(attn)`，`out.weight=0` → step0 恒等但 ∂L/∂(q,k,v) ≠ 0，第一步就学；
B. **attention 输出走 LayerNorm 对齐**（修问题 3 的分布不匹配）；
C. **FFN 独立成块、零初始化投影**（修问题 2 的竞争：两通路不再共享残差预算，
   各自的 out 投影独立零初始化，谁有用谁自己长）；
D. **门控做逐 token 空间加权**（CGA 的 pixel attention 思路）而非逐通道：
   局部 token [B,64,D] 与全局 K/V 网格 [B,64,D] 天然空间对齐，
   `G = sigmoid(Conv([local_grid, ctx_grid]))` 按空间位置决定"哪里信全局、哪里信局部"。

参数量与 v1 相当（~10.3M），token 数不变（64），与旧 ckpt 的 arch.json 兼容加载（v1 权重可热启动）。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

POS_DIM = 64
POS_DIM_ANGLE = 96


class GlobalFusionV2(nn.Module):
    """目标感知加性门控融合 v2：局部 Q × 全局 K/V → 零初始化输出 → 逐 token 空间门控。

    与 v1 的关键区别（见模块 docstring 的四点 A-D）。
    """

    def __init__(self, dim: int, heads: int = 8, pos_dim: int = POS_DIM,
                 ls_init: float = 0.0, dropout: float = 0.0,
                 gate_init: float = 0.05, use_gate: bool = True,
                 use_spatial_bias: bool = False, spatial_heads: int = 8,
                 ls_max: Optional[float] = None, res_cap: Optional[float] = None,
                 pos_code_dim: Optional[int] = None):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.pos_dim = int(pos_dim)
        self.ls_max, self.res_cap = ls_max, res_cap
        self.use_gate = use_gate
        self.use_spatial_bias = use_spatial_bias
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)          # 修复 B：attention 输出分布对齐
        self.pos_mlp = nn.Sequential(nn.Linear(self.pos_dim, dim), nn.GELU(),
                                     nn.Linear(dim, dim))
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)             # 修复 A：零初始化 out 投影
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # 修复 C：FFN 独立零初始化投影（不再与 attention 共享 ls 预算）
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU())
        self.ffn_out = nn.Linear(dim * 2, dim)
        nn.init.zeros_(self.ffn_out.weight)
        nn.init.zeros_(self.ffn_out.bias)
        self.drop = nn.Dropout(dropout)
        # 修复 D：逐 token 空间门控（CGA pixel attention；local 8×8 网格与 K/V 网格对齐）
        if use_gate:
            hid = max(dim // 4, 32)
            self.gate = nn.Sequential(nn.Linear(2 * dim, hid), nn.GELU(),
                                      nn.Linear(hid, 1))
            g = min(max(float(gate_init), 1e-4), 1 - 1e-4)
            with torch.no_grad():
                self.gate[-1].bias.fill_(math.log(g / (1 - g)))
                self.gate[-1].weight.mul_(0.1)
        self.side = int(round(math.sqrt(64)))       # 8×8 token 网格
        # 兼容 v1 ckpt 的字段名（旧 ckpt 的 ls1/ls2 加载时被忽略，无 unexpected 冲突）
        self.ls1 = nn.Parameter(torch.zeros(1))
        self.ls2 = nn.Parameter(torch.zeros(1))
        # 兼容 v1 的 spatial bias（保留可选）
        if use_spatial_bias:
            self.spatial_mlp = nn.Sequential(nn.Linear(4, 64), nn.GELU(),
                                             nn.Linear(64, spatial_heads))
            self.spatial_heads = spatial_heads

    def _mha(self, q, kv, attn_bias: Optional[torch.Tensor] = None):
        b, n, c = q.shape
        hd = c // self.heads
        q = self.q(q).view(b, n, self.heads, hd).transpose(1, 2)
        k = self.k(kv).view(b, kv.shape[1], self.heads, hd).transpose(1, 2)
        v = self.v(kv).view(b, kv.shape[1], self.heads, hd).transpose(1, 2)
        mask = None
        if attn_bias is not None:
            mask = attn_bias[:, :, None, :].expand(-1, -1, n, -1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return o.transpose(1, 2).reshape(b, n, c)

    def forward(self, local: torch.Tensor, glob: torch.Tensor,
                pos_code: Optional[torch.Tensor] = None,
                aux_out: Optional[dict] = None,
                pos_box: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, c = local.shape
        q = self.norm_q(local)
        if pos_code is not None:
            q = q + self.pos_mlp(pos_code.to(q.dtype)).unsqueeze(1)
        bias = None
        if self.use_spatial_bias and pos_box is not None:
            bias = self._spatial_bias(pos_box, glob.shape[1], local.dtype)
        attn_out = self._mha(q, self.norm_kv(glob), bias)
        ctx = self.out(self.norm_ctx(attn_out))      # 零初始化：step0 ctx≡0
        if self.use_gate and ctx is not None:
            g = torch.sigmoid(self.gate(torch.cat([local, ctx], dim=-1)))   # [B,64,1] 逐 token
            ctx = g * ctx
            if aux_out is not None:
                aux_out["fusion_gate_mean"] = float(g.mean().detach())
        ffn_out = self.ffn_out(self.ffn(local))       # 独立零初始化 FFN 通路
        from .scale_edge import cap_residual
        res = ctx + ffn_out
        res = cap_residual(res, local, self.res_cap)
        if aux_out is not None:
            aux_out["arch_residual_global"] = res.pow(2).mean()
        return local + res

    def _spatial_bias(self, pos_box, n_global: int, dtype):
        """v1 的空间偏置逻辑（网格相对几何 → head 偏置），保留兼容。"""
        if pos_box is None or n_global <= 0:
            return None
        n_side = int(round(math.sqrt(n_global)))
        if n_side * n_side != n_global:
            return None
        b = pos_box.shape[0]
        cx, cy, w, h = (pos_box[:, i] for i in range(4))
        idx = torch.arange(n_side, device=pos_box.device, dtype=torch.float32)
        gx = ((idx + 0.5) / n_side)[None, None, :].expand(b, n_side, n_side)
        gy = gx.transpose(1, 2)
        cxb, cyb = cx[:, None, None], cy[:, None, None]
        wb = (0.5 * w)[:, None, None].clamp(min=1e-4)
        hb = (0.5 * h)[:, None, None].clamp(min=1e-4)
        feat = torch.stack([gx - cxb, gy - cyb, (gx - cxb) / wb, (gy - cyb) / hb], dim=-1)
        feat = feat.clamp(-8.0, 8.0).reshape(b, n_global, 4)
        return self.spatial_mlp(feat.to(dtype)).transpose(1, 2)


def maybe_attach_from_ckpt(model, ckpt_dir, verbose: bool = True):
    """评测端：arch.json 标记 global_fusion_v2 时挂 v2（普通版走 global_fusion.py）。"""
    import json
    from pathlib import Path

    path = Path(ckpt_dir) / "arch.json"
    if not path.exists():
        return None
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not bool(cfg.get("global_fusion_v2", False)):
        return None
    info = attach_global_fusion_v2(
        model, dim=int(cfg.get("dim", 1024)), heads=int(cfg.get("heads", 8)),
        pos_dim=int(cfg.get("pos_dim", 64)), ls_init=float(cfg.get("ls_init", 0.0)),
        verbose=verbose, gate_init=float(cfg.get("gate_init", 0.05)),
        use_gate=bool(cfg.get("use_gate", True)),
        use_spatial_bias=bool(cfg.get("use_spatial_bias", False)),
        ls_max=cfg.get("ls_max"), res_cap=cfg.get("res_cap"))
    return info


def attach_global_fusion_v2(model, dim: int = 1024, heads: int = 8, pos_dim: int = POS_DIM,
                            ls_init: float = 0.0, verbose: bool = True,
                            gate_init: float = 0.05, use_gate: bool = True,
                            use_spatial_bias: bool = False,
                            ls_max: Optional[float] = None,
                            res_cap: Optional[float] = None) -> dict:
    """挂 GlobalFusionV2（trainable），返回写进 arch.json 的配置。"""
    inner = getattr(model, "module", model)
    inner.global_fusion = GlobalFusionV2(dim, heads=heads, pos_dim=pos_dim,
                                         ls_init=ls_init, gate_init=gate_init,
                                         use_gate=use_gate,
                                         use_spatial_bias=use_spatial_bias,
                                         ls_max=ls_max, res_cap=res_cap)
    ref = next(inner.projector.parameters())
    inner.global_fusion = inner.global_fusion.to(device=ref.device, dtype=ref.dtype)
    n = sum(p.numel() for p in inner.global_fusion.parameters())
    info = {"global_fusion_v2": True, "global_fusion": True, "dim": dim, "heads": heads,
            "pos_dim": int(pos_dim), "pos_angle": bool(int(pos_dim) == POS_DIM_ANGLE),
            "ls_init": float(ls_init), "gate_init": gate_init, "use_gate": bool(use_gate),
            "use_spatial_bias": bool(use_spatial_bias), "params": int(n),
            "ls_max": ls_max, "res_cap": res_cap}
    if verbose:
        print(f"[fusion-v2] 挂载完成：{n/1e6:.2f}M 参数（out/ffn_out 零初始化，"
              f"step0 与基线逐位一致，attention 通路第一步即可学）", flush=True)
    return info
