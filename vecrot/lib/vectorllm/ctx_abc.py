"""A/B/C 三个上下文模块（对应 `docs/06_MULTI_MODULE_DESIGNS.md` §3/§4/§5）。

三个模块**互相独立**：同一时间只挂一个，各自与"无模块"的同配方对照比。
共同约定（§2 安全壳，三个模块逐字一致）：

* 输入 `x ∈ [B,64,C]`（RADIO 局部 token + 原有位置编码），输出同形状同 dtype；
  插入点是 projector **之前**，下游 projector / Qwen3 / 坐标词表完全不动。
* 上下文特征 `F ∈ [B,Cf,Hf,Wf]`：整幅**原始瓦片**过同一个冻结 RADIO 得到的空间 token
  （按行优先展平成网格）。`Hf/Wf` 可以是 8×8（瓦片→128）也可以更高（瓦片→384/768），
  由调用方决定并记录；`ctx_dim=Cf`。
* 几何 `geom`：`crop[B,5]=(cx,cy,W,H,angle)`（**转正裁剪窗口**，瓦片像素，y 向下、正角顺时针）
  与 `obj[B,5]`（**未外扩的目标 OBB**）、`tile[B,2]=(W,H)`。裁剪与目标的角度在本项目的 rot
  裁剪里相同，但代码不假设这一点（两者分别传入）。
* 采样：局部 token `i=(a,b)` 的格心 `p=((a+.5)*W/8,(b+.5)*H/8)`，用与 `vectorllm.rbox.crop_to_tile`
  **逐字一致**的逆仿射映回瓦片坐标，再按 `align_corners=False` 口径归一到 `[-1,1]` 交给
  `grid_sample`。越界点（瓦片外 / 特征网格外）显式 mask，**绝不**让 padding 当证据。
* 安全壳：`out` 投影 weight/bias 零初始化；`ls = ls_max · sigmoid(ls_raw)`（初值 0.05）；
  逐 token `‖Δ_i‖ ≤ res_cap·‖X_i‖`（float32 算范数）；无效点 `Δ_i = 0`；
  全部无效时**精确**回退 `Y = X`（不产生 NaN）。
* 仪表盘：`aux_out["arch_residual_ctx"]`（mean Δ²，训练期正则用）、`ctx_valid_frac`、
  `ctx_trunc_frac`（被残差上限截断的比例）、`ctx_gate_mean`、`ctx_h_gate_mean`。

参考实现里只借鉴**抽象思想**（可选择上下文范围 / 先粗后细路由 / 几何偏置进入注意力），
不照搬论文模块，也不声称本文实现等于那些论文的方法。
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LOCAL_GRID = 8          # 128² 局部裁剪 → 8×8=64 个 token（processor/vocab 按此写死）


# --------------------------------------------------------------------------- 安全壳


def bounded_ls(raw: torch.Tensor, ls_max: Optional[float]) -> torch.Tensor:
    """有界 layer-scale：`ls = ls_max · σ(raw)`（`None` 时退回 raw）。"""
    if ls_max is None:
        return raw
    return float(ls_max) * torch.sigmoid(raw)


def cap_residual(res: torch.Tensor, x: torch.Tensor, cap: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """逐 token 残差范数上限：`‖res_j‖ ≤ cap·‖x_j‖`。返回 `(res, 被截断比例)`。

    范数一律在 float32 里算（bf16 下 `norm` 容易溢出/失真），再乘回原 dtype。
    """
    if cap is None or float(cap) <= 0:
        return res, res.new_zeros(())
    dt = res.dtype
    r32, x32 = res.float(), x.float()
    rn = r32.norm(dim=-1, keepdim=True)
    xn = x32.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    factor = (float(cap) * xn / rn.clamp(min=1e-6)).clamp(max=1.0)
    truncated = (factor < 1.0).to(torch.float32).mean()
    return (r32 * factor).to(dt), truncated


# --------------------------------------------------------------------------- 几何（与 rbox.py 逐字一致）


def local_centers(crop_size: torch.Tensor, grid: int = LOCAL_GRID,
                  dtype=torch.float32) -> torch.Tensor:
    """crop 像素格心：`crop_size[B,2]=(W,H)` → `[B,grid*grid,2]`（行优先，与 RADIO 展平顺序一致）。"""
    W = crop_size[:, 0:1]
    H = crop_size[:, 1:2]
    step = torch.arange(grid, device=crop_size.device, dtype=dtype) + 0.5
    xs = step.view(1, 1, grid) * (W / grid).unsqueeze(-1)          # [B,1,grid]
    ys = step.view(1, grid, 1) * (H / grid).unsqueeze(-1)
    gx = xs.expand(-1, grid, -1).reshape(-1, grid * grid)          # 行优先：(a,b) with b inner
    gy = ys.expand(-1, -1, grid).reshape(-1, grid * grid)
    return torch.stack((gx, gy), dim=-1)


def crop_to_tile(pts: torch.Tensor, crop5: torch.Tensor) -> torch.Tensor:
    """crop 像素 → 瓦片像素。`pts[B,N,2]`、`crop5[B,5]=(cx,cy,W,H,angle)`。

    与 `vectorllm.rbox.crop_to_tile` 相同：`x = cx + c·du − s·dv`、`y = cy + s·du + c·dv`
    （`du=u−W/2`, `dv=v−H/2`）。
    """
    ang = crop5[:, 4].unsqueeze(-1)
    c, s = torch.cos(ang).unsqueeze(-1), torch.sin(ang).unsqueeze(-1)
    du = pts[..., 0] - crop5[:, 2:3] / 2.0
    dv = pts[..., 1] - crop5[:, 3:4] / 2.0
    x = crop5[:, 0:1] + c.squeeze(1) * du - s.squeeze(1) * dv
    y = crop5[:, 1:2] + s.squeeze(1) * du + c.squeeze(1) * dv
    return torch.stack((x, y), dim=-1)


def tile_to_norm(pts: torch.Tensor, tile_wh: torch.Tensor) -> torch.Tensor:
    """瓦片像素 → `grid_sample(align_corners=False)` 的归一化坐标 `[-1,1]`。"""
    return pts / tile_wh.unsqueeze(1) * 2.0 - 1.0


def tile_to_obj_local(pts: torch.Tensor, obj5: torch.Tensor) -> torch.Tensor:
    """瓦片像素 → **目标 OBB** 的归一化局部坐标 `(u,v)`：框边 = ±0.5。"""
    ang = obj5[:, 4].unsqueeze(-1)
    c, s = torch.cos(ang), torch.sin(ang)
    dx = pts[..., 0] - obj5[:, 0:1]
    dy = pts[..., 1] - obj5[:, 1:2]
    u = (c * dx + s * dy) / obj5[:, 2:3].clamp(min=1e-6)
    v = (-s * dx + c * dy) / obj5[:, 3:4].clamp(min=1e-6)
    return torch.stack((u, v), dim=-1)


def sample_feature(Fmap: torch.Tensor, norm: torch.Tensor, inside_tile: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """在特征图上双线性采样。`Fmap[B,Cf,Hf,Wf]`、`norm[B,N,2]` → `feat[B,N,Cf]`、`mask[B,N]`。

    越界点用 `padding_mode="zeros"` + 显式 mask 处理（`mask=False` 的值一律不可信）。
    `inside_tile` 是"该点原本是否落在瓦片内"（裁剪窗口越界时会有 False）。
    """
    B, C, Hf, Wf = Fmap.shape
    grid = norm.reshape(B, -1, 1, 2).to(Fmap.dtype)
    sampled = F.grid_sample(Fmap, grid, mode="bilinear", padding_mode="zeros",
                            align_corners=False)
    feat = sampled.reshape(B, C, -1).transpose(1, 2)
    ok = inside_tile.bool() & (norm[..., 0].abs() <= 1.0) & (norm[..., 1].abs() <= 1.0)
    return feat, ok


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """`x[B,N,d]`、`mask[B,N]` → 掩码均值 `[B,d]`（无有效点时给零向量）。"""
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=eps)


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`scores[B,N,M]`、`mask[B,M]` → 掩码 softmax；整行无效时返回全 0（不是 NaN）。"""
    m = mask.unsqueeze(1).expand_as(scores)
    filled = scores.masked_fill(~m, torch.finfo(scores.dtype).min)
    w = filled.softmax(-1) * m.to(scores.dtype)
    return w / w.sum(-1, keepdim=True).clamp(min=1e-6)


# --------------------------------------------------------------------------- 共享壳


class _CtxBase(nn.Module):
    """三个模块共用的残差壳：门控 + 有界 ls + 逐 token 上限 + 仪表盘。"""

    def __init__(self, dim: int, ls_max: Optional[float] = 0.2,
                 res_cap: Optional[float] = 0.08, ls_init: float = 0.05):
        super().__init__()
        self.dim, self.ls_max, self.res_cap = dim, ls_max, res_cap
        raw = 0.0 if ls_init is None else math.log(ls_init / (1 - ls_init)) if 0 < ls_init < 1 else 0.0
        self.ls_raw = nn.Parameter(torch.tensor(float(raw)))
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)      # 零初始化输出投影 → step0 严格等价原模型
        nn.init.zeros_(self.out.bias)

    def _residual(self, x: torch.Tensor, delta_raw: torch.Tensor, valid_tok: torch.Tensor,
                  aux_out: Optional[dict], tag: str) -> torch.Tensor:
        ls = bounded_ls(self.ls_raw, self.ls_max)
        res = ls * self.out(delta_raw)
        vm = valid_tok
        vm = vm.view(-1, 1, 1) if vm.ndim == 1 else (vm.unsqueeze(-1) if vm.ndim == 2 else vm)
        res = res * vm.to(res.dtype)                     # 无效 token 精确置 0（不是"接近 0"）
        res, trunc = cap_residual(res, x, self.res_cap)
        if aux_out is not None:
            with torch.no_grad():
                aux_out[f"arch_residual_{tag}"] = res.float().pow(2).mean()
                aux_out[f"ctx_valid_frac_{tag}"] = float(valid_tok.float().mean())
                aux_out[f"ctx_trunc_frac_{tag}"] = float(trunc)
                aux_out[f"ctx_ls_{tag}"] = float(bounded_ls(self.ls_raw, self.ls_max).detach())
        return x + res, res


def _geom_from_tensors(geom) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`geom` 允许是 dict 或 (crop[B,5], obj[B,5], tile[B,2]) 三元组。"""
    if isinstance(geom, dict):
        return geom["crop"], geom["obj"], geom["tile"]
    crop, obj, tile = geom
    return crop, obj, tile


# --------------------------------------------------------------------------- A：ACDF


class ACDF(_CtxBase):
    """A — 对齐一致性差分融合。

    每个局部 token 在整图特征 `F` 上取**物理同位置**的上下文 `z_i`，与 `q_i` 一起编码
    （含 `q−z`、`q⊙z` 与几何量），逐 token/逐通道门控后残差注入。检验的假设是
    "同位置配对"比"泛化的全局注意力"更有用——而不是"上下文越多越好"。
    """

    def __init__(self, dim: int, ctx_dim: int, hidden: int = 256, use_diff: bool = True,
                 ls_max: Optional[float] = 0.2, res_cap: Optional[float] = 0.08,
                 gate_init: float = 0.05, ls_init: float = 0.05):
        super().__init__(dim, ls_max, res_cap, ls_init)
        self.use_diff = bool(use_diff)
        hid = int(hidden)
        self.norm_x = nn.LayerNorm(dim)
        self.norm_z = nn.LayerNorm(ctx_dim)
        self.q = nn.Linear(dim, hid)
        self.z = nn.Linear(ctx_dim, hid)
        n_in = (4 if use_diff else 2) * hid + 4
        self.e = nn.Sequential(nn.Linear(n_in, hid), nn.GELU(), nn.Linear(hid, hid))
        self.gate = nn.Sequential(nn.Linear(3 * hid, hid), nn.GELU(), nn.Linear(hid, hid))
        self.proj = nn.Linear(hid, dim)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(gate_init / (1 - gate_init)))
        self.kind = "acdf"

    def forward(self, x: torch.Tensor, Fmap: torch.Tensor, geom, aux_out: Optional[dict] = None,
                nb=None, ctx_fp=None) -> torch.Tensor:
        crop, obj, tile = _geom_from_tensors(geom)
        B = x.shape[0]
        if ctx_fp is None:
            ctx_fp = Fmap
        pts = crop_to_tile(local_centers(crop[:, [2, 3]], dtype=torch.float32,
                                         grid=LOCAL_GRID).to(x.device), crop.float()).to(x.dtype)
        inside = ((pts[..., 0] >= 0) & (pts[..., 0] < tile[:, 0:1])
                  & (pts[..., 1] >= 0) & (pts[..., 1] < tile[:, 1:2]))
        z, valid = sample_feature(ctx_fp, tile_to_norm(pts, tile.float()), inside)
        # 几何：目标 OBB 局部坐标 + 距框边的有符号距离 + 尺度
        uv = tile_to_obj_local(pts, obj.float()).to(z.dtype)
        sd = 0.5 - torch.maximum(uv[..., 0].abs(), uv[..., 1].abs())
        scale = (crop[:, 2:3] / obj[:, 2:3].clamp(min=1e-6))
        g = torch.stack((uv[..., 0], uv[..., 1], sd, torch.log(scale).expand_as(sd)), -1)

        q = self.q(self.norm_x(x.to(z.dtype)))
        zz = self.z(self.norm_z(z))
        parts = [q, zz] + ([q - zz, q * zz] if self.use_diff else [])
        e = self.e(torch.cat(parts + [g.to(q.dtype)], -1))
        gate = torch.sigmoid(self.gate(torch.cat((q, zz, e), -1)))
        delta_raw = self.proj(gate * e)
        out, _res = self._residual(x, delta_raw, valid, aux_out, "ctx")
        if aux_out is not None:
            with torch.no_grad():
                aux_out["ctx_gate_mean"] = float(gate.mean())
                aux_out["ctx_pair_valid_frac"] = float(valid.float().mean())
        return out


# --------------------------------------------------------------------------- B：NRS


class NRS(_CtxBase):
    """B — 旋转框方向条件的邻域选择路由。

    以目标 OBB 局部坐标在 `s∈{1.5,3}` 采 `6×6` 点（共 72 个），按 inner/左/右/上/下 与
    尺度分组（≤10 组），组描述做粗路由（soft 权重，可微），组内再对有效 token 做细粒度
    注意力。`grouped=False` 时退化成"同样 72 点的 dense cross-attention"，是必要对照。

    注意：inner/ring 只是**框几何代理**，不是前景/背景或邻楼标签。
    """

    def __init__(self, dim: int, ctx_dim: int, hidden: int = 256, ctx_hidden: int = 128,
                 scales: Tuple[float, ...] = (1.5, 3.0), grid: int = 6, grouped: bool = True,
                 use_pos_bias: bool = True, topk: int = 0,
                 ls_max: Optional[float] = 0.2, res_cap: Optional[float] = 0.08,
                 ls_init: float = 0.05):
        super().__init__(dim, ls_max, res_cap, ls_init)
        self.scales = tuple(float(s) for s in scales)
        self.grid, self.grouped, self.use_pos_bias, self.topk = int(grid), bool(grouped), bool(use_pos_bias), int(topk)
        hid, ch = int(hidden), int(ctx_hidden)
        self.kind = "nrs_grp" if grouped else "nrs_dense"
        self.ctx_proj = nn.Conv2d(ctx_dim, ch, 1)
        self.norm_x = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, hid)
        self.k = nn.Linear(ch, hid)
        self.v = nn.Linear(ch, hid)
        self.kg = nn.Linear(ch, hid)
        self.vg = nn.Linear(ch, hid)
        self.region_emb = nn.Embedding(5, hid)      # inner / left / right / up / down
        self.scale_emb = nn.Linear(len(self.scales), hid)
        self.bias = nn.Sequential(nn.Linear(2 + 2 * hid, hid), nn.GELU(), nn.Linear(hid, 1))
        self.mix = nn.Sequential(nn.Linear(2 * hid, hid), nn.GELU(), nn.Linear(hid, dim))
        self.heads = 4
        self.hd = hid // self.heads
        assert self.hd * self.heads == hid, "hidden 必须能被 heads 整除"

    # ---- 采样点与分组（几何分区，非语义标签）
    def _points(self, device, dtype):
        """返回 `(u, v, scale_idx, region_id)` 的固定点表（局部坐标，框边=0.5）。"""
        us, vs, sidx, rid = [], [], [], []
        for si, s in enumerate(self.scales):
            step = s / self.grid
            for a in range(self.grid):
                for b in range(self.grid):
                    u = -s / 2 + (a + 0.5) * step
                    v = -s / 2 + (b + 0.5) * step
                    us.append(u); vs.append(v); sidx.append(si)
                    if max(abs(u), abs(v)) <= 0.5:
                        rid.append(0)                                   # inner
                    elif abs(u) >= abs(v):
                        rid.append(1 if u < 0 else 2)                   # left / right
                    else:
                        rid.append(3 if v < 0 else 4)                   # up / down（y 向下）
        return (torch.tensor(us, device=device, dtype=dtype), torch.tensor(vs, device=device, dtype=dtype),
                torch.tensor(sidx, device=device), torch.tensor(rid, device=device))

    def forward(self, x: torch.Tensor, Fmap: torch.Tensor, geom, aux_out: Optional[dict] = None,
                nb=None, ctx_fp=None) -> torch.Tensor:
        crop, obj, tile = _geom_from_tensors(geom)
        B, N, _ = x.shape
        if ctx_fp is None:
            ctx_fp = self.ctx_proj(Fmap)
        u, v, sidx, rid = self._points(x.device, torch.float32)
        # 局部坐标 → 瓦片像素（用**目标 OBB**，不是裁剪框：这些点定义在目标坐标系里）
        ang = obj[:, 4].unsqueeze(1)
        px = obj[:, 0:1] + torch.cos(ang) * (u.unsqueeze(0) * obj[:, 2:3]) - torch.sin(ang) * (v.unsqueeze(0) * obj[:, 3:4])
        py = obj[:, 1:2] + torch.sin(ang) * (u.unsqueeze(0) * obj[:, 2:3]) + torch.cos(ang) * (v.unsqueeze(0) * obj[:, 3:4])
        pts = torch.stack((px, py), -1).to(x.dtype)
        inside = ((pts[..., 0] >= 0) & (pts[..., 0] < tile[:, 0:1])
                  & (pts[..., 1] >= 0) & (pts[..., 1] < tile[:, 1:2]))
        feats, valid = sample_feature(ctx_fp, tile_to_norm(pts, tile.float()), inside)   # [B,M,ch]
        # 局部 token 自己的瓦片坐标（用于路由偏置里的相对位置）
        token_pts = crop_to_tile(local_centers(crop[:, [2, 3]], dtype=torch.float32,
                                               grid=LOCAL_GRID).to(x.device), crop.float()).to(x.dtype)

        q = self.q(self.norm_x(x))                                                      # [B,N,hid]
        if not self.grouped:
            K = self.k(feats); Vv = self.v(feats)
            w = masked_softmax(q @ K.transpose(1, 2) / math.sqrt(self.hd * self.heads), valid)
            h = w @ Vv
            delta_raw = self.mix(torch.cat((q, h), -1))
            out, _ = self._residual(x, delta_raw, valid.any(-1), aux_out, "ctx")
            if aux_out is not None:
                with torch.no_grad():
                    aux_out["ctx_valid_frac"] = float(valid.float().mean())
                    aux_out["ctx_gate_mean"] = float(w.max(-1).values.mean())
            return out

        # 分组：G = 尺度 × 区域；空组由 mask 处理（不产生 NaN，也不参与 softmax）
        G = len(self.scales) * 5
        gid = sidx * 5 + rid                                                           # [M]
        onehot = torch.nn.functional.one_hot(gid, G).to(x.dtype)                        # [M,G]
        uf = valid.to(x.dtype)                                                          # [B,M]
        counts_raw = uf @ onehot                                                        # [B,G]
        gmask = counts_raw > 0                                                          # [B,G]（先判空，再 clamp！）
        counts = counts_raw.clamp(min=1e-6)                                             # [B,G]
        gsum = (uf.unsqueeze(-1) * feats).transpose(1, 2) @ onehot                       # [B,ch,G]
        gmean = (gsum / counts.unsqueeze(1)).transpose(1, 2)                             # [B,G,ch]
        scores_all = q @ self.k(feats).transpose(1, 2) / math.sqrt(self.hd * self.heads)  # [B,N,M]
        Vv = self.v(feats)
        per_group = []
        for g in range(G):
            m = (gid == g).unsqueeze(0) & valid                                            # [B,M]
            per_group.append(masked_softmax(scores_all, m) @ Vv)                           # [B,N,hid]
        Vg = torch.stack(per_group, 2)                                                     # [B,N,G,hid]
        # 组级路由打分 + 位置/尺度/区域偏置（组内点均值 → 组位置；token 自身位置作参考点）
        gscore = (q @ self.kg(gmean).transpose(1, 2)) / math.sqrt(self.hd * self.heads)     # [B,N,G]
        if self.use_pos_bias:
            uv_s = tile_to_obj_local(pts, obj.float()).to(x.dtype)                          # [B,M,2]
            uv_t = tile_to_obj_local(token_pts, obj.float()).to(x.dtype)                     # [B,N,2]
            gxy = torch.einsum("bm,bmk,mg->bgk", uf, uv_s, onehot) / counts.unsqueeze(-1)    # [B,G,2]
            du = uv_t.unsqueeze(2) - gxy.unsqueeze(1)                                       # [B,N,G,2]
            gsc = torch.zeros(B, G, len(self.scales), device=x.device, dtype=x.dtype)
            gsc[:, torch.arange(G, device=x.device), torch.arange(G, device=x.device) // 5] = 1.0
            rid_of_group = torch.arange(G, device=x.device) % 5                              # 组 → 区域
            region_vec = self.region_emb(rid_of_group).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
            scale_vec = self.scale_emb(gsc).unsqueeze(1).expand(-1, N, -1, -1)
            bias_in = torch.cat((du, region_vec, scale_vec), -1)                            # [B,N,G,2+2*hid]
            gscore = gscore + self.bias(bias_in).squeeze(-1)
        gw = masked_softmax(gscore, gmask)                                                  # [B,N,G]
        h = (gw.unsqueeze(-1) * Vg).sum(2)                                                  # [B,N,hid]
        if self.topk and 0 < self.topk < G:                                                 # 可选：只留 top-k（仍用软权重）
            keep = gscore.topk(self.topk, dim=-1).indices
            m = torch.zeros_like(gw, dtype=torch.bool).scatter_(-1, keep, True) & gmask.unsqueeze(1)
            gw = masked_softmax(gscore, m.expand(B, N, G))
            h = (gw.unsqueeze(-1) * Vg).sum(2)
        delta_raw = self.mix(torch.cat((q, h), -1))
        out, _ = self._residual(x, delta_raw, gmask.any(-1), aux_out, "ctx")
        if aux_out is not None:
            with torch.no_grad():
                aux_out["ctx_valid_frac"] = float(valid.float().mean())
                aux_out["ctx_gate_mean"] = float(gw.max(-1).values.mean())
                aux_out["ctx_empty_group_frac"] = float((~gmask).float().mean())
                aux_out["ctx_group_entropy"] = float(
                    (-(gw.clamp_min(1e-6).log() * gw).sum(-1)).mean())
        return out


# --------------------------------------------------------------------------- C：NCR


class NCR(_CtxBase):
    """C — 预测邻居条件的竞争关系融合。

    输入是**检测器预测框**（不是 GT）：目标框 `b0` + 同瓦片最近 `K≤6` 个邻居框及其分数。
    每个邻居框做旋转 ROI `3×3` 采样并 mask 均值 → `r_j`；目标同样得到 `r_0`。相对几何
    （中心差、宽高比、`sin2Δθ/cos2Δθ`、detector score）进注意力打分。**无邻居时精确回退 identity**。

    ⚠️ 邻居是"竞争实例"而不是"负类"：模块只把它们作为条件输入，不把视觉特征往反方向推。
    """

    def __init__(self, dim: int, ctx_dim: int, hidden: int = 256, ctx_hidden: int = 128,
                 roi: int = 3, ls_max: Optional[float] = 0.2, res_cap: Optional[float] = 0.08,
                 ls_init: float = 0.05):
        super().__init__(dim, ls_max, res_cap, ls_init)
        hid, ch = int(hidden), int(ctx_hidden)
        self.roi = int(roi)
        self.kind = "ncr"
        self.ctx_proj = nn.Conv2d(ctx_dim, ch, 1)
        self.norm_x = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, hid)
        self.k = nn.Linear(ch, hid)
        self.v = nn.Linear(ch, hid)
        self.t = nn.Linear(ch, hid)
        self.geo = nn.Sequential(nn.Linear(7, hid), nn.GELU(), nn.Linear(hid, hid))
        self.mix = nn.Sequential(nn.Linear(5 * hid, hid), nn.GELU(), nn.Linear(hid, dim))

    def _roi_mean(self, ctx_fp, boxes, tile, roi):
        """旋转框内 `roi×roi` 采样 + mask 均值。boxes `[B,K,5]` → `[B,K,ch]`、`[B,K]`。"""
        B, K, _ = boxes.shape
        step = (torch.arange(roi, device=boxes.device, dtype=boxes.dtype) + 0.5) / roi - 0.5
        gy, gx = torch.meshgrid(step, step, indexing="ij")
        gx = gx.reshape(-1).view(1, 1, -1)                                          # [1,1,roi²]
        gy = gy.reshape(-1).view(1, 1, -1)
        ang = boxes[..., 4:5]
        c, s = torch.cos(ang), torch.sin(ang)
        dx = gx * boxes[..., 2:3]
        dy = gy * boxes[..., 3:4]
        px = boxes[..., 0:1] + c * dx - s * dy                                      # [B,K,roi²]
        py = boxes[..., 1:2] + s * dx + c * dy
        pts = torch.stack((px, py), -1)
        inside = ((pts[..., 0] >= 0) & (pts[..., 0] < tile[:, 0:1].unsqueeze(1))
                  & (pts[..., 1] >= 0) & (pts[..., 1] < tile[:, 1:2].unsqueeze(1)))
        feat, ok = sample_feature(ctx_fp, tile_to_norm(pts.reshape(B, K * roi * roi, 2).to(boxes.dtype),
                                                       tile.float()),
                                  inside.reshape(B, K * roi * roi))
        feat = feat.view(B, K, roi * roi, -1)
        ok = ok.view(B, K, roi * roi)
        m = ok.to(feat.dtype)
        pooled = (feat * m.unsqueeze(-1)).sum(2) / m.sum(2).clamp(min=1e-6).unsqueeze(-1)
        return pooled, ok.any(-1)

    def forward(self, x: torch.Tensor, Fmap: torch.Tensor, geom, aux_out: Optional[dict] = None,
                nb=None, ctx_fp=None) -> torch.Tensor:
        """`nb` = dict(boxes[B,K,5], scores[B,K], mask[B,K])；`mask=False` 表示无该邻居。"""
        crop, obj, tile = _geom_from_tensors(geom)
        if nb is None or nb.get("boxes") is None or nb["boxes"].shape[1] == 0:
            if aux_out is not None:
                with torch.no_grad():
                    aux_out["arch_residual_ctx"] = x.new_zeros(())
                    aux_out["ctx_valid_frac_ctx"] = 0.0
            return x
        if ctx_fp is None:
            ctx_fp = self.ctx_proj(Fmap)
        boxes = nb["boxes"].to(x.device)
        scores = nb.get("scores")
        nbmask = nb.get("mask", torch.ones(boxes.shape[:2], device=boxes.device, dtype=torch.bool))
        nbmask = nbmask.to(device=x.device).bool()
        r_j, ok_j = self._roi_mean(ctx_fp, boxes, tile, self.roi)                    # [B,K,ch]
        r_0, _ = self._roi_mean(ctx_fp, obj[:, None, :], tile, self.roi)             # [B,1,ch]
        r_0 = r_0[:, 0]
        valid = nbmask & ok_j

        ang0 = obj[:, 4:5]
        dx = boxes[..., 0] - obj[:, 0:1]
        dy = boxes[..., 1] - obj[:, 1:2]
        c, s = torch.cos(ang0), torch.sin(ang0)
        relu = (c * dx + s * dy) / obj[:, 2:3].clamp(min=1e-6)
        relv = (-s * dx + c * dy) / obj[:, 3:4].clamp(min=1e-6)
        dang = boxes[..., 4] - obj[:, 4:5]
        geo = torch.stack((relu, relv,
                           torch.log((boxes[..., 2] / obj[:, 2:3].clamp(min=1e-6)).clamp(min=1e-3)),
                           torch.log((boxes[..., 3] / obj[:, 3:4].clamp(min=1e-6)).clamp(min=1e-3)),
                           torch.sin(2 * dang), torch.cos(2 * dang),
                           (scores.to(x.device) if scores is not None
                            else torch.zeros_like(boxes[..., 0]))), -1)

        q = self.q(self.norm_x(x))
        K = self.k(r_j) + self.geo(geo.to(q.dtype))
        a = masked_softmax(q @ K.transpose(1, 2) / math.sqrt(K.shape[-1]), valid)
        n = a @ self.v(r_j)
        t = self.t(r_0).unsqueeze(1).expand_as(n)
        h = self.mix(torch.cat((q, t, n, t - n, q * n), -1))
        out, _ = self._residual(x, h, valid.any(-1), aux_out, "ctx")
        if aux_out is not None:
            with torch.no_grad():
                aux_out["ctx_gate_mean"] = float(a.max(-1).values.mean())
                aux_out["ctx_n_neighbors"] = float(valid.float().sum(-1).mean())
                aux_out["ctx_no_neighbor_frac"] = float((~valid.any(-1)).float().mean())
        return out


# --------------------------------------------------------------------------- 挂载


KINDS = ("acdf", "nrs", "nrs_grp", "nrs_dense", "ncr")


def build_context_module(kind: str, dim: int, ctx_dim: int, **kw):
    """按名字建模块。`kw` 里多余的键（未在该模块出现的）会被忽略并返回 warning 列表。"""
    kind = str(kind).lower()
    if kind == "nrs_grp":       # NRS(grouped=True) 的存档 kind → 归一化
        kind = "nrs"
    allowed = {
        "acdf": {"hidden", "use_diff", "ls_max", "res_cap", "gate_init", "ls_init"},
        "nrs": {"hidden", "ctx_hidden", "scales", "grid", "grouped", "use_pos_bias", "topk",
                "ls_max", "res_cap", "ls_init"},
        "ncr": {"hidden", "ctx_hidden", "roi", "ls_max", "res_cap", "ls_init"},
    }[("nrs" if kind == "nrs_dense" else kind)]
    picked = {k: v for k, v in kw.items() if k in allowed and v is not None}
    ignored = sorted(set(kw) - set(picked))
    if kind == "acdf":
        mod = ACDF(dim, ctx_dim, **picked)
    elif kind in ("nrs", "nrs_dense"):
        picked.setdefault("grouped", kind == "nrs")
        mod = NRS(dim, ctx_dim, **picked)
    elif kind == "ncr":
        mod = NCR(dim, ctx_dim, **picked)
    else:
        raise ValueError(f"unknown context module kind: {kind}")
    return mod, ignored


def attach_context_module(model, kind: str, ctx_dim: int, **kw):
    """把上下文模块挂到 `model.context_module` 上（在 `load_state_dict` **之前** 调用）。"""
    inner = getattr(model, "module", model)
    dim = inner.projector.in_features if hasattr(inner.projector, "in_features") else None
    if dim is None:
        dim = getattr(inner, "visual_hidden_size", None) or inner.visual_pos_embeddings.weight.shape[1]
    mod, ignored = build_context_module(kind, int(dim), int(ctx_dim), **kw)
    inner.context_module = mod
    info = {"kind": mod.kind, "dim": int(dim), "ctx_dim": int(ctx_dim),
            "params": int(sum(p.numel() for p in mod.parameters())),
            "ls_max": mod.ls_max, "res_cap": mod.res_cap, "ignored": ignored}
    return info


# --------------------------------------------------------------------------- 数据侧工具


def tile_tensor_for_ctx(image, size: int = 384) -> torch.Tensor:
    """整幅瓦片 → `[3,size,size]` 的 0~1 张量（只 rescale 不 normalize，与视觉塔侧一致）。

    ⚠️ 这里**直接**把瓦片缩到 `size`（不像 crop 那样先 432 再 128）。理由：瓦片本身是
    1024²，两级缩放只会再糊一次；`docs/06` §7 阶段0 要求记录实际分辨率与网格尺寸，
    所以 `size` 必须显式写进实验记录（8×8 与 48×48 的信息量差一个量级）。
    """
    import numpy as np
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError("tile_tensor_for_ctx 需要 PIL.Image")
    im = image.convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(im, dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def ctx_geometry(crop_center, crop_size, crop_angle: float, obj, tile_size) -> torch.Tensor:
    """→ `[12]`：`crop(cx,cy,W,H,angle)` + `obj(cx,cy,w,h,angle)` + `tile(W,H)`。"""
    return torch.tensor([
        float(crop_center[0]), float(crop_center[1]),
        float(crop_size[0]), float(crop_size[1]), float(crop_angle),
        float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3]), float(obj[4]),
        float(tile_size[0]), float(tile_size[1]),
    ], dtype=torch.float32)


def geom_split(ctx_geom: torch.Tensor):
    """`ctx_geom[B,12]` → `dict(crop[B,5], obj[B,5], tile[B,2])`（模块的 geom 入参）。"""
    return {"crop": ctx_geom[:, 0:5], "obj": ctx_geom[:, 5:10], "tile": ctx_geom[:, 10:12]}


def select_neighbors(boxes, scores, obj, k: int = 6, self_iou: float = 0.5):
    """从同瓦片的**检测器预测框**里挑目标最近的邻居。

    规则（`docs/06` §5）：先排除与目标框 IoU>0.5 的"自身 proposal"（**不能**按 IoU 排除所有
    重叠框，那会误删贴邻建筑），其余按中心距离升序取前 `k` 个；HBB 一律转成 `θ=0` 的 OBB。

    返回 `([k,5] float32, [k] float32, [k] bool)`。
    """
    import numpy as np

    ox, oy, ow, oh = float(obj[0]), float(obj[1]), float(obj[2]), float(obj[3])
    cand = []
    for b, s in zip(boxes, scores):
        x1, y1, x2, y2 = [float(v) for v in b[:4]]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = abs(x2 - x1), abs(y2 - y1)
        if w <= 1 or h <= 1:
            continue
        ix = max(0.0, min(x2, ox + ow / 2) - max(x1, ox - ow / 2))
        iy = max(0.0, min(y2, oy + oh / 2) - max(y1, oy - oh / 2))
        inter = ix * iy
        iou = inter / max(w * h + ow * oh - inter, 1e-6)
        if iou > self_iou:          # 自身 proposal，跳过
            continue
        d = math.hypot(cx - ox, cy - oy)
        cand.append((d, [cx, cy, w, h, 0.0], float(s)))
    cand.sort(key=lambda t: t[0])
    out_b = np.zeros((k, 5), dtype="float32")
    out_s = np.zeros((k,), dtype="float32")
    out_m = np.zeros((k,), dtype=bool)
    for i, (_, bb, ss) in enumerate(cand[:k]):
        out_b[i] = bb
        out_s[i] = ss
        out_m[i] = True
    return torch.from_numpy(out_b), torch.from_numpy(out_s), torch.from_numpy(out_m)


def maybe_attach_from_ckpt(model, ckpt: str, verbose: bool = False):
    """评测端：按 ckpt 里 `arch.json["ctx_module"]` 的配置把模块挂回去（必须在 load_state_dict 之前）。"""
    import json as _json
    from pathlib import Path as _Path

    arch = _Path(ckpt) / "arch.json"
    if not arch.exists():
        return None
    cfg = _json.loads(arch.read_text(encoding="utf-8")).get("ctx_module")
    if not cfg:
        return None
    info = attach_context_module(model, cfg["kind"], ctx_dim=int(cfg["ctx_dim"]),
                                 ls_max=cfg.get("ls_max"), res_cap=cfg.get("res_cap"))
    inner = getattr(model, "module", model)
    _ref = next(inner.projector.parameters())
    inner.context_module = inner.context_module.to(device=_ref.device, dtype=_ref.dtype)
    if verbose:
        print(f"[vl] 上下文模块 {info['kind']} 已挂载：{info['params']/1e6:.2f}M 参数", flush=True)
    return info
