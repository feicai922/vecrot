"""**阴影 / 遮挡感知引导增强（ShadowAwareGuide）** —— 针对"屋面被阴影盖住、边界被树冠/邻楼吃掉"。

问题（遥感建筑轮廓的实际难点）
------------------------------
1. **投影阴影**：太阳高度角低时，楼的影子落在自己屋面或邻居身上，亮度整体掉 30~60%，
   且**局部对比度同时下降** → 靠梯度找边界的模型会把影子边缘当楼边界（画大），
   或把被影子压平的檐口当成"没有边界"（塌角/画小）。
2. **遮挡**：树冠、邻楼、云影盖住一段边界 → 该段**没有任何视觉证据**，只能按几何关系外推；
   而 GT 仍然是闭合多边形。

模块做了三件事（全部只作用在 256² 高分辨率分支，输出仍是 `[B,64,1024]`）
---------------------------------------------------------------------
① **光照不变 / 低证据 / 台阶 通道**（解析计算、零标签）：

       gray   = 0.299R+0.587G+0.114B
       mean_k = avgpool(gray, k=31)                 # 局部光照
       ratio  = gray / (mean_k + 1e-3)              # Retinex 式：**低频阴影被除掉，屋顶边缘保留**
       struct = std_k(ratio) / mean          # 归一化后还剩多少结构（真遮挡/无纹理处偏低）
       step   = |grad mean_k(gray)|          # 低频亮度台阶 = 投影阴影自身的边界

   阴影是**低频**光强变化，除上局部均值就没了；建筑边缘是**高频**，除完仍在 → 编码器同时看到
   "原始 RGB"和"阴影不变版"，边界证据不再被阴影抹掉。

② **低证据 / 遮挡图（shadow prior）**：`prior = σ((τ − struct)/T)`，解析可得（不是 GT，是自监督目标）。
   ⚠️ 刻意**不**把"乘性阴影"本身标成不可信：归一化之后那条边还在（冒烟实测光照不变比率把
   阴影造成的灰度差压掉 **208×**），模型应该用它；只有归一化后仍无结构处（树冠/邻楼真遮挡）
   才标为不可信。
   两个用途：作 `ShadowHead` 的自监督目标；**注入到视觉 token**
   （`x += W_s·pool(σ(shadow))`）——让 LLM 知道"这一格不可靠，该更多依赖上下文与几何外推"。

③ **双路边界证据 + 稠密注入**：`edge_raw`（原图头）与 `edge_illum`（光照不变头）概率取平均后
   再做 Top-K 采样（阴影只压住一路时另一路仍给得出边界）；同时把 `edge_feat` 池化到 8×8
   **加到每一个 token**（旧版只塞进 32 个 boundary token，主 token 拿不到"我这格有没有边界"）。

初始化：**零初始化输出投影**（`out/dense_proj/shadow_proj`）→ step0 与基线逐位一致，
但第一步就有梯度（与 `scale_edge.py` 同一约定，修掉 `ls=0` 学不动那个坑）。

配套数据侧：`add_shadow()`（随机投影阴影 / 树冠暗斑），`--shadow-aug-prob` 打开，默认关。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .global_fusion import EDGE_SIZE, HR_SIZE, BoundaryTokenSampler
from .scale_edge import bounded_ls, cap_residual

GRID_TOKENS = 8
SHADOW_TAU = 0.5       # 结构能量阈值（相对全局均值的比例）：低于它判为"遮挡/低证据"
SHADOW_TEMP = 0.2      # 过渡温度
SHADOW_K = 31          # 局部统计窗口（256² 上 ≈ 4 m，与常见楼宽同量级）


# ---------------------------------------------------------------------------
# 解析通道：光照不变证据 + 低证据先验 + 光照台阶
# ---------------------------------------------------------------------------


def _local(gray, k):
    pad = k // 2
    mean = F.avg_pool2d(F.pad(gray, (pad,) * 4, mode="replicate"), k, stride=1)
    mean_sq = F.avg_pool2d(F.pad(gray * gray, (pad,) * 4, mode="replicate"), k, stride=1)
    return mean, (mean_sq - mean * mean).clamp(min=0).sqrt()


def _norm(x):
    return x / (x.mean(dim=(1, 2, 3), keepdim=True) + 1e-6)


def illumination_channels(hr, k=SHADOW_K):
    """`[B,3,H,W]`(0~1) RGB → `[B,4,H,W]` 的四条**解析**证据通道（零标签）。

    | 通道 | 定义 | 为什么有用 |
    |---|---|---|
    | `gray`   | 0.299R+0.587G+0.114B | 基础亮度 |
    | `ratio`  | `gray / (mean_k + 1e-3)` | **Retinex 光照不变**：投影阴影是乘性低频变化，除上局部均值后基本消掉；屋顶边缘（高频）保留 |
    | `struct` | 局部 `std(ratio)`，按全局均值归一 | **结构能量**：光照归一化后还剩多少边界证据；树冠/邻楼真遮挡处显著偏低 |
    | `step`   | `|∇ mean_k(gray)|`，按均值归一 | **低频亮度台阶**：投影阴影**自身边界**长这样。真楼边（反照率跳变）在 `ratio` 里也是边，而纯光照台阶在 `ratio` 里几乎不是边 → 两条通道一起就能把"影子边"和"楼边"分开 |
    """
    hr = hr.to(torch.float32)
    gray = 0.299 * hr[:, 0:1] + 0.587 * hr[:, 1:2] + 0.114 * hr[:, 2:3]
    mean, _std = _local(gray, k)
    ratio = gray / (mean + 1e-3)
    _rm, r_std = _local(ratio, k)
    struct = _norm(r_std)
    padded = F.pad(mean, (1, 1, 0, 0), mode="replicate")
    gx = (padded[..., 2:] - padded[..., :-2]) * 0.5
    padded = F.pad(mean, (0, 0, 1, 1), mode="replicate")
    gy = (padded[:, :, 2:, :] - padded[:, :, :-2, :]) * 0.5
    step = _norm((gx * gx + gy * gy + 1e-12).sqrt())
    return torch.cat([gray, ratio, struct, step], dim=1)


def shadow_prior(hr, tau=SHADOW_TAU, temp=SHADOW_TEMP):
    """`[B,3,H,W]` → `[B,1,H,W]`(0~1) 的"**低证据 / 遮挡**"先验（解析、无标签）。

    定义在**光照归一化之后**的结构能量上（`struct < tau` → 高）。

    这个定义是刻意的：实测乘性阴影**不会**降低 Weber 对比度（`std/mean` 对乘性变化不变），
    所以"阴影"本身不该被标成不可信 —— 归一化之后那条边还在，模型就该用它。
    真正不可信的是**归一化之后仍然没有结构**的地方（树冠/邻楼/云影真遮挡、无纹理屋面），
    那些地方才需要靠几何关系外推。`ShadowHead` 学的就是这张图。
    （投影阴影的**自身边界**另由 `step` 通道显式给出。）
    """
    struct = illumination_channels(hr)[:, 2:3]
    return torch.sigmoid((tau - struct) / max(temp, 1e-4))


# ===========================================================================
# 编码器（7 通道输入）
# ===========================================================================


class ShadowEdgeEncoder(nn.Module):
    """`[B,3,256,256]` RGB →(`edge_feat[B,C,64,64]`, edge_raw, edge_illum, corner, shadow)。

    输入是 **7** 通道：RGB ⊕ (gray, ratio, struct, step)。原始证据、光照不变证据、结构能量、
    低频光照台阶在同一个卷积里对齐，边界头不必在几种表征之间二选一。
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        c = channels
        self.stem = nn.Sequential(
            nn.Conv2d(7, c, 3, stride=2, padding=1), nn.BatchNorm2d(c), nn.GELU())      # 128
        self.block1 = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=2, padding=1), nn.BatchNorm2d(c), nn.GELU())      # 64
        self.block2 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.GELU())                # 64
        self.edge_raw = nn.Conv2d(c, 1, 1)         # 原图证据
        self.edge_illum = nn.Conv2d(c, 1, 1)       # 光照不变证据
        self.corner = nn.Conv2d(c, 1, 1)           # 角点热图（GT 顶点监督）
        self.shadow = nn.Conv2d(c, 1, 1)           # 阴影/低证据（自监督解析目标）

    def forward(self, hr: torch.Tensor):
        extra = illumination_channels(hr)
        x = torch.cat([hr.to(extra.dtype), extra], dim=1)
        x = x.to(self.stem[0].weight.dtype)
        h = self.block2(self.block1(self.stem(x)))
        return h, self.edge_raw(h), self.edge_illum(h), self.corner(h), self.shadow(h)


# ===========================================================================
# 主体模块
# ===========================================================================


class ShadowAwareGuide(nn.Module):
    """阴影/遮挡感知引导：`context tokens [B,Nl,D]` → `enhanced tokens [B,Nl,D]`（恒等初始化）。

        out = x + ls·( dense_edge + dense_shadow + G ⊙ ctx )
        ctx  : Q=x, K/V=boundary tokens（在**双路平均**边界概率的 Top-K 峰值处采样）
        dense_edge   : pool8(edge_feat) → Linear(C→D)
        dense_shadow : pool8(σ(shadow_logits)) → Linear(1→D)    （"这格不可靠"）
    """

    def __init__(self, dim: int = 1024, hr_size: int = HR_SIZE, edge_size: int = EDGE_SIZE,
                 hr_channels: int = 64, heads: int = 8, n_tokens: int = 32, n_uniform: int = 8,
                 ls_init: float = 1.0, gate_init: float = 0.05, dropout: float = 0.0,
                 use_dense_shadow: bool = True, shadow_hidden: int = 64,
                 ls_max: Optional[float] = None, res_cap: Optional[float] = None):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.ls_max, self.res_cap = ls_max, res_cap
        self.hr_size, self.edge_size = hr_size, edge_size
        self.use_dense_shadow = bool(use_dense_shadow)
        self.encoder = ShadowEdgeEncoder(hr_channels)
        self.sampler = BoundaryTokenSampler(hr_channels, dim, n_tokens=n_tokens, n_uniform=n_uniform)
        hid = max(dim // 4, 32)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.gate = nn.Sequential(nn.Linear(2 * dim, hid), nn.GELU(), nn.Linear(hid, dim))
        with torch.no_grad():
            g = min(max(float(gate_init), 1e-4), 1 - 1e-4)
            self.gate[-1].bias.fill_(math.log(g / (1 - g)))
            self.gate[-1].weight.mul_(0.1)
        self.dense_proj = nn.Sequential(nn.LayerNorm(hr_channels), nn.Linear(hr_channels, dim))
        self.shadow_proj = nn.Sequential(nn.Linear(1, shadow_hidden), nn.GELU(),
                                         nn.Linear(shadow_hidden, dim))
        for mod in (self.dense_proj, self.shadow_proj):
            nn.init.zeros_(mod[-1].weight)       # 零初始化输出投影 → step0 恒等、第一步可学
            nn.init.zeros_(mod[-1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
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

    @staticmethod
    def _pool8(feat: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """`[B,C,H,W]` → `[B,n_tokens,C]`（8×8 网格，必要时沿 token 轴对齐）。"""
        pooled = F.adaptive_avg_pool2d(feat, GRID_TOKENS).flatten(2).transpose(1, 2)
        if pooled.shape[1] != n_tokens:
            pooled = F.adaptive_avg_pool1d(pooled.transpose(1, 2), n_tokens).transpose(1, 2)
        return pooled

    def forward(self, x: torch.Tensor, hr_pixel_values: Optional[torch.Tensor] = None,
                aux_out: Optional[dict] = None) -> torch.Tensor:
        if hr_pixel_values is None:              # 没喂 256² → 纯恒等（不崩、不改语义）
            return x
        feat, e_raw, e_illum, corner_logits, shadow_logits = self.encoder(hr_pixel_values)
        edge_logits = 0.5 * (e_raw + e_illum)     # 双路证据平均：阴影只压一路时另一路仍然有效
        shadow_prob = torch.sigmoid(shadow_logits)
        tokens, _prob = self.sampler(feat, edge_logits)
        ctx = self.drop(self._mha(self.norm_q(x), self.norm_kv(tokens)))
        dense_edge = self.dense_proj(self._pool8(feat, x.shape[1]).to(feat.dtype))
        dense_shadow = None
        if self.use_dense_shadow:
            ps = self._pool8(shadow_prob, x.shape[1])                       # [B,64,1]
            dense_shadow = self.shadow_proj(ps.to(feat.dtype))
        g = torch.sigmoid(self.gate(torch.cat([x, ctx], dim=-1)))
        res = cap_residual(bounded_ls(self.ls, self.ls_max)
                           * (dense_edge + g * ctx + (dense_shadow if dense_shadow is not None else 0.0)),
                           x, self.res_cap)
        out = x + res
        if aux_out is not None:
            aux_out["edge_logits"] = edge_logits
            aux_out["edge_raw_logits"] = e_raw
            aux_out["edge_illum_logits"] = e_illum
            aux_out["corner_logits"] = corner_logits
            aux_out["shadow_logits"] = shadow_logits
            aux_out["boundary_tokens"] = int(tokens.shape[1])
            aux_out["shadow_prob_mean"] = float(shadow_prob.mean().detach())
            aux_out["arch_residual_shadow"] = res.pow(2).mean()
        return out


# ===========================================================================
# 损失
# ===========================================================================


def shadow_loss(logits: torch.Tensor, hr_pixel_values: torch.Tensor,
                tau: float = SHADOW_TAU, temp: float = SHADOW_TEMP) -> Tuple[torch.Tensor, float]:
    """`ShadowHead` 对齐解析阴影先验（**自监督**：不需要任何阴影标注）。"""
    with torch.no_grad():
        target = shadow_prior(hr_pixel_values, tau=tau, temp=temp)
        if target.shape[-2:] != logits.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="bilinear",
                                   align_corners=False)
    loss = F.mse_loss(torch.sigmoid(logits), target.to(logits.dtype))
    return loss, float(loss.detach())


# ===========================================================================
# 数据侧：随机投影阴影 / 树冠遮挡（仅训练期）
# ===========================================================================


def add_shadow(crop, polygon_local: Sequence[Sequence[float]], rng,
               strength: float = 0.35, blur: int = 9, mode: str = "cast"):
    """在裁剪图上画一块**投影阴影**（或树冠暗斑），返回新的 PIL.Image。

    * `mode="cast"`：沿随机方向把外接框"投"出一块平行四边形（太阳影子），只压亮度、不动 GT
      —— 教模型"这条边界落在阴影里，也要按几何关系补出来"；
    * `mode="blob"`：在楼附近贴 1~3 个椭圆暗斑（树冠/邻楼遮挡）。

    ⚠️ 只用于**训练**；评测/推理路径完全不调用（`--shadow-aug-prob` 默认 0 = 关）。
    """
    from PIL import Image, ImageDraw, ImageFilter

    if not polygon_local or len(polygon_local) < 3:
        return crop
    W, H = crop.size
    xs = [float(p[0]) for p in polygon_local]
    ys = [float(p[1]) for p in polygon_local]
    bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
    bw, bh = max(bx2 - bx1, 2.0), max(by2 - by1, 2.0)

    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    if mode == "blob":
        for _ in range(rng.randint(1, 3)):
            cx, cy = rng.uniform(bx1, bx2), rng.uniform(by1, by2)
            rx, ry = rng.uniform(0.18, 0.40) * bw, rng.uniform(0.18, 0.40) * bh
            d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    else:
        theta = rng.uniform(0.0, 2.0 * math.pi)
        dx, dy = math.cos(theta) * bw * 1.1, math.sin(theta) * bh * 1.1
        quad = [(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)]
        d.polygon(quad + [(x + dx, y + dy) for x, y in reversed(quad)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(blur))

    dark = 1.0 - float(strength) * rng.uniform(0.75, 1.0)
    arr = crop.convert("RGB")
    shaded = Image.eval(arr, lambda v: int(max(0, min(255, v * dark))))
    return Image.composite(shaded, arr, mask)


# ===========================================================================
# 挂载 / ckpt 兼容
# ===========================================================================


def attach_shadow_guide(model, dim: int = 1024, heads: int = 8, hr_size: int = HR_SIZE,
                        edge_size: int = EDGE_SIZE, hr_channels: int = 64,
                        boundary_tokens: int = 32, boundary_uniform: int = 8,
                        gate_init: float = 0.05, use_dense_shadow: bool = True,
                        ls_max: Optional[float] = None, res_cap: Optional[float] = None,
                        verbose: bool = True) -> Dict:
    """把 `ShadowAwareGuide` 挂成 `boundary_enhance`（与 `--edge-corner` 互斥：同一个 HR 分支）。"""
    inner = getattr(model, "module", model)
    mod = ShadowAwareGuide(dim, hr_size=hr_size, edge_size=edge_size, hr_channels=hr_channels,
                           heads=heads, n_tokens=boundary_tokens, n_uniform=boundary_uniform,
                           gate_init=gate_init, use_dense_shadow=use_dense_shadow,
                           ls_max=ls_max, res_cap=res_cap)
    ref = next(inner.projector.parameters())
    mod = mod.to(device=ref.device, dtype=ref.dtype)
    inner.boundary_enhance = mod
    info = {"shadow_guide": True, "hr_edge": True, "hr_size": int(hr_size),
            "edge_size": int(edge_size), "hr_channels": int(hr_channels),
            "boundary_tokens": int(boundary_tokens), "boundary_uniform": int(boundary_uniform),
            "use_dense_shadow": bool(use_dense_shadow), "dim": int(dim), "heads": int(heads),
            "ls_max": ls_max, "res_cap": res_cap,
            "shadow_params": int(sum(p.numel() for p in mod.parameters()))}
    if verbose:
        print(f"[shadow] ShadowAwareGuide 挂载完成：{info['shadow_params']/1e6:.2f}M 参数"
              f"（7 通道 = RGB + 光照不变/低证据/台阶 4 通道｜双路边界证据 + 阴影图注入｜"
              f"零初始化输出投影 → step0 恒等）", flush=True)
    return info


def maybe_attach_from_ckpt(model, ckpt_dir, verbose: bool = True) -> Optional[Dict]:
    """评测/续训端：`arch.json` 里有 `shadow_guide` 就挂回来（必须在 load_state_dict 之前）。"""
    import json
    from pathlib import Path

    path = Path(ckpt_dir) / "arch.json"
    if not path.exists():
        return None
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not cfg.get("shadow_guide"):
        return None
    return attach_shadow_guide(
        model, dim=int(cfg.get("dim", 1024)), heads=int(cfg.get("heads", 8)),
        hr_size=int(cfg.get("hr_size", HR_SIZE)), edge_size=int(cfg.get("edge_size", EDGE_SIZE)),
        hr_channels=int(cfg.get("hr_channels", 64)),
        boundary_tokens=int(cfg.get("boundary_tokens", 32)),
        boundary_uniform=int(cfg.get("boundary_uniform", 8)),
        use_dense_shadow=bool(cfg.get("use_dense_shadow", True)),
        ls_max=cfg.get("ls_max"), res_cap=cfg.get("res_cap"), verbose=verbose)
