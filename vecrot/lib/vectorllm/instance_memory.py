"""跨实例空间记忆（Instance Memory）：让 Qwen 感知同图已完成建筑的占用与边界。

设计依据（docs/18_INSTANCE_MEMORY_DESIGN_2026-11-08.md）：
* 全量证据表明 GF 的 cross-attention 通路（ls1≈0.002~0.007）没有打开，四种 K/V 来源全负；
  历史实例信息不走 attention，走 ScaleCond 已验证有效的**加性 FiLM 通路**。
* 编码：已完成多边形栅格化为 64×64 占用图 + 当前 rbox 边界图 → 双通道 depthwise CNN
  → 8×8 网格的 64 个记忆 token → 逐通道门控加性注入（gate 零初始化，step0 恒等）。
* 约束：不增加视觉 token 数（64 不变）、不改 RADIO/Qwen 结构、参数 ~0.1M。

坐标约定：与 vectorllm.rbox 完全一致——
* `occ_canvas` 是瓦片坐标系的全图占用（推理期逐框累积）；
* warp 到裁剪框坐标用 rbox_crop 的同一仿射矩阵（几何零成本）。
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEM_SIZE = 64      # 与 RADIO 8×8 patch token 数一致（不增加序列长度）
MEM_SIDE = 8       # 64 = 8×8


def rasterize_polys(polys: Sequence[Sequence[Sequence[float]]],
                    h: int, w: int, weight: float = 1.0) -> np.ndarray:
    """瓦片（或裁剪框）坐标的多边形列表 → float32 占用图 `[h,w]`。

    `weight < 1` 用于推理期低置信度多边形的衰减写入（防止误差累积）。
    """
    from PIL import Image, ImageDraw

    if not polys:
        return np.zeros((h, w), dtype=np.float32)
    canvas = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(canvas)
    for poly in polys:
        pts = [(float(p[0]), float(p[1])) for p in poly]
        if len(pts) >= 3:
            d.polygon(pts, fill=255)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr * float(weight)


def warp_occ_to_crop(occ_tile: np.ndarray, transform: dict,
                     crop_w: int, crop_h: int) -> np.ndarray:
    """瓦片坐标占用图 → 裁剪框坐标占用图（复用 fast_rbox_crop 的仿射矩阵）。

    与 `rbox.py::crop_to_tile` 互逆：crop = R(-ang)·(tile-c)+center 的窗口采样。
    """
    import cv2

    if not transform.get("ok"):
        return np.zeros((crop_h, crop_w), dtype=np.float32)
    cx, cy = transform["c"]
    ang = transform["angle"]
    W, H = transform["size"]
    c, s = np.cos(ang), np.sin(ang)
    # dst(裁剪)→src(瓦片)：src = R(ang)·(u-W/2, v-H/2) + (cx,cy)，与 fast_rbox_crop 相同
    M = np.array([[c, -s, cx - c * W / 2.0 + s * H / 2.0],
                  [s, c, cy - s * W / 2.0 - c * H / 2.0]], dtype=np.float64)
    out = cv2.warpAffine(occ_tile.astype(np.float32), M, (W, H),
                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return out.astype(np.float32)


def rbox_edge_map(crop_w: int, crop_h: int, rbox, expand: float = 1.45,
                  width: int = 2) -> np.ndarray:
    """当前 rbox（expand 后）在裁剪框内的边界图 `[crop_h,crop_w]`。"""
    from PIL import Image, ImageDraw

    cx, cy, w, h = crop_w / 2.0, crop_h / 2.0, float(rbox[2]) * expand, float(rbox[3]) * expand
    canvas = Image.new("L", (crop_w, crop_h), 0)
    d = ImageDraw.Draw(canvas)
    d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                outline=255, width=max(1, int(width)))
    return np.asarray(canvas, dtype=np.float32) / 255.0


class InstanceMemory(nn.Module):
    """双通道（占用+边界）64×64 → depthwise CNN → 64 token 加性注入。

    输入 `[B,2,64,64]`（0=历史占用，1=当前框边界）；输出与 local 同形的加性修正。
    门控零初始化 → step0 与基线逐位一致，可从任意 ckpt 热启动。
    """

    def __init__(self, dim: int = 1024, hidden: int = 64, ls_max: Optional[float] = None,
                 res_cap: Optional[float] = None):
        super().__init__()
        self.dim = dim
        self.ls_max, self.res_cap = ls_max, res_cap
        self.stem = nn.Sequential(
            nn.Conv2d(2, hidden, 3, padding=1, groups=1), nn.GELU(),      # 融合双通道
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),  # depthwise 几何
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.GELU(),
        )
        # 8×8 网格 → 64 token（与视觉 token 一一对应，不改序列长度）
        self.proj = nn.Linear(hidden, dim)
        # 逐通道门控：零初始化线性输出（非 sigmoid——sigmoid(0)=0.5 会破坏 step0 恒等）。
        # 零初始化 → step0 严格恒等，但 ∂(res)/∂(gate_mlp 权重) ≠ 0，第一步就能学。
        self.gate_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                      nn.Linear(hidden, dim))
        nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(self, local: torch.Tensor, occ_edge: Optional[torch.Tensor],
                aux_out: Optional[dict] = None) -> torch.Tensor:
        """local `[B,64,D]`；occ_edge `[B,2,64,64]`（None = 恒等返回）。"""
        if occ_edge is None:
            return local
        b = local.shape[0]
        f = self.stem(occ_edge.to(self.proj.weight.dtype))          # [B,C,64,64]
        f = F.avg_pool2d(f, kernel_size=64 // MEM_SIDE)             # [B,C,8,8] → 对齐视觉网格
        grid = f.permute(0, 2, 3, 1).reshape(b, MEM_SIZE, -1)       # [B,64,C]
        mem = self.proj(grid)                                        # [B,64,D]
        g = torch.tanh(self.gate_mlp(grid))                         # [B,64,D] 零初始化=0，±1 限幅
        res = g * mem
        from .scale_edge import bounded_ls, cap_residual
        res = cap_residual(res, local, self.res_cap)
        if aux_out is not None:
            aux_out["instance_memory_gate"] = float(g.mean().detach())
            aux_out["arch_residual_memory"] = res.pow(2).mean()
        return local + res


def attach_instance_memory(model, dim: int = 1024, hidden: int = 64,
                           ls_max: Optional[float] = None,
                           res_cap: Optional[float] = None, verbose: bool = True) -> dict:
    """挂载（trainable），返回写进 arch.json 的配置。"""
    inner = getattr(model, "module", model)
    mod = InstanceMemory(dim=dim, hidden=hidden, ls_max=ls_max, res_cap=res_cap)
    ref = next(inner.projector.parameters())
    inner.instance_memory = mod.to(device=ref.device, dtype=ref.dtype)
    n = sum(p.numel() for p in mod.parameters())
    if verbose:
        print(f"[instance_memory] 挂载完成：hidden={hidden} → {n/1e6:.2f}M 参数"
              f"（门控零初始化，step0 与基线逐位一致）", flush=True)
    return {"instance_memory": True, "memory_hidden": int(hidden),
            "memory_params": int(n), "ls_max": ls_max, "res_cap": res_cap}


def maybe_attach_from_ckpt(model, ckpt_dir, verbose: bool = True):
    import json
    from pathlib import Path

    path = Path(ckpt_dir) / "arch.json"
    if not path.exists():
        return None
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not bool(cfg.get("instance_memory", False)):
        return None
    info = attach_instance_memory(
        model, dim=int(cfg.get("dim", 1024)),
        hidden=int(cfg.get("memory_hidden", 64)),
        ls_max=cfg.get("ls_max"), res_cap=cfg.get("res_cap"), verbose=verbose)
    return info
