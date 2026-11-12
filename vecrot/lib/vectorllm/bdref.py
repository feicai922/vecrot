"""框精化头（BDRef）：**从水平检测框回归出一个更紧、带角度的目标框**。

与 `bdr2.py`（一框拆多栋）的区别：这里**只做"一个框 → 一个更准的有向框"**，
输出维度固定为 6，不预测 k、不做候选分解。用户的目标是"用旋转框把裁剪收紧到只装一栋楼"，
所以这里直接优化"框"本身：

    检测框 (x1,y1,x2,y2) + 其周围 224² 裁剪
        └─► BoxRefineNet ─► (dx, dy, log sw, log sh, sin2θ, cos2θ)
    精化框 cx = cx0 + dx·W0, cy = cy0 + dy·H0, w = W0·e^{sw}, h = H0·e^{sh}, θ = ½·atan2(sin2θ,cos2θ)

约束（都做了有界化，避免把框推飞）：
* `dx,dy ∈ (-0.5, 0.5)`（tanh × 0.5）——中心只能挪半个框；
* `sw,sh ∈ (-log2, +log2)`（tanh × log2）——宽高最多减半/加倍；
* 角度用双角编码，θ 的规范与 `rbox.min_area_rect` 一致（`w` 为长边、θ∈(-π/2,π/2]）。
  双角只消除 π 周期歧义，**不解决近正方形目标的长短边交换**；训练时对近方形目标降低角度权重。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

S = 224


class BoxRefineNet(nn.Module):
    def __init__(self, width: int = 32, use_angle: bool = True, box_feat: int = 8):
        super().__init__()
        c = int(width)
        self.use_angle = bool(use_angle)
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, 3, 2, 1), nn.BatchNorm2d(c), nn.ReLU(True),          # 112
            nn.Conv2d(c, c * 2, 3, 2, 1), nn.BatchNorm2d(c * 2), nn.ReLU(True),  # 56
        )
        self.body = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(c * 2, c * 2, 3, 1, 1), nn.BatchNorm2d(c * 2), nn.ReLU(True))
              for _ in range(3)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.box_mlp = nn.Sequential(nn.Linear(box_feat, c), nn.ReLU(True))
        n_out = 6 if self.use_angle else 4
        self.head = nn.Sequential(nn.Linear(c * 2 + c, 128), nn.ReLU(True), nn.Linear(128, n_out))
        nn.init.zeros_(self.head[-1].weight)      # 零初始化 → 起点就是"不改框"
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, img: torch.Tensor, box_feat: torch.Tensor):
        ang_bias = torch.tensor([0.0, 1.0], device=img.device, dtype=img.dtype) * (1.0 if self.use_angle else 0.0)
        f = self.pool(self.body(self.stem(img))).flatten(1)
        g = self.box_mlp(box_feat)
        raw = self.head(torch.cat((f, g), -1))
        # 有界化
        dx = 0.5 * torch.tanh(raw[:, 0])
        dy = 0.5 * torch.tanh(raw[:, 1])
        sw = math.log(2.0) * torch.tanh(raw[:, 2])
        sh = math.log(2.0) * torch.tanh(raw[:, 3])
        if not self.use_angle:
            return torch.stack((dx, dy, sw, sh), -1)
        # ⚠️ 修 NaN（2026-11-06）：head 零初始化时 raw=0，angle=(0,0) 会让 atan2(0,0) 的梯度变 NaN。
        #    加 (0,1) 偏置：初始化即 θ=0（"不改框"），且梯度有限。
        ang = F.normalize(raw[:, 4:6] + ang_bias, dim=-1, eps=1e-6)
        return torch.stack((dx, dy, sw, sh), -1), ang


# --------------------------------------------------------------------------- 几何


def apply_refine(det_boxes: torch.Tensor, pred: torch.Tensor, ang: Optional[torch.Tensor] = None
                 ) -> torch.Tensor:
    """`det_boxes[B,4]=(x1,y1,x2,y2)` + 6 维预测 → 有向框 `[B,5]=(cx,cy,w,h,θ)`。

    角度 = ½·atan2(sin2θ, cos2θ) 与 `min_area_rect` 的规范一致（长边为 w、θ∈(-π/2,π/2]）。
    """
    x1, y1, x2, y2 = det_boxes[:, 0], det_boxes[:, 1], det_boxes[:, 2], det_boxes[:, 3]
    cx0, cy0 = (x1 + x2) / 2, (y1 + y2) / 2
    W0 = (x2 - x1).clamp(min=1.0)
    H0 = (y2 - y1).clamp(min=1.0)
    cx = cx0 + pred[:, 0] * W0
    cy = cy0 + pred[:, 1] * H0
    w = W0 * torch.exp(pred[:, 2])
    h = H0 * torch.exp(pred[:, 3])
    if ang is None:
        th = torch.zeros_like(w)
    else:
        th = 0.5 * torch.atan2(ang[:, 0], ang[:, 1])
    # 统一规范：w 为长边、θ 归到 (-π/2, π/2]
    swap = w < h
    w2 = torch.where(swap, h, w)
    h2 = torch.where(swap, w, h)
    th = torch.where(swap, th + math.pi / 2, th)
    th = torch.remainder(th + math.pi / 2, math.pi) - math.pi / 2
    return torch.stack((cx, cy, w2, h2, th), -1)


def target_from_obb(det_boxes: torch.Tensor, obb: torch.Tensor) -> torch.Tensor:
    """GT OBB `[B,5]` 相对检测框 → 训练目标 `[B,6]=(dx,dy,log sw,log sh,sin2θ,cos2θ)`。"""
    x1, y1, x2, y2 = det_boxes[:, 0], det_boxes[:, 1], det_boxes[:, 2], det_boxes[:, 3]
    cx0, cy0 = (x1 + x2) / 2, (y1 + y2) / 2
    W0 = (x2 - x1).clamp(min=1.0)
    H0 = (y2 - y1).clamp(min=1.0)
    dx = ((obb[:, 0] - cx0) / W0).clamp(-0.5, 0.5)
    dy = ((obb[:, 1] - cy0) / H0).clamp(-0.5, 0.5)
    sw = torch.log((obb[:, 2] / W0).clamp(1e-3, 1e3)).clamp(-math.log(2), math.log(2))
    sh = torch.log((obb[:, 3] / H0).clamp(1e-3, 1e3)).clamp(-math.log(2), math.log(2))
    th = obb[:, 4]
    return torch.stack((dx, dy, sw, sh, torch.sin(2 * th), torch.cos(2 * th)), -1)


def box_corners(det_boxes: torch.Tensor, pred: torch.Tensor, ang: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
    """可微地把预测参数变成 4 个角点 `[B,4,2]`（画多边形用的顺序：左上→右上→右下→左下）。

    用途：训练时直接对**角点**做 L1，比对着归一化参数回归更贴近"框准不准"。
    """
    obb = apply_refine(det_boxes, pred, ang)                      # [B,5]
    cx, cy, w, h, th = obb[:, 0], obb[:, 1], obb[:, 2], obb[:, 3], obb[:, 4]
    c, s_ = torch.cos(th), torch.sin(th)
    dx = torch.stack((-w / 2, w / 2, w / 2, -w / 2), -1)           # [B,4]
    dy = torch.stack((-h / 2, -h / 2, h / 2, h / 2), -1)
    x = cx[:, None] + c[:, None] * dx - s_[:, None] * dy
    y = cy[:, None] + s_[:, None] * dx + c[:, None] * dy
    return torch.stack((x, y), -1)


def gt_corners(obb: torch.Tensor) -> torch.Tensor:
    """真值 OBB `[B,5]` → 4 角点 `[B,4,2]`（与 `box_corners` 同序）。"""
    cx, cy, w, h, th = obb[:, 0], obb[:, 1], obb[:, 2], obb[:, 3], obb[:, 4]
    c, s_ = torch.cos(th), torch.sin(th)
    dx = torch.stack((-w / 2, w / 2, w / 2, -w / 2), -1)
    dy = torch.stack((-h / 2, -h / 2, h / 2, h / 2), -1)
    x = cx[:, None] + c[:, None] * dx - s_[:, None] * dy
    y = cy[:, None] + s_[:, None] * dx + c[:, None] * dy
    return torch.stack((x, y), -1)
