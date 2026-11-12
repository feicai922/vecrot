"""BDR v2：**稠密中心点 + 尺寸**头（替代 v1 的 3 槽回归）。

v1 的教训（实测）：3 槽回归的子框 IoU 只有 0.54，而下游矢量模型需要子框 IoU ≥0.72 才吃得下
（检测框 IoU 0.811 时整体 p@0.75 才 0.447；子框 0.54 时端到端几乎没涨）。所以改成
标准稠密检测范式：在 224² 裁剪图上预测**每栋楼的中心热力图 + 尺寸 + 偏移**，峰值数即 k。

输出（stride=4, 56×56）：
  hm   [B,1,56,56]  中心热力图（focal loss，Gaussian 目标）
  size [B,2,56,56]  相对裁剪边长的 (w,h)
  off  [B,2,56,56]  亚像素偏移
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BDRNet2(nn.Module):
    """`angle=True` 时额外输出双角 `(sin2θ, cos2θ)`（docs/06 §6 的最小版本要求）。

    双角只消除 **π 周期**歧义，**不**解决近正方形的长短边交换 / 角度不稳定（见 §6 与 `05` §三）。
    """

    def __init__(self, width: int = 32, angle: bool = False):
        super().__init__()
        self.use_angle = bool(angle)
        c = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, 3, 2, 1), nn.BatchNorm2d(c), nn.ReLU(True),        # 112
            nn.Conv2d(c, c * 2, 3, 2, 1), nn.BatchNorm2d(c * 2), nn.ReLU(True),  # 56
        )
        self.body = nn.Sequential(
            *[nn.Sequential(nn.Conv2d(c * 2, c * 2, 3, 1, 1), nn.BatchNorm2d(c * 2), nn.ReLU(True))
              for _ in range(4)]
        )
        self.hm = nn.Conv2d(c * 2, 1, 1)
        self.size = nn.Conv2d(c * 2, 2, 1)
        self.off = nn.Conv2d(c * 2, 2, 1)
        nn.init.constant_(self.hm.bias, -4.6)      # 初始概率 ≈0.01
        if self.use_angle:
            self.ang = nn.Conv2d(c * 2, 2, 1)       # (sin2θ, cos2θ)
            nn.init.zeros_(self.ang.weight)
            nn.init.constant_(self.ang.bias, 0.0)   # 初始 θ=0（sin0=0, cos0=1 → 下方做归一化）

    def forward(self, x):
        f = self.body(self.stem(x))
        if not self.use_angle:
            return self.hm(f), self.size(f), self.off(f)
        a = self.ang(f)
        return self.hm(f), self.size(f), self.off(f), a


def focal_loss(pred_logit, gt_hm, alpha=2.0, beta=4.0):
    p = torch.sigmoid(pred_logit).clamp(1e-4, 1 - 1e-4)
    pos = gt_hm >= 0.999          # 注意：高斯峰实测为 0.99975，用 1-1e-4 会取不到任何正样本（踩过）
    neg = ~pos
    pos_loss = -((1 - p) ** alpha) * torch.log(p) * pos.float()
    neg_loss = -((1 - gt_hm) ** beta) * (p ** alpha) * torch.log(1 - p) * neg.float()
    n = pos.float().sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n


def gaussian_target(hm, cx, cy, sigma, radius=None):
    """在 hm 上打高斯峰（就地）。cx,cy 为 heatmap 坐标。"""
    H, W = hm.shape[-2], hm.shape[-1]
    ys = torch.arange(H, device=hm.device)[:, None].float()
    xs = torch.arange(W, device=hm.device)[None, :].float()
    g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma * sigma))
    hm.copy_(torch.maximum(hm, g))
