"""BDR（Box Decompose & Refine）：把一个检测框分解成 k 个子框，供下游矢量模型逐个出多边形。

为什么长这样：
* 我们的实测（gzda_v0 val，1,945 栋）里 **20.8% 的目标挤在同一个检测框内**（261 个 2 楼框、
  97 个 3 楼框…），而这些目标的 p@0.75 只有 **0.3259**，单楼框内是 0.4473 —— 因为流水线是
  "1 个框 → 1 个多边形"，物理上画不出两栋楼。
* 所以模块不该再往视觉 token 上贴东西（那一路已实测全是噪声），而应该**改"框→实例"的基数**。
* 本模块**完全不碰主干与 LLM**：只预测 k 与 k 个子框，然后让现有模型跑 k 次。
  因此它不可能把生成训坏（这是本项目里唯一被证明安全的改动方式：不改权重）。

输入：以检测框为中心、外扩 `expand` 的裁剪图（224×224）+ 框的几何量
输出：k∈{1,2,3} 的概率、3 个子框（相对输入框归一化的 cx,cy,w,h）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BDRNet(nn.Module):
    MAX_K = 3

    def __init__(self, width: int = 32, img_size: int = 224):
        super().__init__()
        c = width
        self.stem = nn.Sequential(
            nn.Conv2d(3, c, 3, 2, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, 3, 2, 1), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, 3, 2, 1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 8, 3, 2, 1), nn.BatchNorm2d(c * 8), nn.ReLU(inplace=True),
            nn.Conv2d(c * 8, c * 8, 3, 2, 1), nn.BatchNorm2d(c * 8), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat = c * 8 + 3
        self.trunk = nn.Sequential(nn.Linear(feat, 256), nn.ReLU(inplace=True))
        self.head_k = nn.Linear(256, self.MAX_K)
        self.head_box = nn.Linear(256, self.MAX_K * 4)

    def forward(self, img, geo):
        x = self.pool(self.stem(img)).flatten(1)
        x = self.trunk(torch.cat([x, geo], dim=1))
        k_logits = self.head_k(x)
        boxes = self.head_box(x).view(-1, self.MAX_K, 4)
        return k_logits, boxes


def bdr_loss(k_logits, boxes, k_gt, sub_gt, geo_weight: float = 1.0):
    """CE(k) + 掩码 L1(子框)。子框已按位置排序，无需 Hungarian 匹配。"""
    loss_k = F.cross_entropy(k_logits, k_gt - 1)
    bsz = boxes.shape[0]
    mask = (torch.arange(BDRNet.MAX_K, device=boxes.device)[None, :] < k_gt[:, None]).float()
    l1 = (boxes - sub_gt).abs().sum(dim=-1)
    loss_b = (l1 * mask).sum() / mask.sum().clamp(min=1.0)
    return loss_k + geo_weight * loss_b, loss_k.detach(), loss_b.detach()
