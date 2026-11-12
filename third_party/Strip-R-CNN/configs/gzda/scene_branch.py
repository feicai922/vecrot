"""场景分类分支（重建版）：在检测器上挂一个「本窗口有没有建筑」的二分类头。

## 为什么需要重建

`vecrot/models/detector_strip_s_scene/strip_s_gzda_scene.py` 的
``custom_imports`` 引用了 ``configs.gzda.scene_branch`` / ``scene_hooks`` /
``continue_hooks`` 三个模块，但上游（YXB-NKU/Strip-R-CNN）与本地
``configs/gzda`` 里都**没有** ``scene_branch.py``。

原始实现拿不到，但权重把结构完整暴露了。从
``best_mAP_iter_37620.pth`` 抽出的 ``scene_head`` 只有 6 个张量：

    scene_head.fc1.weight  (512, 1280)     ← 5 层 × 256 通道，全局池化后拼接
    scene_head.fc1.bias    (512,)
    scene_head.norm.weight (512,)          ← LayerNorm
    scene_head.norm.bias   (512,)
    scene_head.fc2.weight  (2, 512)        ← 二分类：无建筑 / 有建筑
    scene_head.fc2.bias    (2,)

这与配置里的 ``scene_head=dict(in_channels=256, num_levels=5,
hidden_channels=512, dropout=0.1, ...)`` 完全对应。因此可以按这个形状精确重建，
加载权重后 ``missing/unexpected`` 应为 0。

## 语义

* 输入：``extract_feat()`` 的 5 层 FPN 特征
* 每层做全局平均池化 → 5 个 256 维向量拼接 → 1280
* ``fc1`` → LayerNorm → GELU → Dropout → ``fc2`` → 2 logits
* 类别 0 = 无建筑，类别 1 = 有建筑（按损失权重与 ``min_pos_instances`` 的惯例）

## 用途

大范围推理时先判场景，无建筑的窗口直接跳过，**不跑 RPN/RoI 与轮廓生成**。
这一层判定比跑完整检测便宜得多（只有 5 次 GAP + 两层全连接）。

## 与训练口径的关系

原始 ``scene_branch.py`` 不在仓库，本实现是**按权重形状重建**的等价结构。
数值等价性未与原训练脚本比对过，因此：

* 结构上权重能 100% 加载（可验证）
* 分类判决的阈值需要用户按自己的数据标定（默认 0.5，可在 manifest 里改）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrotate.models.builder import ROTATED_DETECTORS
from mmrotate.models.detectors.strip_rcnn import StripRCNN


class SceneHead(nn.Module):
    """5 层特征 → 二分类 logits。"""

    def __init__(self, in_channels=256, num_levels=5, hidden_channels=512,
                 dropout=0.1):
        super().__init__()
        self.num_levels = int(num_levels)
        self.in_channels = int(in_channels)
        flat = self.num_levels * self.in_channels
        self.fc1 = nn.Linear(flat, int(hidden_channels))
        self.norm = nn.LayerNorm(int(hidden_channels))
        self.drop = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(hidden_channels), 2)

    def forward(self, feats):
        """feats: list[Tensor]，每个 (N, C, H, W)。返回 (N, 2) logits。"""
        pooled = []
        for f in feats[: self.num_levels]:
            # 全局平均池化：每层压成一个 C 维向量
            pooled.append(F.adaptive_avg_pool2d(f, 1).flatten(1))
        x = torch.cat(pooled, dim=1)
        x = self.fc1(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.drop(x)
        return self.fc2(x)

    def predict_proba(self, feats):
        """返回「有建筑」的概率 (N,)。softmax 后取类别 1。"""
        logits = self.forward(feats)
        return logits.softmax(dim=-1)[:, 1]


@ROTATED_DETECTORS.register_module()
class StripRCNNScene(StripRCNN):
    """Strip-R-CNN + 场景二分类头。

    只在推理期用 ``scene_head`` 做跳过判定；训练相关内容（loss）不实现，
    因为本仓库只做部署推理（开发文档 §1.1）。
    """

    def __init__(self, scene_head=None, **kwargs):
        super().__init__(**kwargs)
        cfg = scene_head or {}
        self.with_scene = cfg is not None and bool(cfg)
        if self.with_scene:
            self.scene_head = SceneHead(
                in_channels=cfg.get("in_channels", 256),
                num_levels=cfg.get("num_levels", 5),
                hidden_channels=cfg.get("hidden_channels", 512),
                dropout=cfg.get("dropout", 0.1),
            )
            #: 判为「有建筑」的概率阈值。推理时可由调用方覆盖。
            self.scene_threshold = float(cfg.get("score_threshold", 0.5))

    # ---------------------------------------------------------- 推理

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        """带场景跳过的 simple_test。

        与父类不同点：先算场景概率；低于阈值时**直接返回空结果**，
        不跑 RPN / RoI / NMS，省掉绝大部分算力。
        """
        if not self.with_scene:
            return super().simple_test(img, img_metas, rescale=rescale, **kwargs)

        feats = self.extract_feat(img)
        prob = float(self.scene_head.predict_proba(feats)[0].item())
        # 记录到最后一个 img_meta，便于上层读取（不改变返回值签名）
        try:
            img_metas[0]["scene_prob"] = prob
            img_metas[0]["scene_has_building"] = prob >= self.scene_threshold
        except (IndexError, TypeError, KeyError):
            pass

        if prob < self.scene_threshold:
            # 判为无建筑：返回与「检出 0 个框」完全一致的形状，
            # 让下游无需特判。num_classes 个空数组。
            n_cls = self.roi_head.bbox_head.num_classes
            empty = [img.new_zeros((0, 6)) for _ in range(n_cls)]
            return [empty]

        return super().simple_test(img, img_metas, rescale=rescale, **kwargs)
