"""解码期占用惩罚（OccupancyLogitBias，docs/21 方向 C）。

生成每个顶点时，对"已占用区域"的坐标 token 施加 logits 惩罚：
    score[<xN>] -= λ · occ_bias[y, N]    （x 阶段）
    score[<yN>] -= λ · occ_bias[N, x]    （y 阶段；此处 x 是已选定的横坐标）

硬性抑制新轮廓侵入已完成区域；只惩罚占用**内部**（保留一个边界带，
不切断联排楼的合法贴墙共界）。零训练改动、零参数，推理期即用。

与方向 A（文本条件）/B（画图）正交：A/B 告诉模型"邻居在哪"，C 在解码端
强制"别画进邻居"。可任意叠加。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from transformers import LogitsProcessor

from .geometry import GRID_SIZE


def occupancy_bias(occ: np.ndarray, grid: int = GRID_SIZE,
                   margin: int = 1) -> np.ndarray:
    """占用图（任意分辨率）→ 网格级惩罚表 `[grid,grid] ∈ [0,1]`。

    `margin`：占用边界带内的像素不惩罚（合法贴墙区）。
    """
    if occ.ndim != 2:
        raise ValueError(occ.shape)
    h, w = occ.shape
    if (h, w) != (grid, grid):
        import cv2
        occ = cv2.resize(occ.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
    if margin > 0:
        # 二值化（INTER_AREA 会产生中间灰度，直接腐蚀后残影仍是"半惩罚"）→ 腐蚀 → 只留内部
        import cv2
        occ = (occ > 0.5).astype(np.uint8)
        k = 2 * margin + 1
        # ⚠️ cv2.erode 默认 borderValue 是极大值（不侵蚀图像边界）——必须显式补 0，
        #    否则贴边的占用（联排楼常见）永远不产生边界带。
        eroded = cv2.erode(occ, np.ones((k, k), np.uint8),
                           borderType=cv2.BORDER_CONSTANT, borderValue=0)
        return np.clip(eroded.astype(np.float32), 0.0, 1.0)
    return np.clip(occ, 0.0, 1.0).astype(np.float32)


class OccupancyLogitBias(LogitsProcessor):
    """逐 token 的坐标 logits 惩罚。

    用法（generate 的 logits_processor 列表）：
        bias = OccupancyLogitBias(occ_bias_table, lam=1.0, tokenizer=tok)
        ... generate(..., logits_processor=[bias])

    需要知道当前生成的是 <x> 还是 <y>：通过已生成序列里的坐标 token 模式判断
    （成对 <x><y> —— 若上一个坐标 token 是 y 或序列刚开始，下一个是 x）。
    """

    def __init__(self, bias_table: np.ndarray, lam: float, tokenizer,
                 grid: int = GRID_SIZE):
        self.bias = torch.from_numpy(bias_table)          # [grid, grid]
        self.lam = float(lam)
        self.grid = grid
        self.x_ids = torch.tensor(
            [tokenizer.convert_tokens_to_ids(f"<x{i}>") for i in range(grid)])
        self.y_ids = torch.tensor(
            [tokenizer.convert_tokens_to_ids(f"<y{i}>") for i in range(grid)])
        self._id2xy = {}
        for i in range(grid):
            xi = tokenizer.convert_tokens_to_ids(f"<x{i}>")
            yi = tokenizer.convert_tokens_to_ids(f"<y{i}>")
            if isinstance(xi, int) and xi >= 0:
                self._id2xy[xi] = ("x", i)
            if isinstance(yi, int) and yi >= 0:
                self._id2xy[yi] = ("y", i)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # input_ids: [B, seq]；scores: [B, vocab]
        for b in range(input_ids.shape[0]):
            seq = input_ids[b].tolist()
            axis = self._next_axis(seq)
            if axis is None:
                continue
            if axis == "x":
                # 下一个是 <xN>：惩罚 occ[:, N] 的所有行 → 对每个 N 惩罚列占用
                for n in range(self.grid):
                    tid = int(self.x_ids[n])
                    if tid >= 0:
                        scores[b, tid] -= self.lam * float(self.bias[:, n].max())
            else:
                # 上一个 <xN> 已选：惩罚 occ[N, :] → 行占用按 N
                x_n = self._last_x(seq)
                if x_n is not None:
                    for n in range(self.grid):
                        tid = int(self.y_ids[n])
                        if tid >= 0:
                            scores[b, tid] -= self.lam * float(self.bias[n, x_n])
        return scores

    def _last_x(self, seq):
        for tid in reversed(seq):
            info = self._id2xy.get(tid)
            if info and info[0] == "x":
                return info[1]
            if info and info[0] == "y":
                return None      # x 之后紧跟的是 y —— 没有悬空 x
        return None

    def _next_axis(self, seq):
        """成对模式：遇 y 后下一个是 x；遇 x 后下一个是 y；序列无坐标 → x。"""
        for tid in reversed(seq):
            info = self._id2xy.get(tid)
            if info is None:
                continue          # 普通文本 token，继续往前找最近的坐标 token
            return "y" if info[0] == "x" else "x"
        return "x"
