"""SFT 数据加载：把 `tools/build_abstain_dataset.py` 造的 jsonl 变成训练张量。

一条 jsonl 记录长这样::

    {"image": ".../train_positive_0000123.png", "tile": "...", "crop_bbox": [...],
     "expand_ratio": 1.43, "prompt": "<|im_start|>user\\n<pixel>\\nPlease ...<|im_start|>assistant\\n",
     "target": "<x21><y21><x106><y21>..."  或  "<no_object>",
     "label": "building" | "no_object", "kind": "positive|pair|empty_tile|background"}

关键点（踩过的坑都写在注释里）：

1. `<pixel>` 必须展开成 **64 个占位 token**（= 8×8 视觉 token），与推理端 processor 行为一致；
2. **prompt 部分的 label 全部置 -100**，只在目标序列上算 loss，否则模型会去学"预测提问"；
3. 目标序列末尾要补 `<|im_end|>`，让模型学会**在哪里停**（推理时就是靠它停的）；
4. `<no_object>` 是**单个 token**（训练前已 `add_tokens`），不能被切开；
5. 目标量化网格是 0~127（格心映射），与推理端 `vectorllm.geometry` 完全一致。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .geometry import GRID_SIZE
from .abstain import NO_OBJECT_TOKEN  # noqa: F401
from .prompts import PIXEL_TOKEN, STOP_WORDS, raw_prompt
from .vocab_ext import format_target as _fmt_target

IGNORE_INDEX = -100
NUM_VISUAL_TOKENS = (GRID_SIZE // 16) ** 2  # 128/16 = 8 → 64 个视觉 token
_end_token = STOP_WORDS[0]  # "<|im_end|>"


def expand_pixel_placeholder(text: str, num_tokens: int = NUM_VISUAL_TOKENS) -> str:
    """把第一个 `<pixel>` 替换成 `num_tokens` 个占位 token（与 processor 的行为一致）。"""
    placeholder = "<|reserved_pixel_placeholder|>"
    head = text.replace(PIXEL_TOKEN, placeholder * num_tokens, 1)
    return head.replace(placeholder, PIXEL_TOKEN)


class AbstainSFTDataset(Dataset):
    """读 jsonl → (pixel_values, input_ids, attention_mask, labels)。

    Args:
        jsonl: 数据文件（`tools/build_abstain_dataset.py` 的产物）。
        image_processor: `VectorLLMImageProcessor` 实例（负责 resize 到 128、/255、不归一化）。
        tokenizer: **已经 add_tokens(["<no_object>"]) 的** tokenizer。
        include_no_object: 关掉可以把负样本过滤掉（做"只训框抖动"的消融时用）。
        keep_kinds: 只保留这几类样本（默认全部）。
        max_length: 超长样本截断（正常不会触发：64+~40 token）。
    """

    def __init__(
        self,
        jsonl: str | Path,
        image_processor,
        tokenizer,
        include_no_object: bool = True,
        keep_kinds: Optional[Sequence[str]] = None,
        max_length: int = 1024,
        negative_repeat: int = 1,
        positive_repeat: int = 1,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: List[Dict] = []
        keep = set(keep_kinds) if keep_kinds else None
        self.negative_repeat = max(1, int(negative_repeat))
        self.positive_repeat = max(1, int(positive_repeat))
        for line in open(jsonl, encoding="utf-8"):
            record = json.loads(line)
            if keep is not None and record.get("kind") not in keep:
                continue
            if not include_no_object and record.get("label") == "no_object":
                continue
            # 负样本过采样：直接提高负样本在 token 层面的占比（比只调 loss 权重更硬）
            repeat = (self.negative_repeat if record.get("label") == "no_object"
                      else self.positive_repeat)
            self.records.extend([record] * repeat)
        if not self.records:
            raise ValueError(f"no usable records in {jsonl}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        image = Image.open(record["image"]).convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"][0]

        prompt = expand_pixel_placeholder(record["prompt"])
        target = record["target"] + _end_token

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(target_ids)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        out = {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "n_target_tokens": torch.tensor(len(target_ids), dtype=torch.long),
        }
        if self.use_hr_edge:
            from .global_fusion import edge_label_from_polygon, hr_tensor_from_crop

            out["hr_pixel_values"] = hr_tensor_from_crop(crop, self.hr_size)
            # 负样本（no_object）没有多边形 → 全零边界图，正好教它"这里没有边界"
            out["edge_labels"] = edge_label_from_polygon(
                local_polygon or [], crop.size, self.edge_size, self.edge_width)
        return out


def jitter_box(box, image_size, shift: float = 0.0, aspect: float = 0.0, rng=None):
    """裁剪框再抖一次：**中心平移 + 长宽比**（部署用的是检测框，检测框相对 GT 有偏移/松紧差异）。

    Args:
        shift: 中心平移幅度 = shift × 框半边长（如 0.08 = 每边最多 8%）
        aspect: x/y 各自独立乘 `1±aspect`（如 0.10 = ±10%），模拟检测框长宽比不一致
    """
    import random as _random

    if (shift <= 0 and aspect <= 0) or rng is None:
        return box
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if shift > 0:
        cx += rng.uniform(-shift, shift) * w / 2.0
        cy += rng.uniform(-shift, shift) * h / 2.0
    if aspect > 0:
        w *= 1.0 + rng.uniform(-aspect, aspect)
        h *= 1.0 + rng.uniform(-aspect, aspect)
    nx1 = max(0.0, cx - w / 2.0); ny1 = max(0.0, cy - h / 2.0)
    nx2 = min(float(image_size[0]), cx + w / 2.0); ny2 = min(float(image_size[1]), cy + h / 2.0)
    if nx2 - nx1 < 8 or ny2 - ny1 < 8:
        return box
    return (int(round(nx1)), int(round(ny1)), int(round(nx2)), int(round(ny2)))


def occlude_corners(crop, polygon, origin, tile_image, rng, corner_frac: float = 0.35,
                    max_corners: int = 4):
    """**四角遮挡增强**：在目标建筑外接框的角上贴不规则"树冠"斑块，GT 保持不变。

    为什么专打四角：遮挡最容易吃掉的是角点，而角点决定拓扑（缺一个角就会被画成缺角 L 形）。
    只遮角、不动 GT，等于直接教模型"这个角看不见，但要按矩形关系补出来"。

    抽成**模块级函数**是为了两个数据集共用同一实现：`CropCocoDataset`（单域 WHU）和
    `MultiCocoDataset`（多域混采）—— 之前只有前者接了遮挡增强，多域那条线根本拿不到
    这个增益（2026-10-31 补齐）。随机性由调用方传入的 `rng` 决定，保证多 worker 可复现。

    Args:
        crop: 已裁好的 PIL 图（局部坐标原点在裁剪框左上角）。
        polygon: 该栋楼在**瓦片**坐标下的 GT 多边形。
        origin: 裁剪框左上角在瓦片里的坐标 (x1, y1)。
        tile_image: 整张瓦片（用来取"树冠"纹理）。
        rng: `random.Random` 实例。
        corner_frac: 每个角斑块半径 = 该比例 × 目标框短边。
        max_corners: 最多同时遮几个角。
    """
    ox, oy = origin
    local = [(x - ox, y - oy) for x, y in polygon]
    xs = [p[0] for p in local]; ys = [p[1] for p in local]
    bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
    bw, bh = bx2 - bx1, by2 - by1
    if bw < 12 or bh < 12:
        return crop

    # 纹理：从瓦片里随机取一块，并**提高绿色通道**模拟树冠
    tw = th = 64
    tx = rng.randint(0, max(1, tile_image.size[0] - tw))
    ty = rng.randint(0, max(1, tile_image.size[1] - th))
    import numpy as _np
    from PIL import Image as _Image
    arr = _np.asarray(tile_image.crop((tx, ty, tx + tw, ty + th)).convert("RGB")).astype(_np.float32)
    arr[..., 1] = _np.clip(arr[..., 1] * 1.18 + 12, 0, 255)
    texture = _Image.fromarray(arr.astype(_np.uint8)).resize(crop.size, _Image.BILINEAR)

    from PIL import ImageDraw as _ImageDraw
    mask = _Image.new("L", crop.size, 0)
    draw = _ImageDraw.Draw(mask)
    corners = [(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)]
    rng.shuffle(corners)
    k = rng.randint(1, max(1, max_corners))
    r0 = corner_frac * min(bw, bh)
    for cx, cy in corners[:k]:
        for _ in range(rng.randint(2, 4)):     # 每个角 2~4 个椭圆叠成不规则团块
            rx = r0 * rng.uniform(0.6, 1.15)
            ry = r0 * rng.uniform(0.6, 1.15)
            dx = rng.uniform(-0.35, 0.35) * r0
            dy = rng.uniform(-0.35, 0.35) * r0
            draw.ellipse([cx - rx + dx, cy - ry + dy, cx + rx + dx, cy + ry + dy], fill=255)
    out = crop.copy()
    out.paste(texture, (0, 0), mask)
    return out


def expand_dense_instances(instances, alpha: float = 1.0, beta: float = 0.3,
                           cap: float = 4.0, seed: int = 0):
    """**G5 粘连难例采样**：按"与最近邻建筑的最小间距 / 自身短边"给样本加权，做整数复制。

    动机（遥感实例分割 E 类方法的共识 + 我们的实测）：密集联排/城中村里两栋楼在像素上只隔
    几像素甚至共墙，是"一个 prompt 画出两个楼"的主因；而目标域数据里这类样本占比很低
    （gzda_v0 train 里 `d_nn < 0.3×短边` 只占少数），均匀采样时模型见得太少。

    权重 `w = exp(-d_nn / (beta * short_side))`（贴邻 → w≈1，孤立的楼 → w≈0），
    复制次数 `1 + round(min(alpha * w, cap))`。**纯数据侧、零参数**，不改网络结构。

    为什么用"复制"而不是 `WeightedRandomSampler`：复制对 DataLoader 的确定性/epoch 语义
    完全透明（`__len__` 仍然等于真实样本数的加权和），不会引入跨 worker 的采样错位。
    """
    if alpha <= 0:
        return instances
    by_tile = {}
    for it in instances:
        by_tile.setdefault(it["file_name"], []).append(it)

    def _gap(a, b):
        ax1, ay1, ax2, ay2 = a["bbox"][:4]
        bx1, by1, bx2, by2 = b["bbox"][:4]
        dx = max(0.0, max(ax1, bx1) - min(ax2, bx2))
        dy = max(0.0, max(ay1, by1) - min(ay2, by2))
        return (dx * dx + dy * dy) ** 0.5

    out, ws = [], []
    for it in instances:
        short = max(1.0, min(it["bbox"][2] - it["bbox"][0], it["bbox"][3] - it["bbox"][1]))
        mates = by_tile[it["file_name"]]
        d_nn = min((_gap(it, m) for m in mates if m is not it), default=1e9)
        import math as _m
        w = _m.exp(-d_nn / (beta * short)) if d_nn < 1e8 else 0.0
        ws.append(w)
        out.append(it)
    import statistics as _st
    wbar = _st.mean(ws) if ws else 0.0
    idx = []
    for i, it in enumerate(out):
        w = ws[i] / wbar if wbar > 1e-9 else 0.0
        k = 1 + int(round(min(alpha * w, cap)))
        idx.extend([i] * max(1, k))
    import random as _random
    _random.Random(seed).shuffle(idx)
    res = [out[i] for i in idx]
    print(f"[data] 粘连难例加权采样 alpha={alpha} beta={beta}: {len(instances)} → {len(res)} 条"
          f"（最近邻间距中位 {_st.median([min((_gap(it, m) for m in by_tile[it['file_name']] if m is not it), default=0.0) for it in instances[:3000]]):.1f}px，"
          f"权重均值 {wbar:.3f}）", flush=True)
    return res


def _aug_rng(seed: int, worker: int, call: int):
    """按 `(seed, worker_id, 调用计数)` 派生 RNG。

    不能共用一份跨 worker 的 `self.rng`：DataLoader 的 worker 是 fork 出来的，各 worker
    会拿到完全相同的 RNG 状态，于是产生同一串增强（外部审阅指出，已核实）。
    `random.Random` 只接受 int/float/str/bytes —— 不能直接喂 tuple（踩过）。
    """
    import random as _random

    s = ((int(seed) * 1_000_003) ^ (int(worker) * 79_219) ^ (int(call) * 15_485_863))
    return _random.Random(s)


class CropCocoDataset(Dataset):
    """**在线裁剪**的 COCO 数据集：每张瓦片按 GT 框裁出一栋楼的样本。

    专为 WHU-Mix-Vector 这类"瓦片级 + 每图十几栋"的数据设计：
    不做离线裁剪（那样要落 50 万张 PNG），而是在 `__getitem__` 里读瓦片、按框裁剪、
    缩到 128 网格算目标序列。配合 DataLoader 多 worker 即可。

    Args:
        root: WHU-Mix 根目录（含 `{split}/{split}.json` 与 `{split}/image/`）。
        split: train / val / test-1 / test-2
        jitter: (lo, hi) 裁剪框外扩比随机区间；训练建议 1.0~1.6，验证用固定值。
        max_instances: 只取 N 栋。**默认先按固定种子随机打乱再取前 N**
            （`ordered_subset=True` 可恢复旧行为：按 COCO 标注顺序取前 N）。
            旧行为会让"数据量消融"混入场景组成差异 —— COCO 标注顺序与采集区块相关，
            取前 N 等价于"只训前几个场景"（外部审阅 `codex建议.md` 指出，已核实）。

    两个与多 worker 有关的小设计（同样来自那次审阅）：
        * 裁剪抖动的随机数**不再共用一份跨 worker 的 RNG**（fork 后各 worker 状态相同，
          会让不同 worker 产生同一串增强），改为按 `(seed, worker_id, 调用计数)` 派生；
        * 瓦片读不到时仍会换邻近样本兜底，但**会计数并打印**，避免"缓存缺文件"静默改变数据分布。
    """

    def __init__(self, root, split: str, image_processor, tokenizer, jitter=(1.0, 1.6),
                 max_instances: int | None = None, seed: int = 0, fixed_ratio: float | None = None,
                 max_length: int = 1024, ordered_subset: bool = False,
                 occ_aug_prob: float = 0.0, occ_corner_frac: float = 0.35,
                 occ_max_corners: int = 4, box_shift: float = 0.0, box_aspect: float = 0.0,
                 vertex_count_token: bool = False, use_global: bool = False,
                 use_hr_edge: bool = False, hr_size: int = 256, edge_size: int = 64,
                 edge_width: int = 2, use_scale_cond: bool = False, use_zoom: bool = False,
                 zoom_ratio: float | None = None, use_corner: bool = False,
                 corner_sigma: float = 1.2, code_dim: int = 64,
                 shadow_aug_prob: float = 0.0, shadow_aug_strength: float = 0.35,
                 shadow_aug_blob: bool = False, dense_alpha: float = 0.0,
                 dense_beta: float = 0.3, dense_cap: float = 4.0,
                 crop_mode: str = "aabb", mask_expand: float = 1.02, pos_angle: bool = False,
                 det_glob_npz: str | None = None,
                 use_ctx: bool = False, ctx_tile_size: int = 384,
                 ctx_neighbors: dict | None = None, ctx_k: int = 6,
                 tile_whitelist=None, use_memory: bool = False, mem_frac: float = 0.7,
                 use_neighbor_prompt: bool = False, neighbor_k: int = 3,
                 neighbor_side: int = 8, use_neighbor_draw: bool = False,
                 draw_alpha: float = 0.25, use_density_code: bool = False):
        import random

        import numpy as np

        # 目标序列是否以 `<nvK>` 打头（顶点数显式监督，见 src/vectorllm/vocab_ext.py）
        self.vertex_count_token = bool(vertex_count_token)

        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.jpeg = str(root)
        self.jitter = jitter
        self.fixed_ratio = fixed_ratio
        self.seed = seed
        self.occ_aug_prob = float(occ_aug_prob)
        self.occ_corner_frac = float(occ_corner_frac)   # 每个角被遮挡的半径 = 该比例 × bbox 短边
        self.occ_max_corners = int(occ_max_corners)
        self.box_shift = float(box_shift)      # 中心平移抖动（占框半边长比例）
        self.box_aspect = float(box_aspect)    # 长宽比抖动（x/y 各自 ±）
        self.use_global = bool(use_global)     # 是否额外喂"整幅瓦片"做全局上下文融合
        self.pos_angle = bool(pos_angle)       # OBB 条件位置编码（96 维）
        # 高分辨率边界分支：额外喂原始分辨率 crop + 栅格化的边界监督标签（仅训练期）
        self.use_hr_edge = bool(use_hr_edge)
        self.hr_size = int(hr_size)
        self.edge_size = int(edge_size)
        self.edge_width = int(edge_width)
        # 尺度条件 / 放大分支 / 角点热图（scale_edge.py，2026-11-03）
        self.crop_mode = str(crop_mode)      # aabb | mask(填黑) | rbox(转正+填黑) | rot(只转正)
        self.pos_angle = bool(pos_angle)     # OBB 条件位置编码（转正裁剪时把 θ 也编进去）
        # 检测器多层 FPN token（global_fusion.DetGlobalAdapter 的输入）；按瓦片名索引
        # ⚠️ 2026-11-08 修：以前用 npz（压缩）建 dict → 每个 DataLoader worker 各自解压 590MB，
        #    实测 RSS 冲到 60GB、60 分钟 0 步。改成 mmap 的非压缩 .npy：进程间共享页，RSS 恒定。
        self._det_tokens = None
        self._det_index = None
        if det_glob_npz:
            import numpy as _np
            _p = str(det_glob_npz)
            if _p.endswith(".npy"):
                self._det_tokens = _np.load(_p, mmap_mode="r")
                _nj = _p[:-4] + "_names.json"
                _names = json.loads(Path(_nj).read_text())
                self._det_index = {str(n): i for i, n in enumerate(_names)}
            else:                                    # 兼容旧 npz
                _z = _np.load(_p, allow_pickle=True)
                self._det_tokens = _z["tokens"]
                self._det_index = {str(n): i for i, n in enumerate(_z["names"])}
            print(f"[data] 检测器全局 token（mmap）: {len(self._det_index)} 瓦片, "
                  f"shape={self._det_tokens.shape}", flush=True)
        self.mask_expand = float(mask_expand)
        # 上下文模块 A/B/C（ctx_abc.py，2026-11-05）：整幅瓦片特征 + 几何（+ 可选邻居框）
        self.use_ctx = bool(use_ctx)
        self.ctx_tile_size = int(ctx_tile_size)
        self.ctx_neighbors = ctx_neighbors or {}
        self.ctx_k = int(ctx_k)
        self.n_ctx_no_nbr = 0
        self.use_scale_cond = bool(use_scale_cond) or bool(use_zoom)
        self.use_zoom = bool(use_zoom)
        self.zoom_ratio = None if zoom_ratio in (None, 0) else float(zoom_ratio)
        self.use_corner = bool(use_corner) and self.use_hr_edge
        self.corner_sigma = float(corner_sigma)
        self.code_dim = int(code_dim)
        # 随机投影阴影 / 树冠暗斑（shadow_guide.add_shadow，仅训练期）
        self.shadow_aug_prob = float(shadow_aug_prob)
        self.shadow_aug_strength = float(shadow_aug_strength)
        self.shadow_aug_mode = "blob" if shadow_aug_blob else "cast"
        self.n_shadow_aug = 0
        self.n_occ_aug = 0
        self.rng = random.Random(seed)
        self._calls = 0
        self.n_missing = 0
        self.n_substituted = 0

        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        from whumix import iter_instances

        instances = list(iter_instances(root, split, max_instances=None))
        if tile_whitelist:      # C 模块：只用"有检测框"的那批瓦片（见 docs/06 §5 的邻居来源约束）
            _wl = set(tile_whitelist)
            _before = len(instances)
            instances = [it for it in instances if it["file_name"] in _wl]
            print(f"[data] 白名单瓦片过滤：{_before} → {len(instances)} 栋", flush=True)
        if not instances:
            raise ValueError(f"no instances in {root}/{split}")
        if max_instances and max_instances < len(instances):
            if ordered_subset:
                instances = instances[:max_instances]        # 旧行为：按标注顺序取前 N
            else:
                random.Random(seed).shuffle(instances)       # 固定种子随机抽，避免场景偏置
                instances = instances[:max_instances]
            self._log_composition(instances, max_instances, ordered_subset)
        if dense_alpha and dense_alpha > 0:        # G5：粘连难例加权采样（纯数据侧）
            instances = expand_dense_instances(instances, alpha=dense_alpha,
                                               beta=dense_beta, cap=dense_cap, seed=seed)
        self.instances = instances
        # 跨实例记忆（2026-11-08）：同瓦片兄弟多边形索引，teacher forcing 用。
        # mem_frac：每样本随机抽一部分兄弟作为"已完成"（模拟推理期的任意进度）。
        self.use_memory = bool(use_memory)
        self.mem_frac = float(mem_frac)
        self._sib_polys: Dict[str, list] = {}
        if self.use_memory:
            for it in instances:
                self._sib_polys.setdefault(it["file_name"], []).append(it["polygon"])
        # 邻居轮廓文本条件（docs/21 方向 A）：共用 _sib_polys 兄弟索引
        self.use_neighbor_prompt = bool(use_neighbor_prompt)
        self.neighbor_k = int(neighbor_k)
        self.neighbor_side = int(neighbor_side)
        if self.use_neighbor_prompt and not self._sib_polys:
            for it in instances:
                self._sib_polys.setdefault(it["file_name"], []).append(it["polygon"])
        # 方向 B：邻居轮廓画进输入图（对照臂，docs/21）
        self.use_neighbor_draw = bool(use_neighbor_draw)
        self.draw_alpha = float(draw_alpha)
        if self.use_neighbor_draw and not self._sib_polys:
            for it in instances:
                self._sib_polys.setdefault(it["file_name"], []).append(it["polygon"])
        # 方向 D：布局密度标量进 ScaleCond 编码（LANet density prior）
        self.use_density_code = bool(use_density_code)
        if self.use_density_code and not self._sib_polys:
            for it in instances:
                self._sib_polys.setdefault(it["file_name"], []).append(it["polygon"])

    def __len__(self) -> int:
        return len(self.instances)

    def _occlude_corners(self, crop, polygon, origin, tile_image):
        """按 `occ_aug_prob` 决定是否对这张裁剪图做四角遮挡增强（实现在模块级 `occlude_corners`）。"""
        self._calls += 1
        info = __import__("torch").utils.data.get_worker_info()
        worker = info.id if info is not None else -1
        rng = _aug_rng(self.seed, worker, self._calls)
        if rng.random() >= self.occ_aug_prob:
            return crop
        occluded = occlude_corners(crop, polygon, origin, tile_image, rng,
                                  corner_frac=self.occ_corner_frac,
                                  max_corners=self.occ_max_corners)
        if occluded is not crop:
            self.n_occ_aug += 1
        return occluded

    def _add_shadow(self, crop, polygon, origin):
        """按 `shadow_aug_prob` 决定是否给这张裁剪图加**投影阴影 / 树冠暗斑**（只压亮度，GT 不动）。

        为什么要这个增强：阴影把檐口压平后，"边界在哪"这件事在像素上真的消失了，
        模型只能按几何关系外推 —— 训练时见过这种样本，推理时才不会把影子边缘当楼边界
        （典型错误是把楼画大一圈）。实现在 `vectorllm.shadow_guide.add_shadow`。
        """
        from .shadow_guide import add_shadow

        info = __import__("torch").utils.data.get_worker_info()
        worker = info.id if info is not None else -1
        rng = _aug_rng(self.seed, worker, self._calls + 17)
        if rng.random() >= self.shadow_aug_prob:
            return crop
        local = [(x - origin[0], y - origin[1]) for x, y in polygon]
        out = add_shadow(crop, local, rng, strength=self.shadow_aug_strength,
                         mode=self.shadow_aug_mode)
        if out is not crop:
            self.n_shadow_aug += 1
        return out

    def _log_composition(self, instances, n: int, ordered: bool) -> None:
        """打印子集的场景/形状/面积概况，让"数据量消融"可复盘（随机抽 vs 按顺序取前 N）。"""
        import statistics as _st

        tiles = len({item["file_name"] for item in instances})
        sides = [max(item["bbox"][2] - item["bbox"][0], item["bbox"][3] - item["bbox"][1]) for item in instances]
        verts = [len(item["polygon"]) for item in instances]
        print(f"[data] 子集 {n} 栋 / {tiles} 瓦片（{'按标注顺序' if ordered else '固定种子随机抽'}）："
              f"长边中位 {_st.median(sides):.1f}px、顶点中位 {_st.median(verts)}、"
              f"首瓦片 {instances[0]['file_name']}", flush=True)

    def _jitter_ratio(self) -> float:
        """裁剪外扩比：按 `(seed, worker_id, 调用计数)` 派生随机数。

        不能共用一份 `self.rng`：DataLoader 的 worker 是 fork 出来的，各 worker 会拿到
        完全相同的 RNG 状态，于是不同 worker 产生同一串抖动序列（外部审阅指出，已核实）。
        """
        import random as _random

        import torch as _torch

        self._calls += 1
        info = _torch.utils.data.get_worker_info()
        worker = info.id if info is not None else -1
        # random.Random 只接受 int/float/str/bytes —— 不能直接喂 tuple（踩过）
        seed = (int(self.seed) * 1_000_003) ^ (int(worker) * 79_219) ^ (int(self._calls) * 104_729)
        return _random.Random(seed).uniform(*self.jitter)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        import random as _random

        from PIL import Image

        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        from whumix import crop_box_for

        # 瓦片可能还在往本地缓存里拷（或个别文件缺失）：换邻近样本重试，别让训练直接崩。
        # 但**必须计数**：静默兜底会让"缓存不全"变成"数据分布悄悄变了"。
        item, image = None, None
        for attempt in range(8):
            candidate = self.instances[(index + attempt) % len(self.instances)]
            image_path = Path(self.jpeg) / self.split / "image" / candidate["file_name"]
            if not image_path.exists():
                self.n_missing += 1
                continue
            try:
                image = Image.open(image_path).convert("RGB")
                item = candidate
                if attempt:
                    self.n_substituted += 1
                break
            except Exception:
                self.n_missing += 1
                continue
        if (self.n_missing + self.n_substituted) and (self.n_missing + self.n_substituted) % 200 == 0:
            print(f"[data] ⚠️ 缺图 {self.n_missing} 次 / 兜底替换 {self.n_substituted} 次"
                  f"（检查 {self.jpeg}/{self.split}/image 是否拷全）", flush=True)
        if item is None:
            raise FileNotFoundError(f"裁剪所需瓦片连续 8 个都不可用，检查本地缓存: {self.jpeg}/{self.split}/image")
        ratio = self.fixed_ratio if self.fixed_ratio else self._jitter_ratio()
        crop_angle = 0.0          # OBB 条件位置编码用（转正裁剪的真实朝向）
        tfm = None                # 跨实例记忆的占用 warp 需要（rot 分支会填充）
        if self.crop_mode in ("mask", "rbox", "rot"):
            # 旋转框/填黑裁剪（遥感专用）：框外涂黑把"邻楼"从输入里物理删掉；
            # rbox 还额外把建筑最长边转到与坐标轴对齐。
            from .rbox import aabb_mask_crop, min_area_rect, rbox_crop, tile_to_crop
            poly = item["polygon"]
            if self.crop_mode in ("rbox", "rot"):
                # rbox = 转正 + 框外涂黑；rot = 只转正、**保留全部上下文**（docs/01 §4 实测推荐）
                crop, tfm = rbox_crop(image, min_area_rect(poly), expand=ratio,
                                      mask_outside=(self.crop_mode == "rbox"),
                                      mask_expand=self.mask_expand)
            else:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                crop, tfm = aabb_mask_crop(image, (min(xs), min(ys), max(xs), max(ys)),
                                           expand=ratio, mask_expand=self.mask_expand)
            if crop is None:      # 极小目标（短边 <8px）转不正 → 回退到轴对齐裁剪，别让整个 epoch 挂掉
                x1, y1, x2, y2 = crop_box_for(poly, image.size, ratio)
                crop = image.crop((x1, y1, x2, y2))
                local_poly = poly
            else:
                # 尺度编码/全局位置编码都要一个"裁剪框（瓦片坐标）"：用旋转窗口的外接框
                x1, y1 = tfm["c"][0] - tfm["size"][0] / 2, tfm["c"][1] - tfm["size"][1] / 2
                x2, y2 = tfm["c"][0] + tfm["size"][0] / 2, tfm["c"][1] + tfm["size"][1] / 2
                local_poly = tile_to_crop(poly, tfm)
                crop_angle = float(tfm.get("angle", 0.0))
        else:
            x1, y1, x2, y2 = crop_box_for(item["polygon"], image.size, ratio)
            x1, y1, x2, y2 = jitter_box((x1, y1, x2, y2), image.size, self.box_shift, self.box_aspect,
                                        _random.Random((int(self.seed) * 7919) ^ (int(self._calls) * 104729)))
            crop = image.crop((x1, y1, x2, y2))
            local_poly = item["polygon"]
        if self.occ_aug_prob > 0:
            crop = self._occlude_corners(crop, item["polygon"], (x1, y1), image)
        if self.shadow_aug_prob > 0:
            crop = self._add_shadow(crop, item["polygon"], (x1, y1))
        # 方向 B：GT 兄弟邻居轮廓画进裁剪图（半透明灰）——与方向 A 同信息、不同载体
        if self.use_neighbor_draw:
            from .rbox import tile_to_crop as _t2c

            _sibs_b = [p for p in self._sib_polys.get(item["file_name"], [])
                       if p is not item["polygon"]]
            _rng_b = _random.Random((int(self.seed) * 31) ^ (int(self._calls) * 7919))
            _kb = _rng_b.randint(0, min(3, len(_sibs_b)))
            _done_b = _rng_b.sample(_sibs_b, _kb) if _sibs_b else []
            if _done_b:
                from PIL import ImageDraw as _ID

                _ov = crop.convert("RGBA")
                _dr = _ID.Draw(_ov, "RGBA")
                for _p in _done_b:
                    if self.crop_mode in ("mask", "rbox", "rot") and tfm is not None and tfm.get("ok"):
                        _cl = _t2c(_p, tfm)
                    else:
                        _cl = [(px - x1, py - y1) for px, py in _p]
                    _cl = [(u, v) for u, v in _cl
                           if -20 <= u <= crop.size[0] + 20 and -20 <= v <= crop.size[1] + 20]
                    if len(_cl) >= 3:
                        _a = int(self.draw_alpha * 255)
                        _dr.polygon(_cl, fill=(128, 128, 128, _a // 2), outline=(90, 90, 90, _a))
                crop = _ov.convert("RGB")
        pixel_values = self.image_processor(crop, return_tensors="pt")["pixel_values"][0]

        # 跨实例空间记忆（teacher forcing）：随机抽兄弟多边形栅格化成"已完成占用"，
        # 与当前 rbox 边界一起组成 [2,64,64]。兄弟多边形同步转到裁剪框坐标。
        if self.use_memory:
            import numpy as _np_mem
            from .instance_memory import rasterize_polys, rbox_edge_map, warp_occ_to_crop

            sibs = [p for p in self._sib_polys.get(item["file_name"], [])
                    if p is not item["polygon"]]
            _rng = _random.Random((int(self.seed) * 7919) ^ (int(self._calls) * 15485863))
            k = int(len(sibs) * self.mem_frac * _rng.random() * 2)     # 0 ~ mem_frac·2 均匀
            done = _rng.sample(sibs, min(k, len(sibs))) if sibs else []
            occ_tile = rasterize_polys(done, image.size[1], image.size[0])
            if self.crop_mode in ("mask", "rbox", "rot") and tfm is not None and tfm.get("ok"):
                occ_crop = warp_occ_to_crop(occ_tile, tfm, tfm["size"][0], tfm["size"][1])
                edge = rbox_edge_map(tfm["size"][0], tfm["size"][1],
                                     (tfm["c"][0], tfm["c"][1],
                                      item["bbox"][2] - item["bbox"][0],
                                      item["bbox"][3] - item["bbox"][1]), expand=1.0)
            else:   # 轴对齐裁剪：占用直接平移
                occ_crop = occ_tile[int(y1):int(y1) + crop.size[1], int(x1):int(x1) + crop.size[0]]
                if occ_crop.shape[:2] != crop.size[::-1]:
                    occ_crop = np.zeros((crop.size[1], crop.size[0]), dtype=np.float32)
                edge = rbox_edge_map(crop.size[0], crop.size[1],
                                     (crop.size[0] / 2, crop.size[1] / 2, x2 - x1, y2 - y1),
                                     expand=1.0)
            import cv2 as _cv2
            occ_s = _cv2.resize(occ_crop, (64, 64), interpolation=_cv2.INTER_AREA)
            edge_s = _cv2.resize(edge, (64, 64), interpolation=_cv2.INTER_AREA)
            _mem_maps = torch.from_numpy(_np_mem.stack([occ_s, edge_s]).astype(_np_mem.float32))
        else:
            _mem_maps = None

        ctx_out = {}
        if self.use_ctx:
            from .ctx_abc import ctx_geometry, select_neighbors, tile_tensor_for_ctx
            from .rbox import min_area_rect as _min_area_rect

            if self.crop_mode in ("mask", "rbox", "rot") and isinstance(tfm, dict) and tfm.get("ok"):
                crop_c, crop_ang = tfm["c"], float(tfm.get("angle", 0.0))
            else:      # aabb / 极小目标回退路径：轴对齐、角度 0
                crop_c, crop_ang = ((x1 + x2) / 2.0, (y1 + y2) / 2.0), 0.0
            obj = _min_area_rect(item["polygon"])
            ctx_out["ctx_tile_pixel_values"] = tile_tensor_for_ctx(image, self.ctx_tile_size)
            ctx_out["ctx_geom"] = ctx_geometry(crop_center=crop_c, crop_size=crop.size,
                                               crop_angle=crop_ang, obj=obj,
                                               tile_size=image.size)
            nbr = self.ctx_neighbors.get(item.get("file_name") or "")
            if nbr:
                b_, s_, m_ = select_neighbors(nbr["boxes"], nbr["scores"], obj, k=self.ctx_k)
                ctx_out["ctx_neighbors"] = b_
                ctx_out["ctx_neighbor_scores"] = s_
                ctx_out["ctx_neighbor_mask"] = m_
            else:
                self.n_ctx_no_nbr += 1

        from .geometry import GRID_SIZE, pixel_to_grid

        local = (local_poly if self.crop_mode in ("mask", "rbox", "rot")
                 else [(x - x1, y - y1) for x, y in item["polygon"]])
        grid = pixel_to_grid(local, crop.size, GRID_SIZE)
        if self.vertex_count_token:
            grid = grid[:60]        # 60 点 × 2 token + prompt 仍在 max_length 内，避免 <nvK> 与点数不一致
            target = _fmt_target(grid, with_count=True)
        else:
            target = _fmt_target(grid, with_count=False)

        # 邻居轮廓文本条件（方向 A）：随机抽 GT 兄弟（teacher forcing）→ 裁剪框 128 网格
        # token 块。插在 <pixel> 之后、指令之前（先看图，再读邻居文本，再执行指令）。
        # K 随机含 0 → 模型学会"无邻居时忽略该段"。
        _nbr_block = ""
        if self.use_neighbor_prompt and tfm is not None and tfm.get("ok"):
            from .neighbor_prompt import build_neighbor_context

            sibs = [p for p in self._sib_polys.get(item["file_name"], [])
                    if p is not item["polygon"]]
            _rng_n = _random.Random((int(self.seed) * 104729) ^ (int(self._calls) * 15485863))
            k = _rng_n.randint(0, min(self.neighbor_k, len(sibs)))
            done = _rng_n.sample(sibs, k) if sibs else []
            _nbr_block = build_neighbor_context(done, tfm,
                                                max_neighbors=self.neighbor_k,
                                                max_side=self.neighbor_side)

        _raw = raw_prompt()
        if _nbr_block:
            _raw = _raw.replace(PIXEL_TOKEN, PIXEL_TOKEN + "\n" + _nbr_block, 1)
        prompt = expand_pixel_placeholder(_raw)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target + _end_token, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + target_ids)[: self.max_length]
        labels = ([IGNORE_INDEX] * len(prompt_ids) + list(target_ids))[: self.max_length]
        item_out = {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "n_target_tokens": torch.tensor(len(target_ids), dtype=torch.long),
        }
        if _mem_maps is not None:
            item_out["memory_maps"] = _mem_maps
        if self.use_global:
            from .global_fusion import POS_DIM, POS_DIM_ANGLE, pos_box_from_box, pos_code_from_box

            item_out["global_pixel_values"] = self.image_processor(image, return_tensors="pt")["pixel_values"][0]
            item_out["pos_code"] = pos_code_from_box(
                (x1, y1, x2, y2), image.size,
                dim=(POS_DIM_ANGLE if self.pos_angle else POS_DIM),
                angle=(crop_angle if self.pos_angle else None))
            # 原始归一化框 (cx/W, cy/H, w/W, h/H)：空间注意力偏置必须吃这个（pos_code 是 Fourier 编码）
            item_out["pos_box"] = pos_box_from_box((x1, y1, x2, y2), image.size)
            item_out["crop_angle"] = torch.tensor(float(crop_angle))
        if self._det_index is not None:
            _nm = item.get("file_name") or item.get("name") or ""
            _i = self._det_index.get(str(_nm))
            if _i is not None:
                import numpy as _np
                item_out["det_global_tokens"] = torch.from_numpy(
                    _np.array(self._det_tokens[_i], dtype="float32"))
        # 尺度条件 / 放大分支（scale_edge.py）：尺度编码一律基于"目标框 vs 实际裁剪框"
        if self.use_scale_cond or self.use_zoom:
            from .scale_edge import build_zoom_inputs, object_box_of, scale_code

            obj_box = object_box_of(item["polygon"])
            crop_box = (x1, y1, x2, y2)
            # 方向 D：density_code 拼在 scale_code 后 → 总长 = code_dim（80 = 64 + 16）。
            # scale_code 恒用 64；密度段 16；ScaleCond 的 mlp 输入按 code_dim=80 构建。
            _sc_dim = 64 if self.use_density_code else self.code_dim
            item_out["scale_code"] = scale_code(obj_box, crop_box, dim=_sc_dim)
            if self.use_density_code:
                from .scale_edge import density_code

                sibs_b = [p for p in self._sib_polys.get(item["file_name"], [])
                          if p is not item["polygon"]]
                nb_boxes = []
                for p in sibs_b:
                    xs_ = [q[0] for q in p]; ys_ = [q[1] for q in p]
                    nb_boxes.append((min(xs_), min(ys_), max(xs_), max(ys_)))
                d_code = density_code(obj_box, nb_boxes, dim_extra=self.code_dim - 64)
                item_out["scale_code"] = torch.cat([item_out["scale_code"], d_code])
            if self.use_zoom:
                # 主裁剪比从"目标框 vs 裁剪框"反推（抖动/平移之后才是真实比例）
                r_main = max((crop_box[2] - crop_box[0]) / max(obj_box[2] - obj_box[0], 1e-6),
                             (crop_box[3] - crop_box[1]) / max(obj_box[3] - obj_box[1], 1e-6))
                pv, _code = build_zoom_inputs(image, obj_box, self.image_processor,
                                             r_main=r_main, r_zoom=self.zoom_ratio,
                                             code_dim=self.code_dim)
                item_out["zoom_pixel_values"] = pv
        if self.use_hr_edge:
            from .global_fusion import edge_label_from_polygon, hr_tensor_from_crop

            item_out["hr_pixel_values"] = hr_tensor_from_crop(crop, self.hr_size)
            item_out["edge_labels"] = edge_label_from_polygon(local, crop.size, self.edge_size,
                                                              self.edge_width)
            if self.use_corner:
                from .scale_edge import corner_label_from_polygon

                item_out["corner_labels"] = corner_label_from_polygon(
                    local, crop.size, self.edge_size, sigma=self.corner_sigma)
        if ctx_out:
            item_out.update(ctx_out)
        return item_out


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """右侧 padding（labels 用 -100 填，attention_mask 用 0 填）。"""
    max_len = max(int(item["input_ids"].shape[0]) for item in batch)
    input_ids, labels, attention = [], [], []
    for item in batch:
        pad = max_len - int(item["input_ids"].shape[0])
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([item["labels"], torch.full((pad,), IGNORE_INDEX, dtype=torch.long)]))
        attention.append(torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
    if "global_pixel_values" in batch[0]:
        global_pv = torch.stack([b["global_pixel_values"] for b in batch], dim=0)
        pos_code = torch.stack([b["pos_code"] for b in batch], dim=0)
    out = {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "attention_mask": torch.stack(attention, dim=0),
    }
    if "global_pixel_values" in batch[0]:
        out["global_pixel_values"] = global_pv
        out["pos_code"] = pos_code
        if "pos_box" in batch[0]:
            out["pos_box"] = torch.stack([b["pos_box"] for b in batch], dim=0)
    # 高分辨率边界分支（训练期）；没有就不加 key，模型侧按 None 处理
    if "hr_pixel_values" in batch[0]:
        out["hr_pixel_values"] = torch.stack([b["hr_pixel_values"] for b in batch], dim=0)
    if "edge_labels" in batch[0]:
        out["edge_labels"] = torch.stack([b["edge_labels"] for b in batch], dim=0)
    # 尺度条件 / 放大分支 / 角点热图（scale_edge.py）
    if "scale_code" in batch[0]:
        out["scale_code"] = torch.stack([b["scale_code"] for b in batch], dim=0)
    if "zoom_pixel_values" in batch[0]:
        out["zoom_pixel_values"] = torch.stack([b["zoom_pixel_values"] for b in batch], dim=0)
    if "corner_labels" in batch[0]:
        out["corner_labels"] = torch.stack([b["corner_labels"] for b in batch], dim=0)
    if any("det_global_tokens" in b for b in batch):
        # 只有导出了 token 的瓦片才有该键；缺的用零张量补齐（配合零初始化 adapter，等价于无上下文）
        _ref = next(b["det_global_tokens"] for b in batch if "det_global_tokens" in b)
        out["det_global_tokens"] = torch.stack(
            [b.get("det_global_tokens", torch.zeros_like(_ref)) for b in batch], dim=0)
    if any("memory_maps" in b for b in batch):
        # 跨实例记忆：负样本/无兄弟样本没有该键，用全零双通道图补齐（等价于无历史占用）
        _ref = next(b["memory_maps"] for b in batch if "memory_maps" in b)
        out["memory_maps"] = torch.stack(
            [b.get("memory_maps", torch.zeros_like(_ref)) for b in batch], dim=0)
    # 上下文模块 A/B/C（ctx_abc.py）：整幅瓦片 + 几何 (+ 邻居框)
    for key in ("ctx_tile_pixel_values", "ctx_geom", "ctx_neighbors",
                "ctx_neighbor_scores", "ctx_neighbor_mask"):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out

def dataset_stats(jsonl: str | Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for line in open(jsonl, encoding="utf-8"):
        kind = json.loads(line).get("kind", "?")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


# --------------------------------------------------------------------------- 多域联合训练


def _point_in_polygon(x: float, y: float, poly) -> bool:
    """射线法判点是否在多边形内（无 shapely 依赖，训练进程里也能跑）。"""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xin:
                inside = not inside
    return inside


def _box_hits_building(box, polys) -> bool:
    """框里是否**真的**有建筑（用于筛"难负样本"）。

    只要有一个多边形顶点落在框内、或框的某个角落在多边形内、或两者的边相交，就算"有楼"。
    比"按面积比"更严格一点也没关系：负样本宁可少，也不能错标（错标 = 教模型"看见楼也说没有"）。
    """
    x1, y1, x2, y2 = box
    for poly in polys:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        if max(xs) < x1 or min(xs) > x2 or max(ys) < y1 or min(ys) > y2:
            continue                                   # bbox 都不交，直接跳过
        if any(x1 <= px <= x2 and y1 <= py <= y2 for px, py in poly):
            return True                                # 顶点在框内
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        if any(_point_in_polygon(cx, cy, poly) for cx, cy in corners):
            return True                                # 框角在多边形内
        n = len(poly)
        for i in range(n):
            (ax, ay), (bx, by) = poly[i], poly[(i + 1) % n]
            if (ax <= x2 and bx >= x1 and ay <= y2 and by >= y1) and \
               ((ax < x1) != (bx < x1)) and ((ay < y1) != (by < y1)):
                return True                            # 边穿过框（粗判）
    return False


class MultiCocoDataset(Dataset):
    """**多域**联合训练集：一次采样来自多个 COCO 域（WHU-Mix / gzda / ramp Dhaka / ramp Lubumbashi）。

    设计目标：让"轮廓生成"和"弃权判定"在**同一次训练**里从多头数据一起学，
    而不是像之前那样"先在 WHU 上学轮廓，再单独补一行 embedding 学弃权"。

    Args:
        domains: [(root, split, weight), ...]，weight 是相对采样权重（按实例数归一化后再加权）。
        neg_frac: 每个 batch 里 `<no_object>` 负样本的比例（0 = 全正样本）。
        neg_empty_ratio: 负样本里"纯空瓦片"占多少（其余是"有楼瓦片里的背景框"）。
        jitter: 正样本裁剪框外扩比区间。
        max_total: 可选上限（调试/控时长）。

    正样本支持**四角遮挡增强**（`occ_aug_prob`，与 `CropCocoDataset` 共用 `occlude_corners`）：
    只遮角、GT 不动，教模型"角被树冠吃了也要按矩形关系补出来"。

    负样本是**在线生成**的（不落盘）：从该域随机挑一张瓦片，随机撒一个框，
    用几何校验确认框里没有建筑，才当成 `<no_object>` 样本 —— 和 `build_abstain_dataset.py`
    的配方一致，但省掉了 1.2 万张预切 PNG 和一次 NFS 搬运。
    """

    def __init__(self, domains, image_processor, tokenizer, neg_frac: float = 0.3,
                 neg_empty_ratio: float = 0.5, jitter=(1.0, 1.6), fixed_ratio: float | None = None,
                 max_length: int = 1024, seed: int = 0, max_total: int | None = None,
                 neg_tries: int = 12, domain_balanced: bool = False, max_vertices: int = 120,
                 occ_aug_prob: float = 0.0, occ_corner_frac: float = 0.35,
                 occ_max_corners: int = 4, max_per_domain: int | None = None,
                 box_shift: float = 0.0, box_aspect: float = 0.0,
                 vertex_count_token: bool = False, use_global: bool = False,
                 use_hr_edge: bool = False, hr_size: int = 256, edge_size: int = 64,
                 edge_width: int = 2, use_scale_cond: bool = False, code_dim: int = 64,
                 pos_angle: bool = False):
        import random as _random

        self.vertex_count_token = bool(vertex_count_token)

        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.jitter = jitter
        self.fixed_ratio = fixed_ratio
        self.neg_frac = max(0.0, min(float(neg_frac), 0.9))
        self.neg_empty_ratio = float(neg_empty_ratio)
        self.seed = seed
        self.neg_tries = neg_tries
        # 四角遮挡增强（与 CropCocoDataset 同一实现）：多域版本曾经漏接，见 occlude_corners 的 docstring
        self.occ_aug_prob = float(occ_aug_prob)
        self.occ_corner_frac = float(occ_corner_frac)
        self.occ_max_corners = int(occ_max_corners)
        self.box_shift = float(box_shift)      # 中心平移抖动（占框半边长比例）
        self.box_aspect = float(box_aspect)    # 长宽比抖动（x/y 各自 ±）
        self.use_global = bool(use_global)     # 是否额外喂"整幅瓦片"做全局上下文融合
        self.use_scale_cond = bool(use_scale_cond)   # 尺度编码（ScaleCond）
        self.code_dim = int(code_dim)
        self.use_hr_edge = bool(use_hr_edge)   # 高分辨率边界分支（见 CropCocoDataset 同名参数）
        self.hr_size = int(hr_size)
        self.edge_size = int(edge_size)
        self.edge_width = int(edge_width)
        self.n_occ_aug = 0
        self._calls = 0
        self.domain_balanced = domain_balanced
        self.n_missing_at_init = 0
        self.n_missing = 0
        self.n_substituted = 0
        self.n_neg_fail = 0
        self.stats: Dict[str, int] = {}

        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        from whumix import load_coco, segmentation_to_polygon

        self.domains = []
        for root, split, weight in domains:
            coco = load_coco(Path(root), split)
            images = {im["id"]: im for im in coco["images"]}
            anns: Dict[int, List] = {}
            n_dropped = 0
            for ann in coco["annotations"]:
                poly = segmentation_to_polygon(ann.get("segmentation"))
                if len(poly) < 3:
                    continue
                # 顶点过多的复杂多边形（实测 RAMP/gzda 里有上百个顶点）会把序列拉到 max_length，
                # CE 的 float32 副本 ≈ batch×seq×151927×4B → 直接 OOM（22:55 就是这么挂的）
                if len(poly) > max_vertices:
                    n_dropped += 1
                    continue
                anns.setdefault(ann["image_id"], []).append(poly)
            positives, empties = [], []
            for image_id, image in images.items():
                entry = (Path(root) / split / "image" / image["file_name"],
                         image.get("width"), image.get("height"), anns.get(image_id, []))
                (positives if anns.get(image_id) else empties).append(entry)
            keep = [p for p in positives if Path(p[0]).exists()]
            keep_empty = [p for p in empties if Path(p[0]).exists()]
            self.n_missing_at_init += (len(positives) - len(keep)) + (len(empties) - len(keep_empty))
            positives, empties = keep, keep_empty
            if not positives:
                print(f"[multi] ⚠️ {root}/{split} 本地一张都没有，跳过该域（拷贝没完成？）", flush=True)
                continue
            self.domains.append({"root": root, "split": split, "weight": float(weight),
                                 "positives": positives, "empties": empties,
                                 "n_pos": sum(len(p[3]) for p in positives)})
            print(f"[multi] {root}/{split}: 有楼瓦片 {len(positives)}（{self.domains[-1]['n_pos']} 栋）/ "
                  f"空瓦片 {len(empties)} / 权重 {weight} / 丢弃超长多边形 {n_dropped}", flush=True)
        if domain_balanced:
            total_weight = sum(d["weight"] for d in self.domains)
            for d in self.domains:
                d["prob"] = d["weight"] / max(total_weight, 1e-9)
            print(f"[multi] 域均衡采样：{[(Path(d['root']).name, round(d['prob'], 3)) for d in self.domains]}",
                  flush=True)
        else:
            total_weight = sum(d["weight"] * d["n_pos"] for d in self.domains)
            for d in self.domains:
                d["prob"] = d["weight"] * d["n_pos"] / max(total_weight, 1e-9)
        # 正样本的全局索引（每个域内 (瓦片, 第几栋)）
        # `max_per_domain`：**按域**截断正样本（例如 3 域 × 2 万 = 6 万，域间完全均衡）。
        # 只截 `max_total` 不够：`positive_index` 是按域顺序拼起来的，全局截断等于
        # "大域占满、小域被挤掉"，拿它跟单域的 60k 比较就不是同一条数据量曲线。
        self.positive_index = []
        for di, d in enumerate(self.domains):
            entries = [(di, tile, k) for tile in d["positives"] for k in range(len(tile[3]))]
            if max_per_domain and max_per_domain < len(entries):
                _random.Random(seed + 7919 * (di + 1)).shuffle(entries)
                entries = entries[:max_per_domain]
            self.positive_index.extend(entries)
            print(f"[multi] 域 {Path(d['root']).name} 正样本 {len(entries)}", flush=True)
        if max_total and max_total < len(self.positive_index):
            _random.Random(seed).shuffle(self.positive_index)
            self.positive_index = self.positive_index[:max_total]
        self.n_neg = int(len(self.positive_index) * self.neg_frac / max(1e-9, 1 - self.neg_frac))
        self.total = len(self.positive_index) + self.n_neg
        print(f"[multi] 正样本 {len(self.positive_index)} + 负样本 {self.n_neg} = {self.total}"
              f"（neg_frac={self.neg_frac}，空瓦片占比 {self.neg_empty_ratio}）", flush=True)

    def __len__(self) -> int:
        return self.total

    def _pick_domain(self, rng) -> dict:
        r = rng.random()
        acc = 0.0
        for d in self.domains:
            acc += d["prob"]
            if r <= acc:
                return d
        return self.domains[-1]

    def _ratio(self, rng) -> float:
        return self.fixed_ratio if self.fixed_ratio else rng.uniform(*self.jitter)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        import random as _random

        import torch as _torch
        from PIL import Image

        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
        from whumix import crop_box_for

        info = _torch.utils.data.get_worker_info()
        worker = info.id if info is not None else -1
        self._calls += 1
        seed = ((int(self.seed) * 1_000_003) ^ (int(worker) * 79_219)
                ^ (int(index) * 104_729) ^ (int(self._calls) * 15_485_863))
        rng = _random.Random(seed)

        is_negative = index >= len(self.positive_index)
        if not is_negative:
            # 瓦片可能还在往 /dev/shm 里拷（NFS 很慢）→ 换邻近样本重试，并**计数**
            # （静默兜底会让"拷贝不全"变成"数据分布悄悄变了"，所以必须记 n_missing）
            image = None
            tile = None
            for attempt in range(6):
                di, tile, k = self.positive_index[(index + attempt * 977) % len(self.positive_index)]
                try:
                    image = Image.open(tile[0]).convert("RGB")
                    if attempt:
                        self.n_substituted += 1
                    break
                except Exception:
                    self.n_missing += 1
                    image = None
            if image is None:
                raise FileNotFoundError(f"连续 6 个正样本瓦片都读不到（检查 /dev/shm 拷贝是否完成）：{tile[0]}")
            path, width, height, polys = tile
            poly = polys[k]
            ratio = self._ratio(rng)
            x1, y1, x2, y2 = crop_box_for(poly, image.size, ratio)
            x1, y1, x2, y2 = jitter_box((x1, y1, x2, y2), image.size, self.box_shift, self.box_aspect, rng)
            crop = image.crop((x1, y1, x2, y2))
            local = [(x - x1, y - y1) for x, y in poly]
            if self.occ_aug_prob > 0 and rng.random() < self.occ_aug_prob:
                occluded = occlude_corners(crop, poly, (x1, y1), image, rng,
                                          corner_frac=self.occ_corner_frac,
                                          max_corners=self.occ_max_corners)
                if occluded is not crop:
                    self.n_occ_aug += 1
                    crop = occluded
            self.stats["positive"] = self.stats.get("positive", 0) + 1
            return self._pack(crop, local, no_object=False, image=image, box=(x1, y1, x2, y2))

        # ---- 负样本：随机撒框 + 几何校验（框里必须没有建筑）----
        d = self._pick_domain(rng)
        use_empty = rng.random() < self.neg_empty_ratio and d["empties"]
        pool = d["empties"] if use_empty else d["positives"]
        for _ in range(self.neg_tries):
            path, width, height, polys = pool[rng.randrange(len(pool))]
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                self.n_missing += 1
                continue
            side = rng.uniform(40, 260) if use_empty else rng.uniform(40, 200)
            cx = rng.uniform(side / 2, max(image.size[0] - side / 2, side / 2 + 1))
            cy = rng.uniform(side / 2, max(image.size[1] - side / 2, side / 2 + 1))
            box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
            if not use_empty and _box_hits_building(box, polys):
                continue                                  # 压到楼 → 这个框不能当负样本
            x1, y1 = int(max(0, box[0])), int(max(0, box[1]))
            x2, y2 = int(min(image.size[0], box[2])), int(min(image.size[1], box[3]))
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crop = image.crop((x1, y1, x2, y2))
            self.stats["negative_empty" if use_empty else "negative_background"] = \
                self.stats.get("negative_empty" if use_empty else "negative_background", 0) + 1
            return self._pack(crop, None, no_object=True, image=image, box=(x1, y1, x2, y2))
        # 兜底：采样失败就退化成"空瓦片整块"（仍然安全：空瓦片本来就没有楼）
        self.n_neg_fail += 1
        d0 = self.domains[0]
        path, width, height, polys = d0["empties"][rng.randrange(len(d0["empties"]))]
        image = Image.open(path).convert("RGB")
        side = min(128, image.size[0], image.size[1])
        crop = image.crop((0, 0, side, side))
        return self._pack(crop, None, no_object=True, image=image, box=(0, 0, side, side))

    def _pack(self, crop, local_polygon, no_object: bool, image=None, box=None) -> Dict[str, torch.Tensor]:
        """`image`/`box` 给了才能产出全局瓦片（GlobalFusion）与尺度编码（ScaleCond）——
        多域训练以前一直没接这条链路，导致 `--global-context` 在多域下被静默跳过。"""
        from .geometry import GRID_SIZE, pixel_to_grid

        pixel_values = self.image_processor(crop, return_tensors="pt")["pixel_values"][0]

        ctx_out = {}
        if self.use_ctx:
            from .ctx_abc import ctx_geometry, select_neighbors, tile_tensor_for_ctx
            from .rbox import min_area_rect as _min_area_rect

            if self.crop_mode in ("mask", "rbox", "rot") and isinstance(tfm, dict) and tfm.get("ok"):
                crop_c, crop_ang = tfm["c"], float(tfm.get("angle", 0.0))
            else:      # aabb / 极小目标回退路径：轴对齐、角度 0
                crop_c, crop_ang = ((x1 + x2) / 2.0, (y1 + y2) / 2.0), 0.0
            obj = _min_area_rect(item["polygon"])
            ctx_out["ctx_tile_pixel_values"] = tile_tensor_for_ctx(image, self.ctx_tile_size)
            ctx_out["ctx_geom"] = ctx_geometry(crop_center=crop_c, crop_size=crop.size,
                                               crop_angle=crop_ang, obj=obj,
                                               tile_size=image.size)
            nbr = self.ctx_neighbors.get(item.get("file_name") or "")
            if nbr:
                b_, s_, m_ = select_neighbors(nbr["boxes"], nbr["scores"], obj, k=self.ctx_k)
                ctx_out["ctx_neighbors"] = b_
                ctx_out["ctx_neighbor_scores"] = s_
                ctx_out["ctx_neighbor_mask"] = m_
            else:
                self.n_ctx_no_nbr += 1
        if no_object:
            target = NO_OBJECT_TOKEN
        else:
            grid = pixel_to_grid(local_polygon, crop.size, GRID_SIZE)
            if self.vertex_count_token:
                grid = grid[:60]
                target = _fmt_target(grid, with_count=True)
            else:
                target = _fmt_target(grid, with_count=False)
        prompt = expand_pixel_placeholder(raw_prompt())
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target + _end_token, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + target_ids)[: self.max_length]
        labels = ([IGNORE_INDEX] * len(prompt_ids) + list(target_ids))[: self.max_length]
        item_out = {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "n_target_tokens": torch.tensor(len(target_ids), dtype=torch.long),
        }
        if image is not None and box is not None:
            if self.use_global:
                from .global_fusion import pos_box_from_box, pos_code_from_box

                item_out["global_pixel_values"] = self.image_processor(
                    image, return_tensors="pt")["pixel_values"][0]
                item_out["pos_code"] = pos_code_from_box(
                    box, image.size,
                    dim=(POS_DIM_ANGLE if getattr(self, "pos_angle", False) else POS_DIM),
                    angle=(0.0 if getattr(self, "pos_angle", False) else None))
                item_out["pos_box"] = pos_box_from_box(box, image.size)
                item_out["crop_angle"] = torch.tensor(0.0)
            if self.use_scale_cond:
                from .scale_edge import object_box_of, scale_code

                if local_polygon:      # 正样本：目标框 = 多边形外接框（瓦片坐标）
                    ob = object_box_of([(x + box[0], y + box[1]) for x, y in local_polygon])
                    item_out["scale_code"] = scale_code(ob, box, dim=self.code_dim)
                else:                  # 负样本没有目标框 → 用全零表示"无目标"
                    item_out["scale_code"] = torch.zeros(self.code_dim)
        return item_out


def parse_domains(spec: str):
    """`root:split:weight,root:split:weight` → [(root, split, weight)]。"""
    domains = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.rsplit(":", 2)
        if len(parts) != 3:
            raise SystemExit(f"域规格格式应为 root:split:weight，收到 {item}")
        domains.append((parts[0], parts[1], float(parts[2])))
    return domains


def build_multi_eval_samples(domains, n_per_domain: int = 10, n_empty_per_domain: int = 10,
                             ratio: float = 1.2, seed: int = 0):
    """多域验收样本：每个域取 n 栋楼（裁剪图 + GT）+ n 个负样本（空瓦片块）。

    用固定种子抽样 → 训练中每 1000 步都评**同一批**样本，曲线才可横向比。
    """
    import random as _random

    from PIL import Image

    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    from whumix import crop_box_for, load_coco, segmentation_to_polygon

    rng = _random.Random(seed)
    samples = []
    for root, split, _weight in domains:
        coco = load_coco(Path(root), split)
        images = {im["id"]: im for im in coco["images"]}
        anns = {}
        for ann in coco["annotations"]:
            poly = segmentation_to_polygon(ann.get("segmentation"))
            if len(poly) >= 3:
                anns.setdefault(ann["image_id"], []).append(poly)
        tiles_with = [i for i in images if anns.get(i)]
        tiles_empty = [i for i in images if not anns.get(i)]
        rng.shuffle(tiles_with)
        rng.shuffle(tiles_empty)
        got = 0
        for image_id in tiles_with:                       # 跨瓦片均匀取，避免同一瓦片重复
            if got >= n_per_domain:
                break
            image_meta = images[image_id]
            path = Path(root) / split / "image" / image_meta["file_name"]
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                continue
            poly = anns[image_id][0]
            x1, y1, x2, y2 = crop_box_for(poly, image.size, ratio)
            x1, y1, x2, y2 = jitter_box((x1, y1, x2, y2), image.size, self.box_shift, self.box_aspect, rng)
            crop = image.crop((x1, y1, x2, y2))
            samples.append({"image": crop, "expect": "building",
                            "polygon": [(x - x1, y - y1) for x, y in poly],
                            "domain": f"{Path(root).name}/{split}"})
            got += 1
        got = 0
        for image_id in tiles_empty:
            if got >= n_empty_per_domain:
                break
            image_meta = images[image_id]
            path = Path(root) / split / "image" / image_meta["file_name"]
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                continue
            side = min(128, image.size[0], image.size[1])
            samples.append({"image": image.crop((0, 0, side, side)), "expect": "no_object",
                            "domain": f"{Path(root).name}/{split}"})
            got += 1
    return samples
