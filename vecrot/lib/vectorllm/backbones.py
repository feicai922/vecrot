"""可插拔骨干：**视觉塔 / projector / LLM 三件都能换**。

为什么单独一个模块，而不是改 `modeling.py`
--------------------------------------------
`modeling.VectorLLMForCausalLM` 与 v1 逐 token 对齐（`tests/test_parity_v1.py` 撑着）。
换 LLM 会牵动 `config.text_config = Qwen3Config(...)` 这一行硬编码，改它等于把
"和 v1 完全一致"这条性质弄丢。所以这里另起一套**通用装配器**：

* 视觉塔：`RadioBackbone`（v1 用的 C-RADIO，保持原样）/ `DinoV3Backbone`（本地 SAT 权重）；
* projector：`mlp`(v1 原样) / `mlp2d` / `conv` / `attn` / `resampler`，见 `PROJECTORS`；
* LLM：任何 HF causal LM，按 `LLM_SPECS` 里的条目装配。

它**不是** v1 的替代品，而是"换骨干实验"的载体：训练/评测脚本可以照旧用
`tools/train_sft.py` 的回路，只要模型对象满足
`forward(pixel_values=..., input_ids=..., labels=...)` 与 `generate(pixel_values=..., input_ids=...)`。

坐标词表
--------
v1 的坐标是 256 个 **special token**（`<x0..127>` / `<y0..127>`，id 151670..151925）。
换 LLM 时这些 token 在目标 tokenizer 里通常不存在，必须
`add_tokens(..., special_tokens=True)` 并 `resize_token_embeddings`，**且扩完要重绑 tied lm_head**
（`resize_token_embeddings` 会新建 `nn.Embedding`，原来的共享关系会静默断掉 —— 这是
`docs/impl_sft_loop.md` 里踩过的坑）。

用法::

    from vectorllm.backbones import build_vectorllm, LLM_SPECS
    bundle = build_vectorllm("smollm2-360m", vision="radio_l", projector="mlp2d", dtype="bfloat16")
    bundle.model, bundle.tokenizer, bundle.prompt("<pixel>")
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_SIZE = 128
PIXEL_TOKEN = "<pixel>"
NO_OBJECT_TOKEN = "<no_object>"
COORD_TOKENS: List[str] = (
    [f"<x{i}>" for i in range(GRID_SIZE)] + [f"<y{i}>" for i in range(GRID_SIZE)]
)
# v1 的 prompt / 目标序列用的是 Qwen 的 chat 标记。换 tokenizer 后这两个必须作为
# **单个 special token** 加进去，否则 `raw_prompt()` 会被切成 "<","|","im","_start"… 一片碎片。
CHAT_TOKENS: List[str] = ["<|im_start|>", "<|im_end|>"]

V1_DIR = Path("/data1/wzh26/vectorllm_v1")
TINY_LLM_DIR = Path("/data1/wzh26/model_zoo/tiny_llms")
DINOV3_REPO = Path("/data1/wzh26/code/dinov3")
DINOV3_WEIGHTS = Path("/data1/wzh26/model_zoo/dinov3")


# --------------------------------------------------------------------------- LLM 规格


@dataclass
class LLMSpec:
    name: str
    path: Path
    family: str            # 决定用哪个 ForCausalLM 类
    hidden_size: int
    vocab_size: int
    n_layers: int
    note: str = ""


def _qwen_cache() -> Path:
    """HF cache 里 Qwen3-0.6B 的 snapshot 目录（`.from_pretrained` 不能直接吃 cache 根目录）。"""
    base = Path("/home/wzh26/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")
    snaps = sorted((base / "snapshots").glob("*"))
    return snaps[-1] if snaps else base


LLM_SPECS: Dict[str, LLMSpec] = {
    "qwen3-0.6b": LLMSpec(
        name="qwen3-0.6b", path=_qwen_cache(),
        family="qwen3", hidden_size=1024, vocab_size=151936, n_layers=28,
        note="VectorLLM 原配；这里只用于对照（权重来自 v1 目录，见 weights_dir）",
    ),
    "smollm2-360m": LLMSpec(
        name="smollm2-360m", path=TINY_LLM_DIR / "SmolLM2-360M", family="llama",
        hidden_size=960, vocab_size=49152, n_layers=32,
        note="LlamaForCausalLM，tied embeddings，1.35B token 预训练",
    ),
    "gemma3-270m": LLMSpec(
        name="gemma3-270m", path=TINY_LLM_DIR / "gemma-3-270m", family="gemma3",
        hidden_size=640, vocab_size=262144, n_layers=18,
        note="262k 词表（embed 168M 参数占大头）+ 1 个 KV 头 + sliding window 512",
    ),
    "mobilellm-125m": LLMSpec(
        name="mobilellm-125m", path=TINY_LLM_DIR / "MobileLLM-125M", family="mobilellm",
        hidden_size=576, vocab_size=32000, n_layers=30,
        note="⚠️ 自带 modeling_mobilellm.py（对 transformers 4.41 写），在本环境需 verify",
    ),
}


def set_llm_root(root: str) -> None:
    """把所有小 LLM 的路径重定位到另一个根目录（gpu 节点上建议 /dev/shm，
    NFS 随机读会让 `from_pretrained` 卡上几分钟）。目录名沿用原来的 basename。"""
    root_path = Path(root)
    for spec in LLM_SPECS.values():
        candidate = root_path / spec.path.name
        if candidate.exists():
            spec.path = candidate


def load_llm(spec: LLMSpec, dtype=torch.bfloat16, device=None):
    """按 family 装 LLM。返回 (model, tokenizer, 警告列表)。"""
    warnings: List[str] = []
    if spec.family == "qwen3":
        from transformers import AutoTokenizer, Qwen3ForCausalLM

        model = Qwen3ForCausalLM.from_pretrained(str(spec.path), torch_dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(str(spec.path))
    elif spec.family == "llama":
        from transformers import AutoTokenizer, LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(str(spec.path), torch_dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(str(spec.path))
    elif spec.family == "gemma3":
        from transformers import AutoTokenizer, Gemma3ForCausalLM

        model = Gemma3ForCausalLM.from_pretrained(str(spec.path), torch_dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(str(spec.path))
    elif spec.family == "mobilellm":
        # ⚠️ 实测：MobileLLM 的 tokenizer_config 里 model_type=mobilellm + tokenizer_class=LlamaTokenizer
        # 会让 transformers 4.57 的 `AutoTokenizer.from_pretrained` 返回一个 **bool**（不是 tokenizer）。
        # 所以这里直接用 LlamaTokenizer 构造 tokenizer.model。
        from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer

        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(spec.path), torch_dtype=dtype, trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "MobileLLM 自带 modeling_mobilellm.py 是针对 transformers 4.41 的私有实现，"
                f"在当前版本装不起来：{type(exc).__name__}: {exc}"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(str(spec.path))
        if not hasattr(tokenizer, "get_vocab"):
            tokenizer = LlamaTokenizer(vocab_file=str(spec.path / "tokenizer.model"), legacy=True)
            warnings.append("mobilellm: AutoTokenizer 返回非 tokenizer 对象，已退回 LlamaTokenizer(tokenizer.model)")
    else:
        raise ValueError(f"unknown family {spec.family}")
    if device is not None:
        model = model.to(device)
    return model.eval(), tokenizer, warnings


# --------------------------------------------------------------------------- tokenizer


def ensure_vector_tokens(tokenizer) -> Dict[str, int]:
    """加 `<pixel>` / 256 个坐标 token / `<no_object>` / 两个 chat 标记，幂等。"""
    wanted = [PIXEL_TOKEN, NO_OBJECT_TOKEN] + COORD_TOKENS + CHAT_TOKENS
    missing = [t for t in wanted if t not in tokenizer.get_vocab()]
    if missing:
        tokenizer.add_tokens(missing, special_tokens=True)
    # Llama/GPT2 系 tokenizer 通常没有 pad token（SmolLM2/MobileLLM 都没有），
    # 而评测侧 processor 会 padding=True → 必须补一个，否则 batch 编码直接报错。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return {t: tokenizer.convert_tokens_to_ids(t) for t in wanted}


def resize_and_tie(model, tokenizer):
    """扩词表并**重绑 tied lm_head**（不重绑 = 静默错）。"""
    old = model.get_input_embeddings().weight.shape[0]
    new = len(tokenizer)
    tied = model.get_output_embeddings().weight is model.get_input_embeddings().weight
    if new > old:
        model.resize_token_embeddings(new)
        if tied:
            model.get_output_embeddings().weight = model.get_input_embeddings().weight
    return old, new, tied


def chat_prompt(tokenizer, question: str) -> str:
    """按目标 tokenizer 的 chat 模板拼 prompt；没有模板就退化成朴素文本。"""
    content = f"{PIXEL_TOKEN}\n{question}"
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001
            pass
    return f"User: {content}\nAssistant:"


def encode_prompt(tokenizer, question: str, n_visual: int = 64):
    """把 prompt 切成 token，并把 `<pixel>` **展开成 n_visual 个占位 id**。

    真实流程里这是 `processor` 干的活（`<pixel>` → 64 个视觉占位 token），
    这里显式做一遍，避免"占位 token 数 ≠ 视觉 token 数"的对齐错误。
    """
    text = chat_prompt(tokenizer, question)
    head, sep, tail = text.partition(PIXEL_TOKEN)
    ids = tokenizer(head, add_special_tokens=False).input_ids
    ids += [tokenizer.convert_tokens_to_ids(PIXEL_TOKEN)] * n_visual
    ids += tokenizer(tail if sep else text, add_special_tokens=False).input_ids
    return torch.tensor([ids], dtype=torch.long)


# --------------------------------------------------------------------------- 视觉塔


class RadioBackbone(nn.Module):
    """v1 的 C-RADIO 视觉塔（原样复用，含 432→128 预处理之后的 8×8 特征）。"""

    def __init__(self, weights_dir: Path = V1_DIR, dtype=torch.bfloat16, taps: Optional[List[int]] = None):
        super().__init__()
        from .hub import build_model

        self.dtype = dtype
        self.taps = taps
        if taps:
            # 多层 taps 只借权重，不整套加载 llm/projector
            from .config import VectorLLMConfig
            from .modeling import VectorLLMForCausalLM

            raw = json.loads((Path(weights_dir) / "config.json").read_text())
            self.radio = VectorLLMForCausalLM(VectorLLMConfig.from_dict(raw)).vision_model
        else:
            self.radio = build_model(weights_dir, dtype=dtype, device="cpu").vision_model

    @property
    def hidden_size(self) -> int:
        return 1024

    def forward(self, pixel_values: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        x = pixel_values.to(self.dtype)
        if not self.taps:
            summary, features = self.radio(x)
            return summary, features
        outs = self.radio.radio_model.forward_intermediates(
            x, indices=self.taps, norm=True, output_fmt="NLC"
        )
        feats = [getattr(o, "features", o).float() for o in outs]
        return None, torch.cat(feats, dim=-1)


class DinoV3Backbone(nn.Module):
    """DINOv3 ViT（SAT-493M 卫星预训练权重，本地）。

    ⚠️ 与原流程的两个口径差异，换塔时必须一起改：
    1. **需要 ImageNet mean/std 归一化**，而 C-RADIO 是"0~1、不归一化"；
    2. 没有 summary / register token，只有 patch token；维度 ViT-L=1024 / ViT-B=768。
    """

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, kind: str = "vitl16_sat", dtype=torch.bfloat16):
        super().__init__()
        sys.path.insert(0, str(DINOV3_REPO))
        from dinov3.hub.backbones import _make_dinov3_vit

        self.kind = kind
        is_l = "vitl" in kind
        cfg = dict(
            img_size=224, patch_size=16, in_chans=3,
            pos_embed_rope_base=100, pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype="fp32",
            embed_dim=1024 if is_l else 768, depth=24 if is_l else 12,
            num_heads=16 if is_l else 12, ffn_ratio=4, qkv_bias=True,
            drop_path_rate=0.0, layerscale_init=1.0e-05, norm_layer="layernormbf16",
            ffn_layer="mlp", ffn_bias=True, proj_bias=True, n_storage_tokens=4,
            mask_k_bias=True,
            untie_global_and_local_cls_norm=kind.endswith("sat"),
            pretrained=False, compact_arch_name="vitl" if is_l else "vitb",
        )
        self.model = _make_dinov3_vit(**cfg)
        weight_file = (
            DINOV3_WEIGHTS / "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
            if kind == "vitl16_sat"
            else DINOV3_WEIGHTS / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        )
        state = torch.load(weight_file, map_location="cpu", weights_only=False)
        model_keys = set(self.model.state_dict().keys())
        state = {k: v for k, v in state.items() if k in model_keys}
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.dtype = dtype
        self.register_buffer("_mean", torch.tensor(self.MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(self.STD).view(1, 3, 1, 1), persistent=False)

    @property
    def hidden_size(self) -> int:
        return 1024 if "vitl" in self.kind else 768

    def forward(self, pixel_values: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        x = pixel_values.float()
        if x.max() > 1.5:          # 传入的是 0~255
            x = x / 255.0
        x = (x - self._mean) / self._std
        out = self.model.forward_features(x.to(self.dtype))
        return None, out["x_norm_patchtokens"]


VISION_BACKBONES = {
    "radio_l": lambda dtype: RadioBackbone(dtype=dtype),
    "radio_l_taps": lambda dtype: RadioBackbone(dtype=dtype, taps=[-1, -6, -11]),
    "dino_l_sat": lambda dtype: DinoV3Backbone("vitl16_sat", dtype=dtype),
    "dino_b": lambda dtype: DinoV3Backbone("vitb16", dtype=dtype),
}


# --------------------------------------------------------------------------- projector


class _DtypeMixin:
    """`modeling.extract_feature` 会读 `self.projector.dtype`（v1 的 ProjectorModel 是
    PreTrainedModel，自带这个属性）；普通 nn.Module 没有，这里补上。"""

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class MLPProjector(_DtypeMixin, nn.Module):
    """v1 原样：Linear→GELU→Linear，逐 token；位置编码在视觉空间里相加。"""

    def __init__(self, in_dim: int, out_dim: int, depth: int = 2, pos: bool = True, n_pos: int = 64):
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(in_dim, out_dim)]
        for _ in range(depth - 1):
            layers += [nn.GELU(), nn.Linear(out_dim, out_dim)]
        self.net = nn.Sequential(*layers)
        self.pos = nn.Embedding(n_pos, in_dim) if pos else None

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if self.pos is not None:
            if feats.shape[1] != self.pos.num_embeddings:
                raise ValueError(
                    f"mlp projector 的位置编码写死 {self.pos.num_embeddings} 个位置，收到 {feats.shape[1]} 个 token；"
                    "要换分辨率/网格大小请用 mlp2d / conv / attn"
                )
            feats = feats + self.pos.weight.unsqueeze(0).to(feats.dtype)
        return self.net(feats)


class MLP2DProjector(_DtypeMixin, nn.Module):
    """**2D 可分解位置编码**（行 + 列各 8 个向量）→ 换分辨率/网格大小时可插值，
    v1 的 `nn.Embedding(64)` 做不到这一点（256² 输入 = 256 个位置就得重新学）。"""

    def __init__(self, in_dim: int, out_dim: int, depth: int = 2):
        super().__init__()
        self.row = nn.Parameter(torch.zeros(16, in_dim))
        self.col = nn.Parameter(torch.zeros(16, in_dim))
        nn.init.normal_(self.row, std=0.02)
        nn.init.normal_(self.col, std=0.02)
        self.mlp = MLPProjector(in_dim, out_dim, depth=depth, pos=False)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        b, n, c = feats.shape
        g = int(round(n ** 0.5))
        grid = feats.view(b, g, g, c)
        pos = (self.row[:g].view(1, g, 1, c) + self.col[:g].view(1, 1, g, c)).to(feats.dtype)
        return self.mlp((grid + pos).view(b, n, c))


class ConvProjector(_DtypeMixin, nn.Module):
    """**空间混合**：投影后过一个 depthwise 3×3 conv（8×8 网格上的局部邻域聚合）。
    动机：顶点误差集中在"边界在哪一格"这类**局部**判断上，逐 token 的 MLP 没有任何
    空间混合能力（每个 token 只看到自己那一格）。代价：~9×9 的卷积，可忽略。"""

    def __init__(self, in_dim: int, out_dim: int, depth: int = 2):
        super().__init__()
        self.pre = MLP2DProjector(in_dim, out_dim, depth=depth)
        self.dw = nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim)
        self.pw = nn.Conv2d(out_dim, out_dim, 1)
        self.act = nn.GELU()

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        b, n, _ = feats.shape
        g = int(round(n ** 0.5))
        x = self.pre(feats).transpose(1, 2).view(b, -1, g, g)   # 维度用 out_dim，不是输入维度
        x = self.act(self.pw(self.dw(x)))
        return x.flatten(2).transpose(1, 2)


class AttnProjector(_DtypeMixin, nn.Module):
    """1 层自注意力的视觉融合（64 token，开销极小）。比 Conv 更"全局"，
    代价是 64×64 的注意力矩阵 —— 仍然可忽略。"""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 8):
        super().__init__()
        self.pre = MLP2DProjector(in_dim, out_dim)
        self.attn = nn.MultiheadAttention(out_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(nn.Linear(out_dim, out_dim * 2), nn.GELU(), nn.Linear(out_dim * 2, out_dim))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.pre(feats)
        h = self.norm(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ffn(self.norm(x))


class ResamplerProjector(_DtypeMixin, nn.Module):
    """**Q-Former 式重采样**：N 个 query 把 64 个视觉 token 压成 8 个。
    动机：LLM 侧成本与视觉 token 数线性相关，压到 8 个能直接把 prefill 砍掉一半以上；
    代价是细节有损，必须实测。"""

    def __init__(self, in_dim: int, out_dim: int, n_query: int = 8, n_heads: int = 8):
        super().__init__()
        self.pre = MLP2DProjector(in_dim, out_dim)
        self.query = nn.Parameter(torch.randn(n_query, out_dim) * 0.02)
        self.attn = nn.MultiheadAttention(out_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(nn.Linear(out_dim, out_dim * 2), nn.GELU(), nn.Linear(out_dim * 2, out_dim))

    @property
    def out_tokens(self) -> int:
        return self.query.shape[0]

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.pre(feats)
        q = self.query.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        h = self.attn(self.norm(q), self.norm(x), self.norm(x), need_weights=False)[0]
        return q + h + self.ffn(self.norm(q + h))


class ResidualConvProjector(_DtypeMixin, nn.Module):
    """**增量式**改进：v1 的 MLP 主干照原样保留，旁边挂一条零初始化的空间混合旁路。

    为什么需要这个（见 docs §9.1）：整体替换 projector = 换掉 LLM 看到的视觉分布，
    834 步的小预算补不回来（实测 0.82 → 0.28）。增量式则保证 **step 0 等价于基线**
    （旁路输出被零初始化、`alpha` 从 0 开始），所以最坏情况是"没提升"，不会"崩掉"。
    """

    def __init__(self, in_dim: int, out_dim: int, depth: int = 2):
        super().__init__()
        self.base = MLPProjector(in_dim, out_dim, depth=depth, pos=False)
        self.dw = nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim)
        self.pw = nn.Conv2d(out_dim, out_dim, 1)
        self.act = nn.GELU()
        # ⚠️ 只把 alpha 初始化为 0（输出 = base + 0×branch ⇒ step0 与基线逐位等价）。
        # **不能同时把 pw 也零初始化**：那样 branch 恒为 0，alpha 的梯度 = <dL/dy, branch> = 0，
        # 这条旁路永远学不动（实测 alpha 训完仍是 0，整条任务等于把基线又跑了一遍）。
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        base = self.base(feats)
        b, n, c = base.shape
        g = int(round(n ** 0.5))
        x = base.transpose(1, 2).view(b, -1, g, g)
        branch = self.act(self.pw(self.dw(x))).flatten(2).transpose(1, 2)
        return base + self.alpha * branch


PROJECTORS = {
    "mlp": MLPProjector,        # v1 原样
    "mlp2d": MLP2DProjector,    # 2D 位置编码
    "conv": ConvProjector,      # + 空间混合
    "attn": AttnProjector,      # + 自注意力
    "resampler": ResamplerProjector,  # 64 → 8 token
    "conv_res": ResidualConvProjector,  # 增量式：v1 MLP + 零初始化空间旁路
}


# --------------------------------------------------------------------------- 训练辅助


def set_trainable_generic(model: "GenericVectorLLM", freeze: str = "vision") -> Dict[str, int]:
    """freeze: none | vision | vision_llm（与 tools/train_sft.py 的语义一致）。"""
    for param in model.parameters():
        param.requires_grad = True
    if freeze in ("vision", "vision_llm"):
        for param in model.vision.parameters():
            param.requires_grad = False
    if freeze == "vision_llm":
        for param in model.llm.parameters():
            param.requires_grad = False
    stats = {"trainable": 0, "frozen": 0}
    for param in model.parameters():
        stats["trainable" if param.requires_grad else "frozen"] += param.numel()
    return stats


def build_optimizer_generic(model: "GenericVectorLLM", lr: float, weight_decay: float = 0.0,
                            projector_lr_mult: float = 10.0, include_llm: bool = True):
    """projector 给更大学习率（新模块要更快跟上），LLM 用基准 lr。

    `include_llm=False` 用于"connector 先对齐"的阶段：**不能靠 requires_grad=False 来冻结**，
    因为 DDP 是在优化器之前构造的、它已经把 LLM 参数登记进 bucket，一旦这些参数不产生梯度
    就会报 "Expected to have finished reduction in the prior iteration"（实测踩过）。
    正确做法是让梯度照常算、只是暂时不把 LLM 参数放进优化器，到点再 `add_param_group`。
    """
    groups = [
        {"params": [p for p in model.projector.parameters() if p.requires_grad], "lr": lr * projector_lr_mult},
    ]
    if include_llm:
        groups.append({"params": [p for p in model.llm.parameters() if p.requires_grad], "lr": lr})
    groups = [g for g in groups if g["params"]]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))


# --------------------------------------------------------------------------- 装配


@dataclass
class Bundle:
    model: "GenericVectorLLM"
    tokenizer: object
    vision_name: str
    llm_name: str
    projector_name: str
    info: Dict = field(default_factory=dict)


class GenericVectorLLM(nn.Module):
    """视觉塔 + projector + 任意 HF causal LM。接口与 `VectorLLMForCausalLM` 对齐。"""

    def __init__(self, llm, vision, projector, pixel_id: int, chat_style: str = "hf"):
        super().__init__()
        self.llm = llm
        self.vision = vision
        self.projector = projector
        self.pixel_id = pixel_id
        self.chat_style = chat_style

    # --- 与 v1 模型对齐的属性/接口（train_sft.py 等回路直接复用）
    @property
    def device(self):
        """PreTrainedModel 有 `.device`，普通 nn.Module 没有；
        `run_acceptance` 里 `v.to(model.device)` 会直接 AttributeError（实测崩过一次）。"""
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def language_model(self):
        return self.llm

    @property
    def vision_model(self):
        return self.vision

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_input_embeddings(self, value):
        return self.llm.set_input_embeddings(value)

    def set_output_embeddings(self, value):
        return self.llm.set_output_embeddings(value)

    def resize_token_embeddings(self, n):
        return self.llm.resize_token_embeddings(n)

    # --- 视觉 → LLM 空间
    def encode_visual(self, pixel_values: torch.Tensor) -> torch.Tensor:
        _, features = self.vision(pixel_values.to(next(self.vision.parameters()).dtype))
        return self.projector(features.to(self.projector_dtype))

    @property
    def projector_dtype(self):
        return next(self.projector.parameters()).dtype

    def _merge(self, input_ids, visual):
        embeds = self.llm.get_input_embeddings()(input_ids)
        mask = input_ids == self.pixel_id
        if mask.sum() != visual.shape[0] * visual.shape[1]:
            raise ValueError(
                f"视觉占位 token 数 {int(mask.sum())} ≠ 投影后 token 数 {visual.shape[0] * visual.shape[1]}"
            )
        embeds = embeds.clone()
        embeds[mask] = visual.reshape(-1, visual.shape[-1]).to(embeds.dtype)
        return embeds

    def forward(self, input_ids=None, pixel_values=None, labels=None, attention_mask=None, **kw):
        if pixel_values is not None:
            visual = self.encode_visual(pixel_values)
            input_ids = input_ids.to(visual.device)
            inputs_embeds = self._merge(input_ids, visual)
            labels = labels.clone() if labels is not None else None
            if labels is not None:
                labels[input_ids == self.pixel_id] = -100
            out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, **kw)
        else:
            out = self.llm(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kw)
        return out

    @torch.no_grad()
    def generate(self, input_ids=None, pixel_values=None, attention_mask=None, **kw):
        visual = self.encode_visual(pixel_values)
        input_ids = input_ids.to(visual.device)
        embeds = self._merge(input_ids, visual)
        return self.llm.generate(
            inputs_embeds=embeds,
            attention_mask=attention_mask.to(embeds.device) if attention_mask is not None else None,
            **kw,
        )


def build_vectorllm(
    llm_name: str,
    vision: str = "radio_l",
    projector: str = "mlp",
    dtype=torch.bfloat16,
    device=None,
    projector_kwargs: Optional[Dict] = None,
) -> Bundle:
    spec = LLM_SPECS[llm_name]
    llm, tokenizer, warns = load_llm(spec, dtype=dtype)
    ids = ensure_vector_tokens(tokenizer)
    old, new, tied = resize_and_tie(llm, tokenizer)

    vision_module = VISION_BACKBONES[vision](dtype)
    proj_cls = PROJECTORS[projector]
    proj = proj_cls(vision_module.hidden_size, spec.hidden_size, **(projector_kwargs or {}))

    model = GenericVectorLLM(llm, vision_module, proj, pixel_id=ids[PIXEL_TOKEN])
    model = model.to(dtype=dtype)
    if device is not None:
        model = model.to(device)
    info = {
        "llm": spec.name,
        "vision": vision,
        "projector": projector,
        "vocab_old": old,
        "vocab_new": new,
        "tied_before": bool(tied),
        "params_total": sum(p.numel() for p in model.parameters()),
        "params_llm": sum(p.numel() for p in llm.parameters()),
        "params_vision": sum(p.numel() for p in vision_module.parameters()),
        "params_projector": sum(p.numel() for p in proj.parameters()),
        "warnings": warns,
    }
    return Bundle(model, tokenizer, vision, spec.name, projector, info)
