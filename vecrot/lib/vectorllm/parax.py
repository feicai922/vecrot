"""把 ParaX（ICML'26, "Parameters as Experts"）接到我们的 C-RADIO 视觉塔上。

ParaX 是什么（一句话）
----------------------
不是推理加速框架，而是**视觉 backbone 的 PEFT 适配器**：每个 block 后面挂一个小模块，
用整幅图的 token 均值做路由（router MLP），从共享的"专家中心"里动态聚合出一组
**低秩权重**（projA_g / projB_g），做输入相关的低秩残差更新；同一 stage 内多层共享专家中心。

对我们的意义
------------
* 我们的视觉塔（C-RADIO v2.5-L，320M）**在全部实验里都是冻结的**（全参微调实测更差：
  0.8283 vs 冻结 0.8317）。ParaX 提供了一个"少参数、又能改特征"的中间档；
* 它作用在**视觉特征**上，而我们的实测瓶颈正是"边界精度"（§3），
  并且跨域（WHU→贵州）退化严重（0.7882→0.6529）—— PEFT 适配器是标准的域适配手段；
* ⚠️ 它不是效率方法：推理时每层要现算 B×rank×dim 的动态权重，**比 LoRA 更慢**，
  也不会减少参数量。所以它只能"换精度"，不能进 P2/P3 那条压缩路线。

实现方式
--------
用 `register_forward_hook` 挂在 RADIO 的每个 transformer block 上（不改 radio 源码）：
block 输出 `[cls+registers, patch_tokens]`，只对 patch tokens 过适配器，再拼回去。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum

PARAX_REPO = Path("/data1/wzh26/code/ParaX")


def _load_parax():
    """优先复用官方实现（保证与论文一致），失败再回退到下面的等价内置实现。"""
    if str(PARAX_REPO) not in sys.path:
        sys.path.insert(0, str(PARAX_REPO))
    try:
        from models.parax import ParaXAdapter, parax_parameter_dict  # type: ignore

        return ParaXAdapter, parax_parameter_dict, "official"
    except Exception:  # noqa: BLE001
        return _BuiltinAdapter, _builtin_param_dict, "builtin"


# --------------------------------------------------------------------------- 内置实现
# 与官方 models/parax.py 的 token-only 路径等价（不含 conv 专家），便于在没有 einops/官方仓库时使用。


def _builtin_linear_param(in_features: int, out_features: int, num_layers: int) -> torch.Tensor:
    params = []
    for _ in range(num_layers):
        p = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.trunc_normal_(p, std=0.02)
        params.append(p)
    return torch.stack(params, dim=0)


def _builtin_param_dict(dim: int, rank: int, num_layers: int, **_):
    return {
        "projA_params": _builtin_linear_param(dim, rank, num_layers),
        "projB_params": _builtin_linear_param(rank, dim, num_layers),
    }


class _BuiltinAdapter(nn.Module):
    def __init__(self, dim: int, rank: int, num_params: int, ls_init_value: float = 1.0,
                 rs_init_value: float = 1.0, router_hidden: int = 16, **_):
        super().__init__()
        self.dim, self.rank, self.num_params = dim, rank, num_params
        self.num_groups = 2
        self.norm = nn.LayerNorm(dim)
        self.router = nn.Sequential(nn.Linear(dim, router_hidden), nn.GELU(),
                                    nn.Linear(router_hidden, num_params * self.num_groups))
        self.proj = nn.Linear(rank, rank)
        self.norm_gamma = nn.Parameter(torch.ones(dim) * 1e-5)
        self.layer_scale = nn.Parameter(torch.ones(dim) * ls_init_value)
        self.layer_bias = nn.Parameter(torch.zeros(dim))
        self.res_scale = nn.Parameter(torch.ones(dim) * rs_init_value)
        self.res_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, params) -> torch.Tensor:
        gates = self.router(x.mean(dim=1, keepdim=True))          # B,1,2*G
        g = torch.chunk(gates, chunks=self.num_groups, dim=-1)
        attn_a = g[0].softmax(dim=-1)
        attn_b = g[1].softmax(dim=-1)
        proj_a = einsum(attn_a, params["projA_params"], "b l g, g c2 c1 -> b c2 c1")
        proj_b = einsum(attn_b, params["projB_params"], "b l g, g c2 c1 -> b c2 c1")
        h = self.norm(x)
        h = einsum(h, proj_a, "b l c1, b c2 c1 -> b l c2")
        h = self.proj(h)
        h = einsum(h, proj_b, "b l c2, b c1 c2 -> b l c1")
        h = F.gelu(h) * self.norm_gamma
        return x * self.res_scale + self.res_bias + h * self.layer_scale + self.layer_bias


class ParaXOnProjector(nn.Module):
    """把 ParaX 挂在 **projector 后面**（视觉 token → LLM 空间那一步）。

    与挂在视觉塔上的区别：
    * 视觉塔版（24 个 block 共享专家中心）= 改**特征**；这里 = 改**投影结果**，
      等价于给"特征→LLM 空间"的映射加一个**输入相关的低秩残差**；
    * 只有一层，所以"专家中心"里的 slot 不是给不同层用的，而是给**不同输入**路由用的
      （`n_experts` 个，按整幅图的 token 均值路由）；
    * 参数量很小：rank=64、4 个 expert → ≈0.6M，是 projector 本身的 1/3。

    ⚠️ `ls_init_value=0` 时 step0 与基线**逐位等价**，且 `layer_scale.grad ≠ 0`（已验证），
    这是和 §9.3"零初始化把旁路做成死分支"那次教训对应的正确写法。
    """

    def __init__(self, projector: nn.Module, out_dim: int, rank: int = 64, n_experts: int = 4,
                 router_hidden: int = 16, ls_init_value: float = 0.0):
        super().__init__()
        adapter_cls, param_fn, source = _load_parax()
        self.source = source
        self.projector = projector
        self.out_dim = out_dim
        self.n_experts = n_experts
        self.rank = rank
        self.expert = nn.ParameterDict(param_fn(dim=out_dim, rank=rank, num_layers=n_experts,
                                                parax_kernel_sizes=None, enable_conv=False))
        self.adapter = adapter_cls(dim=out_dim, rank=rank, num_params=n_experts,
                                   router_hidden=router_hidden, ls_init_value=ls_init_value,
                                   rs_init_value=1.0, parax_kernel_sizes=None, enable_conv=False)
        # layer_scale 零初始化时输出恒等；但把 norm_gamma 稍微抬一点，让梯度更快活过来
        if ls_init_value == 0.0 and hasattr(self.adapter, "norm_gamma"):
            with torch.no_grad():
                self.adapter.norm_gamma.fill_(1e-3)

    @property
    def dtype(self):        # modeling.extract_feature 会读 projector.dtype
        return next(self.parameters()).dtype

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        y = self.projector(feats)
        return self.adapter(y, self.expert)


# --------------------------------------------------------------------------- 接线


class ParaXOnVision(nn.Module):
    """把 ParaX 适配器挂到 RADIO 视觉塔的每个 block 上（stage 内共享一套专家中心）。"""

    def __init__(self, blocks: List[nn.Module], dim: int, rank: int = 128,
                 router_hidden: int = 16, n_prefix: int = 9, ls_init_value: float = 1.0,
                 rs_init_value: float = 1.0):
        super().__init__()
        adapter_cls, param_fn, source = _load_parax()
        self.source = source
        self.n_prefix = n_prefix
        self.dim = dim
        self.blocks = blocks
        self.expert = nn.ParameterDict(param_fn(dim=dim, rank=rank, num_layers=len(blocks),
                                                parax_kernel_sizes=None, enable_conv=False))
        self.adapters = nn.ModuleList([
            adapter_cls(dim=dim, rank=rank, num_params=len(blocks), router_hidden=router_hidden,
                        ls_init_value=ls_init_value, rs_init_value=rs_init_value,
                        parax_kernel_sizes=None, enable_conv=False)
            for _ in blocks
        ])
        self._handles = []
        for i, blk in enumerate(blocks):
            self._handles.append(blk.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, index: int):
        adapter = self.adapters[index]

        def hook(module, inputs, output):
            pre, tok = output[:, : self.n_prefix], output[:, self.n_prefix:]
            return torch.cat([pre, adapter(tok, self.expert)], dim=1)

        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def detect_n_prefix(vision_model, img_size: int = 128, patch: int = 16) -> int:
    """实测 RADIO 的 block 输出是 `8 + 64 = 72` 个 token（1 cls + 7 registers + 8×8 patch），
    不是配置里 `num_cls_register_tokens=9` 暗示的 9 —— 猜错会把 1 个 patch token 当 prefix 放过适配器。"""
    try:
        ref = next(vision_model.parameters())
        dummy = torch.zeros(1, 3, img_size, img_size, device=ref.device, dtype=ref.dtype)
        captured = []
        handle = vision_model.radio_model.model.blocks[0].register_forward_hook(
            lambda mod, inp, out: captured.append(out.shape[1]))
        with torch.no_grad():
            vision_model(dummy)
        handle.remove()
        n_patches = (img_size // patch) ** 2
        n_prefix = int(captured[0]) - n_patches
        if 0 <= n_prefix < 32:
            return n_prefix
    except Exception:  # noqa: BLE001
        pass
    return 8


def attach_parax(vision_model, rank: int = 128, router_hidden: int = 16,
                 n_prefix: Optional[int] = None, ls_init_value: float = 1.0,
                 rs_init_value: float = 1.0) -> ParaXOnVision:
    """在 `VectorLLMForCausalLM`（或它的 vision_model）上挂 ParaX。

    返回 `ParaXOnVision` 模块（已作为子模块挂在 `vision_model` 上，参数会随模型移动/保存）。
    """
    radio = getattr(vision_model, "radio_model", vision_model)
    vit = radio.model
    blocks = list(vit.blocks)
    dim = int(vit.embed_dim)
    if n_prefix is None:
        n_prefix = detect_n_prefix(vision_model)
    module = ParaXOnVision(blocks, dim=dim, rank=rank, router_hidden=router_hidden, n_prefix=n_prefix,
                           ls_init_value=ls_init_value, rs_init_value=rs_init_value)
    if hasattr(vision_model, "parax"):
        vision_model.parax.remove()
    vision_model.add_module("parax", module)
    # 视觉塔通常已经在 GPU/bf16 上，新模块默认建在 CPU/fp32 → 必须跟着搬过去
    ref = next(vision_model.parameters())
    module = module.to(device=ref.device, dtype=ref.dtype)
    vision_model.parax = module
    return module


def attach_parax_projector(model, rank: int = 64, n_experts: int = 4,
                           ls_init_value: float = 0.0) -> "ParaXOnProjector":
    """把 ParaX 挂到 `model.projector` 后面（原地替换 projector）。

    `model` 是 `VectorLLMForCausalLM`（或任何有 `.projector` 与 `.config` 的对象）。
    """
    base = model.projector
    # ⚠️ GenericVectorLLM（换骨干那条路径）**没有 .config**：out_dim 从 projector 的最后一层推，
    #    或者从 LLM 的 hidden_size 拿。两处都拿不到才报错。
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "hidden_size", None):
        out_dim = int(cfg.hidden_size)
    else:
        llm = getattr(model, "language_model", None)
        out_dim = int(getattr(getattr(llm, "config", None), "hidden_size", 0) or 0)
        if out_dim == 0:
            raise AttributeError("attach_parax_projector: 拿不到 out_dim（model 和 language_model 都没有 hidden_size）")
    module = ParaXOnProjector(base, out_dim=out_dim, rank=rank, n_experts=n_experts,
                              ls_init_value=ls_init_value)
    ref = next(base.parameters())
    model.projector = module.to(device=ref.device, dtype=ref.dtype)
    return model.projector


def parax_summary(module: ParaXOnVision) -> dict:
    n_tr = sum(p.numel() for p in module.parameters())
    return {"adapter_source": module.source, "n_blocks": len(module.blocks), "dim": module.dim,
            "trainable_params_M": round(n_tr / 1e6, 2)}
