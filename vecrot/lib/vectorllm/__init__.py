"""VectorLLM v2 —— 遥感建筑物矢量轮廓提取（C-RADIO + Qwen3 自回归多边形）。

组织原则：**代码在仓库里，权重只当数据读**。
v1 的做法是把 .py 塞进权重目录、靠 `目录名==包名` + `AutoModel` 注册来加载，
目录改名或升级 transformers 就会崩；v2 不再依赖这些机制。

常用入口::

    from vectorllm import VectorLLMPredictor          # 加载 + 推理
    from vectorllm.geometry import parse_polygon      # 几何工具（不依赖 torch）
    from vectorllm.prompts import prompt_for          # prompt

    pred = VectorLLMPredictor("/data1/wzh26/vectorllm_v1")   # 只读权重目录
    print(pred.predict("crop.png")["polygon"])

注意：本文件用 **惰性导入**（PEP 562），所以 `import vectorllm.geometry` 之类的纯几何用途
不会顺带把 torch / transformers 拖进来 —— `tools/` 下的分析脚本依赖这个性质。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "2.0.0"

_LAZY = {
    # config
    "VectorLLMConfig": ("vectorllm.config", "VectorLLMConfig"),
    "ProjectorConfig": ("vectorllm.config", "ProjectorConfig"),
    # 模型
    "VectorLLMForCausalLM": ("vectorllm.modeling", "VectorLLMForCausalLM"),
    "ProjectorModel": ("vectorllm.modeling", "ProjectorModel"),
    # processor
    "VectorLLMImageProcessor": ("vectorllm.image_processor", "VectorLLMImageProcessor"),
    "VectorLLMProcessor": ("vectorllm.processor", "VectorLLMProcessor"),
    # 加载器
    "VectorLLMPredictor": ("vectorllm.hub", "VectorLLMPredictor"),
    "GenerationSettings": ("vectorllm.hub", "GenerationSettings"),
    "build_model": ("vectorllm.hub", "build_model"),
    "build_processor": ("vectorllm.hub", "build_processor"),
    "load": ("vectorllm.hub", "load"),
    # 几何
    "GRID_SIZE": ("vectorllm.geometry", "GRID_SIZE"),
    "parse_polygon": ("vectorllm.geometry", "parse_polygon"),
    "grid_to_pixel": ("vectorllm.geometry", "grid_to_pixel"),
    "pixel_to_grid": ("vectorllm.geometry", "pixel_to_grid"),
    "clip_polygon": ("vectorllm.geometry", "clip_polygon"),
    "rasterize": ("vectorllm.geometry", "rasterize"),
    "rasterize_masks": ("vectorllm.geometry", "rasterize_masks"),
    "iou_matrix_masks": ("vectorllm.geometry", "iou_matrix_masks"),
    "iou_matrix": ("vectorllm.geometry", "iou_matrix"),
    "match_greedy": ("vectorllm.geometry", "match_greedy"),
    "polygon_iou": ("vectorllm.geometry", "polygon_iou"),
    # 训练数据
    "AbstainSFTDataset": ("vectorllm.data", "AbstainSFTDataset"),
    "collate_fn": ("vectorllm.data", "collate_fn"),
    # 密集实例分离（后处理）
    "resolve_overlaps": ("vectorllm.separate", "resolve_overlaps"),
    "snap_shared_edges": ("vectorllm.separate", "snap_shared_edges"),
    "mask_to_polygon": ("vectorllm.separate", "mask_to_polygon"),
    "pairwise_overlap_area": ("vectorllm.separate", "pairwise_overlap_area"),
    # 弃权
    "NO_OBJECT_TOKEN": ("vectorllm.abstain", "NO_OBJECT_TOKEN"),
    "polygon_signals": ("vectorllm.abstain", "polygon_signals"),
    "consistency_from_samples": ("vectorllm.abstain", "consistency_from_samples"),
    "abstain_score": ("vectorllm.abstain", "abstain_score"),
    "auroc": ("vectorllm.abstain", "auroc"),
    "best_threshold": ("vectorllm.abstain", "best_threshold"),
    "parse_output": ("vectorllm.abstain", "parse_output"),
    # prompt
    "prompt_for": ("vectorllm.prompts", "prompt_for"),
    "raw_prompt": ("vectorllm.prompts", "raw_prompt"),
    "BUILDING_RAW_PROMPT": ("vectorllm.prompts", "BUILDING_RAW_PROMPT"),
    "OBJECT_RAW_PROMPT": ("vectorllm.prompts", "OBJECT_RAW_PROMPT"),
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        module_name, attribute = _LAZY[name]
        value = getattr(importlib.import_module(module_name), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


if TYPE_CHECKING:  # pragma: no cover - 仅供 IDE/类型检查
    from .config import ProjectorConfig, VectorLLMConfig
    from .geometry import (
        GRID_SIZE,
        clip_polygon,
        grid_to_pixel,
        iou_matrix,
        match_greedy,
        parse_polygon,
        pixel_to_grid,
        polygon_iou,
        rasterize,
        rasterize_masks,
        iou_matrix_masks,
    )
    from .hub import GenerationSettings, VectorLLMPredictor, build_model, build_processor, load
    from .data import AbstainSFTDataset, collate_fn  # noqa: F401
    from .abstain import (  # noqa: F401
        NO_OBJECT_TOKEN,
        abstain_score,
        auroc,
        best_threshold,
        consistency_from_samples,
        parse_output,
        polygon_signals,
    )
    from .image_processor import VectorLLMImageProcessor
    from .modeling import ProjectorModel, VectorLLMForCausalLM
    from .processor import VectorLLMProcessor
    from .prompts import BUILDING_RAW_PROMPT, OBJECT_RAW_PROMPT, prompt_for, raw_prompt
    from .separate import (  # noqa: F401
        mask_to_polygon,
        pairwise_overlap_area,
        resolve_overlaps,
        snap_shared_edges,
    )
