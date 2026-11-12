"""从 HF 格式权重目录加载 VectorLLM，并提供一个开箱可用的预测器。

设计取舍（相对 v1）：

- **v1 靠"目录名 == 包名 + AutoModel 注册"来加载**：一旦目录改名（v1 里 README 写
  `vectorllm_hf_0407`、实际目录叫 `vectorllm_v1`）、或换成 transformers 5.x，就直接崩。
  v2 改成"**代码在仓库里、权重只当数据读**"：权重目录可以只放 config/safetensors/tokenizer，
  不需要把 .py 塞进权重目录，也不依赖 `AutoModel`/`auto_map`/`trust_remote_code`。
- `generate()` 走 `inputs_embeds`，因此 HF 返回的 `sequences` **只含新生成的 token**，
  不要再按 prompt 长度切片（v1 的 `test_hf.py` 就踩了这个坑，靠 `numel()==0` 兜底才正确）。

用法::

    from vectorllm.hub import VectorLLMPredictor
    pred = VectorLLMPredictor("/data1/wzh26/vectorllm_v1")
    out = pred.predict("crop.png")
    out["polygon"]        # 裁剪图像素坐标
    out["grid_polygon"]   # 0..127 网格点
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoTokenizer, GenerationConfig, StoppingCriteria, StoppingCriteriaList

from .config import VectorLLMConfig
from .geometry import GRID_SIZE, clip_polygon, grid_to_pixel, parse_polygon
from .image_processor import VectorLLMImageProcessor
from .modeling import VectorLLMForCausalLM
from .processor import VectorLLMProcessor, register_for_auto_classes
from .prompts import DEFAULT_MAX_NEW_TOKENS, STOP_WORDS, prompt_for


class StopWordStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_word: str):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        text = self.tokenizer.decode(input_ids[0]).replace("\r", "").replace("\n", "")
        return text[-self.length :] == self.stop_word


def get_stop_criteria(tokenizer, stop_words: Sequence[str] = STOP_WORDS) -> StoppingCriteriaList:
    return StoppingCriteriaList([StopWordStoppingCriteria(tokenizer, w) for w in stop_words])


def build_config(weights_dir: Path) -> VectorLLMConfig:
    raw = json.loads((Path(weights_dir) / "config.json").read_text(encoding="utf-8"))
    return VectorLLMConfig.from_dict(raw)


def load_state_dict(weights_dir: Path) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    candidates = sorted(Path(weights_dir).glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"no *.safetensors under {weights_dir}")
    state: Dict[str, torch.Tensor] = {}
    for path in candidates:
        state.update(load_file(str(path)))
    return state


def build_model(weights_dir, dtype=torch.bfloat16, device: Optional[str] = None,
                parax_adapter: Optional[str] = None) -> VectorLLMForCausalLM:
    """`parax_adapter`：ParaX 区域适配器目录（内含 `adapter.pt` + `meta.json`）。

    用法（按区域切换，已在贵州数据上实测 pass@0.75 +7.1pp，见 docs §12.9）::

        base = build_model("/data1/wzh26/vectorllm_v1", parax_adapter="work_dirs/parax/mix_gzda")
    """
    weights_dir = Path(weights_dir)
    config = build_config(weights_dir)
    model = VectorLLMForCausalLM(config)
    model = model.to(dtype=dtype)

    # config.json 顶层 tie_word_embeddings=False、内层 llm_config=True；checkpoint 里没有
    # lm_head.weight（与 embed_tokens 共用一个张量），所以先手动绑定再加载。
    if config.llm_config.get("tie_word_embeddings", True):
        model.language_model.lm_head.weight = model.language_model.model.embed_tokens.weight

    if parax_adapter:
        # 先把适配器挂上（会往 vision_model 里注册 parax 子模块），再加载主权重
        import json as _json
        from .parax import attach_parax

        meta_path = Path(parax_adapter) / "meta.json"
        meta = _json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        rank = int(meta.get("rank", 128))
        if meta.get("target") == "projector":
            from .parax import attach_parax_projector

            attach_parax_projector(model, rank=rank, n_experts=int(meta.get("n_experts", 4)))
            # 包了一层之后 base projector 的 key 变成 projector.projector.model.*，与权重文件不一致
            _remap_projector_keys = True
        else:
            attach_parax(model.vision_model, rank=rank)

    state = load_state_dict(weights_dir)
    if locals().get("_remap_projector_keys"):
        state = {(f"projector.projector.{k[len('projector.'):]}" if k.startswith("projector.model.") else k): v
                 for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {"language_model.lm_head.weight"}
    # ParaX 是后挂的，自己的权重从 adapter.pt 单独载入 → 主权重里缺这些 key 是正常的
    real_missing = [k for k in missing if k not in allowed_missing
                    and ".parax." not in k
                    and not k.startswith("projector.adapter.")
                    and not k.startswith("projector.expert.")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"state dict mismatch: missing={real_missing[:5]} unexpected={unexpected[:5]}"
        )

    if parax_adapter:
        adapter_pt = Path(parax_adapter) / "adapter.pt"
        adapter_state = torch.load(str(adapter_pt), map_location="cpu", weights_only=True)
        own = model.state_dict()
        loaded = {k: v for k, v in adapter_state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(loaded, strict=False)
        n = sum(v.numel() for v in loaded.values())
        print(f"[hub] ParaX 适配器已加载：{len(loaded)} 个张量 / {n/1e6:.2f}M 参数（来自 {parax_adapter}）",
              flush=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model


def export_parax_adapter(checkpoint_dir, out_dir, rank: int = 128, note: str = "") -> Path:
    """把训练好的 ParaX 适配器（约 15MB）单独导出成一个可分发目录。

    `checkpoint_dir` 可以是训练产出的 `<job>_final`（只含 parax.* 张量），也可以是完整权重目录。
    """
    import json as _json

    src = Path(checkpoint_dir)
    pt = src / "trainable.pt"
    state = torch.load(str(pt if pt.exists() else src / "model.safetensors"),
                       map_location="cpu", weights_only=True)
    adapter = {k: v for k, v in state.items()
               if k.startswith("parax.") or ".parax." in k
               or k.startswith("projector.adapter.") or k.startswith("projector.expert.")}
    if not adapter:
        raise RuntimeError(f"{checkpoint_dir} 里没有 parax.* 张量，不是 ParaX checkpoint？")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(adapter, str(out / "adapter.pt"))
    target = "projector" if any(k.startswith("projector.adapter.") or k.startswith("projector.expert.")
                                for k in adapter) else "vision"
    n_experts = None
    if target == "projector":
        for k, v in adapter.items():
            if k.endswith("expert.projA_params"):
                n_experts = int(v.shape[0])
    (out / "meta.json").write_text(_json.dumps({
        "rank": rank,
        "target": target,
        "n_experts": n_experts,
        "n_tensors": len(adapter),
        "params_M": round(sum(v.numel() for v in adapter.values()) / 1e6, 2),
        "base": "/data1/wzh26/vectorllm_v1",
        "source_checkpoint": str(src),
        "note": note,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[hub] 适配器导出 → {out}（{sum(v.numel() for v in adapter.values())/1e6:.2f}M 参数，"
          f"{(out/'adapter.pt').stat().st_size/1e6:.1f}MB）")
    return out


def build_processor(weights_dir) -> VectorLLMProcessor:
    """构造 processor。

    ⚠️ `ProcessorMixin.__init__` 会用 `image_processor_class` 这个**字符串**去 HF 的
    `IMAGE_PROCESSOR_MAPPING` 里反查类（`get_possibly_dynamic_module`），所以即使我们直接
    传入实例，也必须先把自定义类注册进去，否则报
    `ValueError: Could not find module VectorLLMImageProcessor in transformers`。
    `register_for_auto_classes()` 是幂等的，且对 4.x/5.x 都安全。
    """
    register_for_auto_classes()
    weights_dir = Path(weights_dir)
    preproc = json.loads((weights_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
    preproc.pop("auto_map", None)
    preproc.pop("processor_class", None)
    preproc.pop("image_processor_type", None)
    image_processor = VectorLLMImageProcessor(**preproc)
    tokenizer = AutoTokenizer.from_pretrained(str(weights_dir))
    return VectorLLMProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        chat_template=getattr(tokenizer, "chat_template", None),
    )


class VertexCountProcessor:
    """按模型自己预测的 `<nvK>` 约束解码：满 2K 个坐标 token 强制结束，不足则禁止结束。

    为什么需要：自回归生成的顶点数误差（pred 中位 4 vs GT 中位 6~15）是当前高 IoU 区间的主要损失来源，
    而"什么时候停"本身没有显式监督。把 `<nvK>` 放在序列开头后，K 就变成了可评测、可约束的量。
    模型没学过该 token（老权重）时自动退化成空处理器。
    """

    def __init__(self, tokenizer, coord_ids, end_ids):
        self.nv = {}
        for k in range(3, 65):
            i = tokenizer.convert_tokens_to_ids(f"<nv{k}>")
            if isinstance(i, int) and i >= 0:
                self.nv[i] = k
        self.nv_ids = sorted(self.nv)
        self.coord_ids = sorted(set(int(v) for v in coord_ids))
        self.end_ids = [int(v) for v in end_ids]
        self.active = bool(self.nv) and bool(self.coord_ids) and bool(self.end_ids)

    def __call__(self, input_ids, scores):
        import torch

        if not self.active:
            return scores
        nv_t = torch.tensor(self.nv_ids, dtype=input_ids.dtype, device=input_ids.device)
        co_t = torch.tensor(self.coord_ids, dtype=input_ids.dtype, device=input_ids.device)
        for row in range(input_ids.shape[0]):
            seq = input_ids[row]
            if seq.numel() == 0:
                # 第一步**不强制**：让模型自己决定要不要以 `<nvK>` 打头。
                # 早期版本在这里强制只允许 `<nv*>`/结束，实测当 `<nv*>` 概率质量还低（16%）时，
                # 强制会让模型进入分布外状态、直接输出 `<|im_end|>`，整批变成空多边形（IoU 0.0003）。
                # 正确做法是"模型选了 nv 才约束"，这样即使 token 没学透也不会把生成能力弄坏。
                continue
            k = next((self.nv[int(t)] for t in seq.tolist() if int(t) in self.nv), None)
            if k is None:
                continue                     # 序列里没有 <nvK>（老权重/模型没选它）→ 该行不约束
            emitted = int(torch.isin(seq[1:], co_t).sum().item())
            if emitted < 2 * k:
                scores[row, self.end_ids] = float("-inf")
            else:
                scores[row, :] = float("-inf")
                scores[row, self.end_ids[0]] = 0.0
        return scores


@dataclass
class GenerationSettings:
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    do_sample: bool = False
    temperature: float = 0.0
    top_k: int = 1
    use_cache: bool = True
    enforce_vertex_count: bool = True      # checkpoint 含 `<nv*>` 时按预测的 K 约束解码


@dataclass
class VectorLLMPredictor:
    """权重目录 → 可直接调用的推理器。"""

    weights_dir: str
    dtype: torch.dtype = torch.bfloat16
    device: Optional[str] = None
    parax_adapter: Optional[str] = None
    model: Optional[VectorLLMForCausalLM] = field(default=None, repr=False)
    processor: Optional[VectorLLMProcessor] = field(default=None, repr=False)

    def __post_init__(self):
        self.weights_dir = str(self.weights_dir)
        if self.model is None:
            self.model = build_model(self.weights_dir, dtype=self.dtype, device=self.device,
                                     parax_adapter=self.parax_adapter)
        if self.processor is None:
            self.processor = build_processor(self.weights_dir)

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    # ------------------------------------------------------------------ 推理
    @torch.inference_mode()
    def generate_text(
        self,
        image,
        target: str = "building",
        prompt: Optional[str] = None,
        settings: Optional[GenerationSettings] = None,
    ) -> str:
        settings = settings or GenerationSettings()
        prompt = prompt or prompt_for(target)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = {
            k: (v.to(self.model.device) if torch.is_tensor(v) else v) for k, v in inputs.items()
        }
        tokenizer = self.tokenizer
        logits_processor = None
        if settings.enforce_vertex_count:
            from transformers import LogitsProcessorList

            from .geometry import GRID_SIZE

            coord_ids = [tokenizer.convert_tokens_to_ids(f"<{ax}{v}>")
                         for ax in ("x", "y") for v in range(GRID_SIZE)]
            vcp = VertexCountProcessor(tokenizer, coord_ids,
                                       [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")])
            if vcp.active:
                logits_processor = LogitsProcessorList([vcp])
        output = self.model.generate(
            **inputs,
            generation_config=GenerationConfig(
                max_new_tokens=settings.max_new_tokens,
                do_sample=settings.do_sample,
                temperature=settings.temperature,
                top_k=settings.top_k,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            ),
            stopping_criteria=get_stop_criteria(tokenizer),
            logits_processor=logits_processor,
            return_dict_in_generate=True,
            use_cache=settings.use_cache,
        )
        # generate() 用 inputs_embeds，sequences 里**只有新 token**，不做任何切片。
        return tokenizer.decode(output.sequences[0], skip_special_tokens=False).strip()

    def predict(
        self,
        image,
        target: str = "building",
        settings: Optional[GenerationSettings] = None,
        offset: Tuple[float, float] = (0.0, 0.0),
    ) -> Dict:
        """对单张（已裁剪的）图出多边形。

        `offset` 用于把裁剪图坐标平移到整图坐标系（gradio demo 里做回映射时用）。
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        text = self.generate_text(image, target=target, settings=settings)
        grid_points = parse_polygon(text)
        polygon = grid_to_pixel(grid_points, image.size, GRID_SIZE, offset=offset)
        return {
            "text": text,
            "grid_polygon": grid_points,
            "polygon": polygon,
            "polygon_clipped": clip_polygon(polygon, image.size)
            if offset == (0.0, 0.0)
            else polygon,
            "image_size": list(image.size),
            "n_vertices": len(grid_points),
        }


def load(weights_dir: str, dtype=torch.bfloat16, device: Optional[str] = None,
         parax_adapter: Optional[str] = None) -> VectorLLMPredictor:
    return VectorLLMPredictor(weights_dir=weights_dir, dtype=dtype, device=device,
                              parax_adapter=parax_adapter)
