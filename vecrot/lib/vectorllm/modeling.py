"""VectorLLM 主体模型：C-RADIO 视觉塔 + MLP projector + Qwen3 解码器。

相对 v1 (`vectorllm_v1/modeling_vectorllm.py`) 的改动只有两处，且都是修复：

1. `forward()` 的 `return_dict` 默认值：v1 写的是 `if not return_dict:`
   （`return_dict is None` 时被当作 True），导致不显式传参时**恒返回 tuple**，
   `model(...).logits` 直接报 AttributeError。v2 按 HF 惯例回落到
   `self.config.use_return_dict`。
2. `generate()` 显式 `use_cache=True`：v1 的 `llm_config["use_cache"]=false` 被写进
   config.json，导致逐步重算整个前缀（640 token 时开销很可观）。
   需要复现 v1 行为时传 `use_cache=False` 即可。

其余计算逻辑与 v1 逐行一致，并有 `tests/test_parity_v1.py` 做逐 token 对齐验证。
"""

from typing import List, Optional

import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .config import ProjectorConfig, VectorLLMConfig
from .radio.hf_model import RADIOConfig as HFRADIOConfig, RADIOModel as HFRADIOModel


IGNORE_INDEX = -100


def prepare_inputs_labels_for_multimodal_vectorllm(
    llm,
    input_ids: torch.LongTensor = None,
    position_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    labels: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_token_id=None,
):
    if pixel_values is None:
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "inputs_embeds": None,
            "labels": labels,
        }

    original_labels = labels
    original_position_ids = position_ids
    original_attention_mask = attention_mask

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
        ).unsqueeze(0).expand(input_ids.shape[0], -1)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    inputs_embeds = llm.get_input_embeddings()(input_ids)
    inputs_embeds = inputs_embeds.clone()
    labels = labels.clone()

    if pixel_values.ndim != 3:
        raise ValueError(f"Expected pixel_values to have shape [B, N, C], got {tuple(pixel_values.shape)}")

    for batch_idx in range(input_ids.shape[0]):
        replace_positions = torch.where(input_ids[batch_idx] == pixel_token_id)[0]
        if replace_positions.numel() == 0:
            continue
        if replace_positions.numel() != pixel_values.shape[1]:
            raise ValueError(
                "The number of image placeholder tokens does not match the projected visual tokens: "
                f"{replace_positions.numel()} vs {pixel_values.shape[1]}"
            )
        inputs_embeds[batch_idx, replace_positions] = pixel_values[batch_idx].to(inputs_embeds.dtype)
        labels[batch_idx, replace_positions] = IGNORE_INDEX

    return {
        "input_ids": None,
        "position_ids": None if original_position_ids is None else position_ids,
        "attention_mask": None if original_attention_mask is None else attention_mask.to(dtype=original_attention_mask.dtype),
        "past_key_values": past_key_values,
        "inputs_embeds": inputs_embeds,
        "labels": None if original_labels is None else labels,
    }


class ProjectorModel(PreTrainedModel):
    config_class = ProjectorConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False
        modules = [
            nn.Linear(config.visual_hidden_size, config.llm_hidden_size, bias=config.bias)
        ]
        for _ in range(1, config.depth):
            modules.append(ACT2FN[config.hidden_act])
            modules.append(
                nn.Linear(config.llm_hidden_size, config.llm_hidden_size, bias=config.bias)
            )
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self.model, x)
        return self.model(x)


class VectorLLMForCausalLM(PreTrainedModel):
    config_class = VectorLLMConfig
    main_input_name = "pixel_values"
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: VectorLLMConfig,
        vision_model=None,
        language_model=None,
        projector=None,
        pos_embeds=None,
    ):
        super().__init__(config)

        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = HFRADIOModel(HFRADIOConfig(**config.vision_config))
            target_dtype = getattr(torch, config.vision_torch_dtype, None)
            if target_dtype is not None:
                self.vision_model = self.vision_model.to(dtype=target_dtype)

        if language_model is not None:
            self.language_model = language_model
        else:
            self.language_model = Qwen3ForCausalLM(Qwen3Config(**config.llm_config))

        if projector is not None:
            self.projector = projector
        else:
            self.projector = ProjectorModel(ProjectorConfig(**config.projector_config))

        width = config.regression_size[0] // config.patch_size
        height = config.regression_size[1] // config.patch_size
        n_pos = width * height
        if pos_embeds is not None:
            self.visual_pos_embeddings = pos_embeds
        else:
            self.visual_pos_embeddings = nn.Embedding(n_pos, config.vision_hidden_size)

        self.pixel_idx = config.pixel_idx
        self.num_cls_register_tokens = config.num_cls_register_tokens
        # 可选架构模块（由 attach_global_fusion / attach_scale_edge 挂上；None = 不启用）
        self.global_fusion = None        # 目标感知全局上下文融合（整幅瓦片作 K/V）
        self.boundary_enhance = None     # (HR) 边界增强 / 边界+角点引导增强
        self.scale_cond = None           # 尺度条件 FiLM（scale_edge.ScaleCond）
        self.zoom_fusion = None          # 放大分支动态融合（scale_edge.ZoomFusion）
        self.context_module = None       # A/B/C 上下文模块（ctx_abc.ACDF/NRS/NCR）
        self.det_global_adapter = None   # 检测器多层 FPN token → K/V 适配器（global_fusion.DetGlobalAdapter）
        self.instance_memory = None      # 跨实例空间记忆（instance_memory.InstanceMemory）

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # `resize_token_embeddings()` 会调用下面两个 setter；v1 没有实现，
    # 导致任何"加 token 再训练"的尝试直接 NotImplementedError。转发给语言模型即可。
    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, new_embeddings):
        return self.language_model.set_output_embeddings(new_embeddings)

    def extract_feature(self, pixel_values, global_pixel_values=None, pos_code=None,
                        hr_pixel_values=None, aux_out=None, scale_code=None,
                        zoom_pixel_values=None, scale_gate=None, pos_box=None,
                        ctx_geom=None, ctx_tile_pixel_values=None,
                        ctx_neighbors=None, ctx_neighbor_scores=None,
                        ctx_neighbor_mask=None, det_global_tokens=None,
                        memory_maps=None):
        """局部裁剪图（+ 可选：全局瓦片、位置/尺度编码、高分辨率原图、放大分支）→ LLM 空间的 64 个 token。

        顺序严格是：视觉塔 → +可学位置编码 → ① 尺度条件 → ② 放大分支融合 →
        ③ 全局上下文融合 → ④ 边界/角点增强 → projector。

        * `global_pixel_values`：整幅瓦片（与局部共用同一个视觉塔，权重共享，只作 K/V）；
        * `pos_code`：裁剪框在整幅图里的归一化位置/尺寸（`global_fusion.pos_code_from_box`）；
        * `hr_pixel_values`：**原始高分辨率 crop**（`global_fusion.hr_tensor_from_crop`），只喂边界分支；
        * `scale_code`：目标相对裁剪框的尺度编码（`scale_edge.scale_code`），喂 ① 与 ②；
        * `zoom_pixel_values`：更紧外扩的同一目标（`scale_edge.build_zoom_inputs`），只喂 ②；
        * `aux_out`：如果给了 dict，就把辅助输出（边界/角点 logits 等）写进去，供训练期算辅助损失。

        所有模块都是 `[B,64,D] → [B,64,D]`，**不增删 token**（processor/坐标词表按 64 写死），
        且都是**零初始化输出投影**：不挂模块或 step0 时与基线逐位一致。
        """
        summary, features = self.vision_model(pixel_values.to(self.vision_model.dtype))
        del summary
        pos_embed = self.visual_pos_embeddings.weight.unsqueeze(0)
        pos_embed = pos_embed.repeat(features.shape[0], 1, 1)
        features = features + pos_embed

        # ① 尺度条件（最先：让后面所有分支都看得到"目标在框里占多大"）
        if getattr(self, "scale_cond", None) is not None:
            features = self.scale_cond(features, scale_code, aux_out=aux_out, gate=scale_gate)
        # ①' 跨实例空间记忆（2026-11-08）：历史多边形占用+当前框边界 → 加性注入
        #     （走 ScaleCond 同款加性通路；GF 的 attention 通路 ls1 实测打不开，不再走 K/V）
        if getattr(self, "instance_memory", None) is not None and memory_maps is not None:
            features = self.instance_memory(features, memory_maps, aux_out=aux_out)
        # ② 放大分支（同一目标、更紧外扩）+ 样本级动态门控
        if getattr(self, "zoom_fusion", None) is not None and zoom_pixel_values is not None:
            _, zfeat = self.vision_model(zoom_pixel_values.to(self.vision_model.dtype))
            zfeat = zfeat + pos_embed[: zfeat.shape[0]]
            features = self.zoom_fusion(features, zfeat, scale_code, aux_out=aux_out)
        # ③ 全局上下文（K/V 两种来源：检测器 FPN token 优先，否则整幅瓦片过视觉塔）
        if (self.global_fusion is not None and det_global_tokens is not None
                and getattr(self, "det_global_adapter", None) is not None):
            gfeat = self.det_global_adapter(det_global_tokens)
            features = self.global_fusion(features, gfeat, pos_code, aux_out=aux_out,
                                          pos_box=pos_box)
        elif self.global_fusion is not None and global_pixel_values is not None:
            _, gfeat = self.vision_model(global_pixel_values.to(self.vision_model.dtype))
            gfeat = gfeat + pos_embed[: gfeat.shape[0]]
            features = self.global_fusion(features, gfeat, pos_code, aux_out=aux_out,
                                          pos_box=pos_box)
        # ④ 边界/角点增强（HR 分支）
        if self.boundary_enhance is not None:
            features = self.boundary_enhance(features, hr_pixel_values=hr_pixel_values,
                                             aux_out=aux_out)
        # ⑤ A/B/C 上下文模块（ctx_abc.py）：整幅瓦片特征 + 几何 (+ 邻居框)
        if self.context_module is not None and ctx_geom is not None and ctx_tile_pixel_values is not None:
            from .ctx_abc import geom_split

            _, tile_feat = self.vision_model(ctx_tile_pixel_values.to(self.vision_model.dtype))
            side = int(round(math.sqrt(tile_feat.shape[1])))
            if side * side == tile_feat.shape[1]:       # 只在网格是正方形时按 2D 采样
                Fmap = tile_feat.reshape(tile_feat.shape[0], side, side, -1).permute(0, 3, 1, 2)
                nb = None
                if ctx_neighbors is not None:
                    nb = {"boxes": ctx_neighbors, "scores": ctx_neighbor_scores,
                          "mask": ctx_neighbor_mask}
                features = self.context_module(features, Fmap, geom_split(ctx_geom),
                                               aux_out=aux_out, nb=nb)

        features = features.to(self.projector.dtype)
        return self.projector(features)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        global_pixel_values: Optional[torch.FloatTensor] = None,
        pos_code: Optional[torch.FloatTensor] = None,
        hr_pixel_values: Optional[torch.FloatTensor] = None,
        aux_out: Optional[dict] = None,
        scale_code: Optional[torch.FloatTensor] = None,
        zoom_pixel_values: Optional[torch.FloatTensor] = None,
        scale_gate: Optional[torch.FloatTensor] = None,
        pos_box: Optional[torch.FloatTensor] = None,
        ctx_geom: Optional[torch.FloatTensor] = None,
        ctx_tile_pixel_values: Optional[torch.FloatTensor] = None,
        ctx_neighbors: Optional[torch.FloatTensor] = None,
        ctx_neighbor_scores: Optional[torch.FloatTensor] = None,
        ctx_neighbor_mask: Optional[torch.BoolTensor] = None,
        det_global_tokens: Optional[torch.FloatTensor] = None,
        memory_maps: Optional[torch.FloatTensor] = None,
    ):
        if inputs_embeds is None and pixel_values is not None:
            if isinstance(pixel_values, list):
                pixel_values = [item.unsqueeze(0) if item.ndim == 3 else item for item in pixel_values]
                pixel_values = torch.cat(pixel_values, dim=0)
            pixel_values = pixel_values.to(self.device)
            projected = self.extract_feature(pixel_values, global_pixel_values=global_pixel_values,
                                             pos_code=pos_code, hr_pixel_values=hr_pixel_values,
                                             aux_out=aux_out, scale_code=scale_code,
                                             zoom_pixel_values=zoom_pixel_values,
                                             scale_gate=scale_gate, pos_box=pos_box,
                                             ctx_geom=ctx_geom,
                                             ctx_tile_pixel_values=ctx_tile_pixel_values,
                                             ctx_neighbors=ctx_neighbors,
                                             ctx_neighbor_scores=ctx_neighbor_scores,
                                             ctx_neighbor_mask=ctx_neighbor_mask,
                                             det_global_tokens=det_global_tokens,
                                             memory_maps=memory_maps)
            llm_inputs = prepare_inputs_labels_for_multimodal_vectorllm(
                llm=self.language_model,
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
                pixel_values=projected,
                pixel_token_id=self.pixel_idx,
            )
            inputs_embeds = llm_inputs["inputs_embeds"]
            attention_mask = llm_inputs["attention_mask"]
            position_ids = llm_inputs["position_ids"]
            labels = llm_inputs["labels"]
            input_ids = llm_inputs["input_ids"]

        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            # 注意：用 logits 的实际最后一维，而不是 config.vocab_size —— 一旦扩了词表
            # （例如新增 <no_object>），两者会不一致，view 会直接报错。
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        use_cache: bool = True,
        global_pixel_values: Optional[torch.FloatTensor] = None,
        pos_code: Optional[torch.FloatTensor] = None,
        hr_pixel_values: Optional[torch.FloatTensor] = None,
        scale_code: Optional[torch.FloatTensor] = None,
        zoom_pixel_values: Optional[torch.FloatTensor] = None,
        scale_gate: Optional[torch.FloatTensor] = None,
        pos_box: Optional[torch.FloatTensor] = None,
        ctx_geom: Optional[torch.FloatTensor] = None,
        ctx_tile_pixel_values: Optional[torch.FloatTensor] = None,
        ctx_neighbors: Optional[torch.FloatTensor] = None,
        ctx_neighbor_scores: Optional[torch.FloatTensor] = None,
        ctx_neighbor_mask: Optional[torch.BoolTensor] = None,
        det_global_tokens: Optional[torch.FloatTensor] = None,
        memory_maps: Optional[torch.FloatTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        # ⚠️ 2026-11-02 修：`forward()` 加了 global_pixel_values/pos_code，`generate()` 签名漏了 →
        # 只要挂了融合模块，第一次验收评测就 NameError（sft_fusion_offinit 在 step500 崩过一次）。
        # ⚠️ 2026-11-03：`scale_code` / `zoom_pixel_values` 同样必须同步（同一个坑，第三次了）。
        if pixel_values is not None:
            if isinstance(pixel_values, list):
                pixel_values = [item.unsqueeze(0) if item.ndim == 3 else item for item in pixel_values]
                pixel_values = torch.cat(pixel_values, dim=0)
            pixel_values = pixel_values.to(self.device)
            input_ids = input_ids.to(self.device)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            projected = self.extract_feature(pixel_values, global_pixel_values=global_pixel_values,
                                             pos_code=pos_code,
                                             hr_pixel_values=hr_pixel_values,
                                             scale_code=scale_code,
                                             zoom_pixel_values=zoom_pixel_values,
                                             scale_gate=scale_gate, pos_box=pos_box,
                                             ctx_geom=ctx_geom,
                                             ctx_tile_pixel_values=ctx_tile_pixel_values,
                                             ctx_neighbors=ctx_neighbors,
                                             ctx_neighbor_scores=ctx_neighbor_scores,
                                             ctx_neighbor_mask=ctx_neighbor_mask,
                                             det_global_tokens=det_global_tokens,
                                             memory_maps=memory_maps).to(input_embeds.dtype)
            batch, seqlen, channels = input_embeds.shape
            flat_embeds = input_embeds.reshape(batch * seqlen, channels)
            selected = input_ids.reshape(batch * seqlen) == self.pixel_idx
            flat_embeds[selected] = projected.reshape(-1, channels).to(flat_embeds.device)
            input_embeds = flat_embeds.reshape(batch, seqlen, channels)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids.to(self.device))

        generate_kwargs.setdefault("use_cache", use_cache)
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            **generate_kwargs,
        )
        return outputs
