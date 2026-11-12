"""VectorLLM + ScaleCond 推理封装（从 vectorllm_v2/tools/eval_detect_vectorize.py 抽取，只保留推理所需部分）。

依赖已随本仓库提供：本目录 `lib/vectorllm` 包。
模型：`models/base_v1`（基座）+ `models/adaptor_sc_055m`（ScaleCond 适配，0.55M 可训）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, GenerationConfig

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # → lib/

from vectorllm.abstain import NO_OBJECT_TOKEN                                # noqa: E402
from vectorllm.geometry import GRID_SIZE, grid_to_pixel, parse_polygon       # noqa: E402
from vectorllm.hub import build_model                                         # noqa: E402
from vectorllm.image_processor import VectorLLMImageProcessor                 # noqa: E402
from vectorllm.processor import VectorLLMProcessor, register_for_auto_classes  # noqa: E402
from vectorllm.prompts import prompt_for                                      # noqa: E402

VL_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'models', 'base_v1')


class Vectorizer:
    def __init__(self, ckpt: str | None, device: str, tau: float | None = None,
                 base: str | None = None):
        # ⚠️ 2026-11-03：换过基座的 ckpt（词表裁剪 / 换骨干）必须在 `backbone.json["base"]` 指的
        #    那个目录上重建模型 —— 否则用 v1 的 151k 词表建模型再 resize 会**丢掉裁剪后的 embedding 行**。
        base_dir = base or VL_BASE
        if base is None and ckpt and (Path(ckpt) / "backbone.json").exists():
            _bb = json.loads((Path(ckpt) / "backbone.json").read_text(encoding="utf-8"))
            base_dir = _bb.get("base") or base_dir
        tokenizer = AutoTokenizer.from_pretrained(ckpt or base_dir)
        model = build_model(base_dir, torch.bfloat16, device)
        state = None
        if ckpt and (Path(ckpt) / "trainable.pt").exists():
            state = torch.load(Path(ckpt) / "trainable.pt", map_location=device, weights_only=True)
            key = next((k for k in state if k.endswith("embed_tokens.weight")), None)
            vocab = int(state[key].shape[0]) if key else len(tokenizer)
        elif ckpt:
            state = None      # 官方零训练目录没有 trainable.pt：直接用 base 权重
            vocab = len(tokenizer)
        else:
            vocab = len(tokenizer)
        if vocab != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(vocab)
            model.language_model.lm_head.weight = model.language_model.model.embed_tokens.weight
        # ⚠️ 2026-11-03 修（重要）：架构模块必须在 `load_state_dict` **之前**挂回去。
        #    以前是加载完再挂 → ckpt 里的 global_fusion.*/boundary_enhance.*（35 个张量）
        #    以及 projector.adapter.*/expert.* 全被当成 unexpected key 丢掉，
        #    模块停在零初始化（ls=0）＝恒等映射，等于**评的是没有融合/ParaX 的模型**。
        self.fusion = self._attach_arch(model, ckpt) if ckpt else None
        # 高分辨率边界分支：ckpt 说需要就从**原始 crop** 现算（推理期不看任何 GT）
        self.hr_size = None
        if self.fusion and self.fusion.get("hr_edge"):
            self.hr_size = int(self.fusion.get("hr_size", 256))
        # 尺度条件 / 放大分支（scale_edge.py）：从目标框现算 zoom 输入
        self.zoom = bool(self.fusion and self.fusion.get("zoom"))
        self.need_scale = bool(self.fusion and (self.fusion.get("scale_cond") or self.zoom))
        if state is not None:
            model.load_state_dict(state, strict=False)
        model.eval()
        ip = VectorLLMImageProcessor(do_resize=True, do_rescale=True, do_normalize=False, do_convert_rgb=True,
                                     pre_resize_size=432, resized_size=GRID_SIZE, patch_size=16)
        register_for_auto_classes()
        self.model = model
        self.tok = tokenizer
        self.proc = VectorLLMProcessor(image_processor=ip, tokenizer=tokenizer,
                                       chat_template=getattr(tokenizer, "chat_template", None))
        self.device = device
        self.prompt = prompt_for("building")
        # ⚠️ vocab_prune 目录里 config.pixel_idx 可能指向 prune 前的旧 id（如 151669），
        # 而 <pixel> 已被重映射到新 id。以 tokenizer 真实编码为准做一致性校正。
        _pix = tokenizer.convert_tokens_to_ids("<pixel>")
        if _pix is not None and _pix >= 0 and _pix != self.model.pixel_idx:
            print(f"[vl] 修正 pixel_idx: {self.model.pixel_idx} -> {_pix}（vocab prune 后映射）", flush=True)
            self.model.pixel_idx = _pix
        self.tau = tau
        print(f"[vl] {ckpt or 'OFFICIAL base'} ready (tau={tau})", flush=True)

    @staticmethod
    def _attach_arch(model, ckpt):
        """按 ckpt 里的 backbone.json / arch.json 把 ParaX 适配器、全局融合/边界增强挂回去。

        必须在 `load_state_dict` 之前调用（见 __init__ 里的注），否则权重会被丢掉。
        """
        import json as _json

        info = {}
        bb = Path(ckpt) / "backbone.json"
        cfg = _json.loads(bb.read_text(encoding="utf-8")) if bb.exists() else {}
        if int(cfg.get("parax_rank", 0) or 0) > 0 and cfg.get("parax_target") in ("projector", "both"):
            from vectorllm.parax import attach_parax_projector

            attach_parax_projector(model, rank=int(cfg.get("parax_proj_rank", 64)),
                                   n_experts=int(cfg.get("parax_experts", 4)), ls_init_value=0.0)
            print("[vl] ParaX@projector 已挂载（权重待 ckpt 载入）", flush=True)
        if (Path(ckpt) / "arch.json").exists():
            # GFv2 优先：arch.json 标 global_fusion_v2 时必须重建 v2（gate 形状与 v1 不兼容）
            from vectorllm.global_fusion_v2 import maybe_attach_from_ckpt as _maybe_gfv2

            v2_info = _maybe_gfv2(model, str(ckpt), verbose=False)
            info = v2_info or {}
            if v2_info is None:
                from vectorllm.global_fusion import maybe_attach_from_ckpt

                info = maybe_attach_from_ckpt(model, str(ckpt), verbose=True) or {}
            # 2026-11-03：尺度条件 / 放大分支 / 边界+角点（scale_edge.py）。**必须晚于**上面那次，
            # 因为 `edge_corner` 要替换掉 global_fusion 挂的 HR 边界分支（同一个 HR 编码器只算一次）。
            from vectorllm.scale_edge import maybe_attach_from_ckpt as _maybe_scale_edge

            se_info = _maybe_scale_edge(model, str(ckpt), verbose=True)
            if se_info:
                info.update(se_info)
            # 阴影/遮挡感知引导（shadow_guide.py）：同样占用 boundary_enhance，放最后
            from vectorllm.shadow_guide import maybe_attach_from_ckpt as _maybe_shadow

            sh_info = _maybe_shadow(model, str(ckpt), verbose=True)
            if sh_info:
                info.update(sh_info)
            # A/B/C 上下文模块（ctx_abc.py）：arch.json 里 ctx_module 存在才挂
            from vectorllm.ctx_abc import maybe_attach_from_ckpt as _maybe_ctx

            ctx_info = _maybe_ctx(model, str(ckpt), verbose=True)
            if ctx_info:
                info.update({"ctx_module": ctx_info})
            # 跨实例空间记忆（instance_memory.py）
            from vectorllm.instance_memory import maybe_attach_from_ckpt as _maybe_mem

            mem_info = _maybe_mem(model, str(ckpt), verbose=True)
            if mem_info:
                info.update(mem_info)
        return info

    @torch.inference_mode()
    def predict(self, crops, batch: int = 12, globals_=None, pos_codes=None, zooms=None,
                scale_gates=None, pos_boxes=None, ctx_inputs=None, det_tokens=None,
                memory_maps=None, scale_codes=None):
        """→ [(polygon in crop coords | None, abstained, z)]。

        * `globals_`/`pos_codes`：整幅瓦片 + 裁剪框位置编码 → 全局上下文融合；
        * `zooms`：`[(zoom_pixel_values[3,H,W], scale_code[64]), ...]` → 尺度条件 + 放大分支
          （由 `vectorllm.scale_edge.build_zoom_inputs` 现算，推理期不读任何标签）。
        """
        outs = []
        for s in range(0, len(crops), batch):
            chunk = crops[s:s + batch]
            inputs = self.proc(text=[self.prompt] * len(chunk), images=chunk, return_tensors="pt", padding=True)
            inputs = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            if globals_ is not None:
                inputs["global_pixel_values"] = torch.stack(globals_[s:s + batch]).to(self.device)
                inputs["pos_code"] = torch.stack(pos_codes[s:s + batch]).to(self.device)
                if pos_boxes is not None:      # 原始归一化框：给空间注意力偏置用
                    inputs["pos_box"] = torch.stack(pos_boxes[s:s + batch]).to(self.device)
            if det_tokens is not None:         # 检测器多层 FPN token（替代整图视觉塔）作 K/V
                inputs["det_global_tokens"] = torch.stack(det_tokens[s:s + batch]).to(self.device)
                # det-KV 不经过 RADIO 整图分支，但 OBB 位置编码与空间注意力偏置仍必须传入；
                # 旧版只有 globals_ 非 None 才传它们，导致 FCOS K/V 评测丢掉全部目标几何。
                if pos_codes is not None:
                    inputs["pos_code"] = torch.stack(pos_codes[s:s + batch]).to(self.device)
                if pos_boxes is not None:
                    inputs["pos_box"] = torch.stack(pos_boxes[s:s + batch]).to(self.device)
            if memory_maps is not None:        # 跨实例空间记忆：[B,2,64,64] 占用+边界
                inputs["memory_maps"] = torch.stack(memory_maps[s:s + batch]).to(self.device)
            if scale_codes is not None:       # 方向 D：外部现算的 scale_code（64+16 密度）
                inputs["scale_code"] = torch.stack(scale_codes[s:s + batch]).to(self.device)
            if zooms is not None:
                inputs["zoom_pixel_values"] = torch.stack(
                    [z[0] for z in zooms[s:s + batch]]).to(self.device)
                inputs["scale_code"] = torch.stack(
                    [z[1] for z in zooms[s:s + batch]]).to(self.device)
            if scale_gates is not None:      # 尺寸门控：1=用模块，0=退回官方路径
                inputs["scale_gate"] = torch.tensor(
                    scale_gates[s:s + batch], device=self.device, dtype=torch.float32)
            if ctx_inputs is not None:        # A/B/C 上下文模块：整幅瓦片 + 几何 (+ 邻居框)
                chunk_ctx = ctx_inputs[s:s + batch]
                inputs["ctx_geom"] = torch.stack([c["ctx_geom"] for c in chunk_ctx]).to(self.device)
                inputs["ctx_tile_pixel_values"] = torch.stack(
                    [c["ctx_tile_pixel_values"] for c in chunk_ctx]).to(self.device)
                if any("ctx_neighbors" in c for c in chunk_ctx):
                    inputs["ctx_neighbors"] = torch.stack(
                        [c.get("ctx_neighbors", torch.zeros(0, 5)) for c in chunk_ctx]).to(self.device)
                    inputs["ctx_neighbor_scores"] = torch.stack(
                        [c.get("ctx_neighbor_scores", torch.zeros(0)) for c in chunk_ctx]).to(self.device)
                    inputs["ctx_neighbor_mask"] = torch.stack(
                        [c.get("ctx_neighbor_mask", torch.zeros(0, dtype=torch.bool))
                         for c in chunk_ctx]).to(self.device)
            if self.hr_size:
                from vectorllm.global_fusion import hr_tensor_from_crop

                inputs["hr_pixel_values"] = torch.stack(
                    [hr_tensor_from_crop(c, self.hr_size) for c in chunk]).to(self.device)
            no_id = self.tok.convert_tokens_to_ids(NO_OBJECT_TOKEN)
            no_id = no_id if isinstance(no_id, int) else None
            zs = None
            if no_id is not None:
                coord_ids = [self.tok.convert_tokens_to_ids(t) for t in
                             ([f"<x{i}>" for i in range(GRID_SIZE)] + [f"<y{i}>" for i in range(GRID_SIZE)])]
                ci = torch.tensor([i for i in coord_ids if isinstance(i, int) and i >= 0], device=self.device)
                logits = self.model(**inputs, return_dict=True).logits[:, -1, :].float()
                zs = (logits[:, no_id] - torch.logsumexp(logits[:, ci], dim=1)).cpu().tolist()
            procs = None
            if no_id is not None:
                # ⚠️ 必须屏蔽 <no_object>：不然"不弃权"时模型仍可能以它开头 → 拿不到多边形（HANDOFF §7.8）
                from transformers import LogitsProcessor, LogitsProcessorList

                class _Suppress(LogitsProcessor):
                    def __call__(self, input_ids, scores):
                        scores[:, no_id] = float("-inf")
                        return scores

                procs = LogitsProcessorList([_Suppress()])
            gen = self.model.generate(
                **inputs,
                generation_config=GenerationConfig(max_new_tokens=int(__import__("os").environ.get("VEC_MAX_NEW_TOKENS", 640)),
                                                   do_sample=False, temperature=0.0, top_k=1,
                                                   eos_token_id=self.tok.eos_token_id,
                                                   pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id),
                return_dict_in_generate=True, use_cache=True, logits_processor=procs)
            for row, crop in enumerate(chunk):
                text = self.tok.decode(gen.sequences[row], skip_special_tokens=False)
                for stop in ("<|im_end|>", "<|endoftext|>"):
                    if stop in text:
                        text = text.split(stop)[0]
                grid = parse_polygon(text)
                poly = grid_to_pixel(grid, crop.size, GRID_SIZE) if len(grid) >= 3 else None
                z = zs[row] if zs is not None else None
                abstained = bool(self.tau is not None and z is not None and z > self.tau)
                outs.append((None if abstained else poly, abstained, z))
        return outs


# --------------------------------------------------------------------------- 指标


