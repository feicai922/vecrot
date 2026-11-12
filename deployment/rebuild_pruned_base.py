"""从 base_v1 重建训练口径的 635 词表基座（official_vocab）。

## 为什么需要这一步

`vecrot/models/adaptor_sc_055m/backbone.json` 声明的基座是
`/data1/wzh26/vectorllm_v2/work_dirs/vocab_prune/official_vocab` —— 一个
**已按 prune 映射重排过的 635 词表基座**。它不在本仓库。

仓库里只有 `base_v1`（原始 Qwen3 全词表 151926 行）。直接拿它建模型再
`resize_token_embeddings(635)` 会截断取前 635 行，导致
`<pixel>`/`<xN>`/`<yN>` 的 embedding 行与 token 语义错位 ——
模型能前向但只输出换行（已实测）。

## 重建依据（实测确认）

逐 token 比对 adaptor tokenizer(635) 与 base tokenizer(151926)：

* 634/635 个 token 能在 base 里按**同名字符串**找到
* 映射严格单调递增（633/634 对），即 prune = 按顺序挑子集
* adaptor 0~376     → base 0~376      （普通 BPE，原样保留）
* adaptor 377~633   → base 151669~151925（`<pixel>`、`<x0..127>`、`<y0..127>`）
* adaptor 634       → `<no_object>`，base 里没有，需新初始化

## 做法

1. 按映射从 base 的 `embed_tokens.weight` 抽行，得到 635×1024
2. 对 `<no_object>` 这一行用被抽走行的均值初始化（与 HF resize 的
   mean-resizing 思路一致，保证量级不突变）
3. 同步处理 `lm_head.weight`（若存在且未 tie）
4. 复制 base 的 config / processor / tokenizer 文件，config 里改 vocab_size 与
   pixel_idx；tokenizer 用 adaptor 的（它定义了 635 词表）
5. 写出 SHA256 与映射统计，供 report 追溯

输出目录默认 `vecrot/models/official_vocab_635`。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_BASE = os.path.join(REPO, "vecrot", "models", "base_v1")
ADAPTOR = os.path.join(REPO, "vecrot", "models", "adaptor_sc_055m")
OUT_DIR = os.path.join(REPO, "vecrot", "models", "official_vocab_635")

#: 这些文件从 base 复制（权重与模型代码）
COPY_FROM_BASE = [
    "config.json", "configuration_vectorllm.py", "modeling_vectorllm.py",
    "image_processing_vectorllm.py", "processing_vectorllm.py",
    "preprocessor_config.json", "generation_config.json",
    "adaptor_base.py", "adaptor_generic.py", "adaptor_mlp.py",
    "adaptor_registry.py", "cls_token.py", "common.py",
    "dinov2_arch.py", "dual_hybrid_vit.py", "enable_cpe_support.py",
    "enable_spectral_reparam.py", "eradio_model.py", "extra_models.py",
    "extra_timm_models.py", "feature_normalizer.py", "forward_intermediates.py",
    "hf_model.py", "input_conditioner.py", "open_clip_adaptor.py",
    "radio_model.py", "vit_patch_generator.py", "vitdet.py",
]
#: tokenizer 相关从 adaptor 取（635 词表定义在它那里）
COPY_FROM_ADAPTOR = [
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
]


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_mapping() -> tuple[list[int], list[str]]:
    """adaptor id → base id 的映射；-1 表示 base 里没有（需新初始化）。"""
    from transformers import AutoTokenizer

    base_tok = AutoTokenizer.from_pretrained(SRC_BASE)
    ad_tok = AutoTokenizer.from_pretrained(ADAPTOR)
    base_tokens = [base_tok.convert_ids_to_tokens(i) for i in range(len(base_tok))]
    ad_tokens = [ad_tok.convert_ids_to_tokens(i) for i in range(len(ad_tok))]

    index: dict[str, int] = {}
    for i, t in enumerate(base_tokens):
        index.setdefault(t, i)
    mapping = [index.get(t, -1) for t in ad_tokens]
    new_tokens = [ad_tokens[i] for i, m in enumerate(mapping) if m < 0]
    return mapping, new_tokens


def main() -> int:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    print(f"源基座 : {SRC_BASE}")
    print(f"适配器 : {ADAPTOR}")
    print(f"输出   : {OUT_DIR}\n")

    mapping, new_tokens = build_mapping()
    n_miss = sum(1 for m in mapping if m < 0)
    print(f"映射：{len(mapping)} 条，命中 {len(mapping) - n_miss}，"
          f"需新建 {n_miss} 个 token {new_tokens}")

    # 单调性检查：prune 是按顺序挑子集，映射应基本单调
    hits = [m for m in mapping if m >= 0]
    mono = sum(1 for i in range(1, len(hits)) if hits[i] > hits[i - 1])
    print(f"单调递增：{mono}/{len(hits) - 1}")
    if mono < len(hits) * 0.9:
        print("⚠ 映射不单调，可能不是纯子集 prune；请人工确认后再继续")
        return 1

    src_path = os.path.join(SRC_BASE, "model.safetensors")
    if not os.path.isfile(src_path):
        print(f"缺少基座权重：{src_path}")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n读取基座权重…")
    tensors = {}
    with safe_open(src_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    print(f"  张量数 {len(tensors)}，总 {sum(t.numel() for t in tensors.values())/1e6:.1f} M")

    # ---- 抽行：embed_tokens 与 lm_head（若存在且独立）
    idx = torch.tensor(mapping, dtype=torch.long)
    new_rows = []
    stats = {}
    for key in list(tensors):
        if not (key.endswith("embed_tokens.weight") or key.endswith("lm_head.weight")):
            continue
        w = tensors[key]
        old_rows = int(w.shape[0])
        if old_rows == len(mapping):
            print(f"  {key}: 已是 {old_rows} 行，跳过")
            continue
        valid = idx >= 0
        picked = w[idx[valid].to(w.device)]
        # 新 token 用被抽走行的均值初始化（量级与 mean-resizing 一致）
        if not valid.all():
            mean = picked.to(torch.float32).mean(dim=0).to(w.dtype)
            pad = mean.unsqueeze(0).expand(int((~valid).sum()), -1)
            out = torch.empty(len(mapping), w.shape[1], dtype=w.dtype)
            out[valid] = picked
            out[~valid] = pad
            picked = out
        tensors[key] = picked.contiguous()
        stats[key] = {"old_rows": old_rows, "new_rows": int(picked.shape[0]),
                      "new_token_init": "mean_of_picked"}
        print(f"  {key}: {old_rows} → {picked.shape[0]} 行")

    if not stats:
        print("⚠ 没有需要抽行的 embedding，检查 key 名")
        return 1

    # tie_word_embeddings=True 时 lm_head 与 embed 共用张量：
    # 抽行后必须显式重新绑定，否则会指向旧张量
    embed_key = next((k for k in tensors if k.endswith("embed_tokens.weight")), None)
    lm_key = next((k for k in tensors if k.endswith("lm_head.weight")), None)
    if embed_key and lm_key:
        tensors[lm_key] = tensors[embed_key]

    print("\n写出权重…")
    out_safetensors = os.path.join(OUT_DIR, "model.safetensors")
    save_file(tensors, out_safetensors, metadata={"format": "pt"})
    print(f"  {out_safetensors}  ({os.path.getsize(out_safetensors)/1e9:.2f} GB)")

    # ---- 复制模型代码与 tokenizer
    print("\n复制配置与 tokenizer…")
    for name in COPY_FROM_BASE:
        src = os.path.join(SRC_BASE, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(OUT_DIR, name))
    for name in COPY_FROM_ADAPTOR:
        src = os.path.join(ADAPTOR, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(OUT_DIR, name))
        else:
            # 有的在子目录 tokenizer/ 下
            alt = os.path.join(ADAPTOR, "tokenizer", name)
            if os.path.isfile(alt):
                shutil.copy2(alt, os.path.join(OUT_DIR, name))

    # ---- 改 config：vocab_size 与 pixel_idx
    cfg_path = os.path.join(OUT_DIR, "config.json")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    from transformers import AutoTokenizer

    ad_tok = AutoTokenizer.from_pretrained(ADAPTOR)
    pix_id = int(ad_tok.convert_tokens_to_ids("<pixel>"))
    old_pix = cfg.get("pixel_idx")
    cfg["pixel_idx"] = pix_id
    llm = cfg.get("llm_config") or {}
    llm["vocab_size"] = len(mapping)
    cfg["llm_config"] = llm
    text_cfg = cfg.get("text_config") or {}
    if "vocab_size" in text_cfg:
        text_cfg["vocab_size"] = len(mapping)
        cfg["text_config"] = text_cfg
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"  config: pixel_idx {old_pix} → {pix_id}, vocab_size → {len(mapping)}")

    # ---- 校验：用新目录能建出 635 行模型
    print("\n校验…")
    sys.path.insert(0, os.path.join(REPO, "vecrot", "lib"))
    from vectorllm.hub import build_model

    m = build_model(OUT_DIR, torch.float32, "cpu")
    rows = int(m.get_input_embeddings().weight.shape[0])
    print(f"  embedding rows = {rows}")
    ok_rows = rows == len(mapping)
    ok_pix = int(m.pixel_idx) == pix_id
    # 抽查：pixel 行应与 base 的 pixel 行一致
    with safe_open(src_path, framework="pt", device="cpu") as f:
        base_emb = f.get_tensor(embed_key)
    same = torch.allclose(m.get_input_embeddings().weight[pix_id].float(),
                          base_emb[mapping[pix_id]].float())
    print(f"  pixel_idx 对齐 = {ok_pix}")
    print(f"  <pixel> 行与 base 对应行一致 = {same}")

    manifest = {
        "generated_from": {
            "base": SRC_BASE,
            "base_sha256": _sha256(src_path),
            "adaptor_tokenizer": ADAPTOR,
        },
        "output": OUT_DIR,
        "output_safetensors_sha256": _sha256(out_safetensors),
        "vocab_size": len(mapping),
        "pixel_idx": pix_id,
        "mapping_stats": {
            "total": len(mapping),
            "hit": len(mapping) - n_miss,
            "newly_initialized": new_tokens,
        },
        "row_rewrites": stats,
        "verification": {
            "embedding_rows": rows,
            "rows_match": ok_rows,
            "pixel_idx_match": ok_pix,
            "pixel_row_equals_base": bool(same),
        },
        "note": (
            "这是按「同名字符串 + 单调子集」推断出的 prune 映射重建的基座，"
            "不是训练时的原始 official_vocab 文件。"
            "映射依据：adaptor tokenizer 与 base tokenizer 逐 token 比对，"
            "634/635 命中且 633/634 单调递增。"
            "若原 prune 映射有额外重排，本重建会与训练口径存在差异——"
            "需用原环境对比验证输出一致性后再用于正式评测。"
        ),
    }
    with open(os.path.join(OUT_DIR, "REBUILD_INFO.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n重建完成：{OUT_DIR}")
    print(f"下一步：把 manifest.json 里 models.vectorizer.base 指向该目录")
    return 0 if (ok_rows and ok_pix and same) else 1


if __name__ == "__main__":
    sys.exit(main())
