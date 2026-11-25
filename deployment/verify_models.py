#!/usr/bin/env python
"""检查模型文件是否放好（文件夹约定版）。

    python deployment/verify_models.py
    python deployment/verify_models.py --models-root "D:/vecrot/vecrot/models"

只做存在性与最小体积检查，**不加载模型、不 import torch**，
所以系统 Python 或 QGIS 的 Python 都能跑。

检查三块：

1. **模型文件夹**：扫 ``vecrot/models/*/model.json``，逐个报"齐备 / 缺哪个文件"。
   权重会查体积下限 —— 抓「下了一半」比只判断存在更实用。
2. **轮廓资产**：适配器 + 635 词表基座（不属于任何检测模型文件夹）。
3. **检测源码树**：Strip-R-CNN 必需；RiO-DETRv2 只在有 rio 模型时可缺。

退出码：0 = 全部就绪；1 = 有硬缺失。
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from backend.contracts import model_registry as mr   # noqa: E402

#: 单个权重文件的最小体积：小于它多半是没下完或下成了网页
MIN_WEIGHT_BYTES = 200 * 1024 * 1024

#: 轮廓链路资产（不属于检测模型文件夹，单独查）
VECTOR_ITEMS = (
    ("vecrot/models/adaptor_sc_055m/trainable.pt",
     "ScaleCond 适配器（网盘 models 包）", True, 500 * 1024),
    ("vecrot/models/official_vocab_635/model.safetensors",
     "轮廓基座（635 词表，按 prune 映射重建）", True, 1400 * 1024 * 1024),
    ("vecrot/models/official_vocab_635/tokenizer.json",
     "轮廓基座 tokenizer", True, 1024),
    ("vecrot/models/official_vocab_635/config.json",
     "轮廓基座配置", True, 256),
    ("vecrot/lib/vectorizer.py",
     "VectorLLM 推理封装（仓库自带）", True, 1024),
    ("vecrot/lib/vectorllm/hub.py",
     "VectorLLM 模型构建代码（仓库自带）", True, 256),
    ("vecrot/models/base_v1/model.safetensors",
     "VectorLLM v1 基座原料（一般不需要）", False, 1700 * 1024 * 1024),
)

#: 检测源码树
SOURCE_ITEMS = (
    ("third_party/Strip-R-CNN",
     "Strip-R-CNN（MMRotate 分支）源码", True, 1024),
    ("third_party/Strip-R-CNN/configs/gzda/gzda_dataset.py",
     "贵州数据集注册模块", True, 1),
    ("third_party/RiO-DETRv2",
     "RiO-DETR / RiO-DETRv2 源码（随 GitHub 分发）", True, 1024),
    ("third_party/RiO-DETRv2/engine/core",
     "RiO-DETR 模型构建入口（engine.core.YAMLConfig）", True, 1),
)

RENAME_HINTS = {
    "official_vocab_635": "vecrot/models/official_vocab_635",
    "best_mAP_iter_37620.pth": "vecrot/models/detector_strip_s_scene",
    "best_mAP_iter_90288.pth": "vecrot/models/detector_strip_s",
    "model.json": "docs/模型下载.md 的「模型文件夹约定」",
}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def size_of(path: str) -> int:
    """文件返回字节数；目录返回其下文件总和。"""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def check_table(items, title: str) -> list[tuple[str, str]]:
    """通用「存在性 + 最小体积」检查，返回必需项的缺失清单。"""
    print(f"\n=== {title} ===")
    missing_required = []
    for rel, desc, required, min_bytes in items:
        full = os.path.join(REPO, *rel.split(os.sep))
        exists = os.path.exists(full)
        size = size_of(full) if exists else 0
        if exists and size >= min_bytes:
            print(f"  ✓ {rel}  ({human(size)})")
            continue
        if not exists:
            reason = "未找到"
        elif size < min_bytes:
            reason = f"体积偏小（{human(size)} < 期望 {human(min_bytes)}），可能没下完"
        else:
            reason = "异常"
        mark = "✗" if required else "-"
        print(f"  {mark} {rel}")
        print(f"      {desc} · {reason}")
        if required:
            missing_required.append((rel, reason))
    return missing_required


def report_models(models_root: str) -> tuple[int, list[str]]:
    """扫模型文件夹并报告。返回 (可用数, 硬问题清单)。"""
    print(f"\n=== 模型文件夹（{models_root}）===")
    presets, skipped = mr.scan(models_root, REPO)
    problems: list[str] = []
    usable = 0
    if not presets:
        print("  ✗ 一个模型都没有：模型包缺少 model.json 声明")
        print("    多半是下载了旧版 models 包：跑 python deployment/migrate_model_folders.py 补写，")
        print("    或按 docs/模型下载.md 的『模型文件夹约定』自己建文件夹")
        problems.append("没有可用的检测模型文件夹")
    for p in presets:
        miss = p.missing()
        if not miss:
            usable += 1
            print(f"  ✓ {p.key}  [{mr.BACKEND_LABELS.get(p.backend, p.backend)}]")
            print(f"      {p.label}")
            weights = p.files.get("weights")
            if weights and os.path.isfile(weights):
                size = size_of(weights)
                if size < MIN_WEIGHT_BYTES:
                    print(f"      ⚠ 权重只有 {human(size)}，可能没下完")
                else:
                    print(f"      权重 {human(size)}")
            continue
        print(f"  ✗ {p.key}  [{p.backend}]")
        for item in miss:
            print(f"      {item}")
        problems.append(f"{p.key} 缺文件")
    for item in skipped:
        print(f"  [!] {os.path.basename(item['folder'])}: {item['reason']}")
    return usable, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="检查模型文件（文件夹约定）")
    ap.add_argument("--models-root", default=None,
                    help="模型根目录，默认 <仓库>/vecrot/models（只影响模型文件夹的扫描；"
                         "轮廓资产与源码树始终相对仓库根检查）")
    args = ap.parse_args(argv)

    models_root = mr.resolve_models_root(REPO, args.models_root)
    print(f"仓库根：{REPO}")
    if args.models_root:
        print("（--models-root 只影响模型文件夹扫描；轮廓资产/源码树仍按仓库根检查）")

    usable, problems = report_models(models_root)
    problems += [rel for rel, _ in check_table(VECTOR_ITEMS, "轮廓链路资产")]
    problems += [rel for rel, _ in check_table(SOURCE_ITEMS, "检测源码树")]

    print()
    if problems:
        print(f"{len(problems)} 项未就绪：")
        for item in problems:
            print(f"  · {item}")
            for key, hint in RENAME_HINTS.items():
                if key in item:
                    print(f"    提示：{hint}")
        print("\n下载与放置说明见 docs/模型下载.md。放好后重跑本脚本。")
        return 1
    print(f"全部就绪：{usable} 个模型文件夹可用。")
    print("环境：python install.py --check 会顺带检查各环境解释器。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
