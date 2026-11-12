#!/usr/bin/env python
"""检查模型文件是否放在正确位置。

    python deployment/verify_models.py

只做存在性与最小体积检查，**不加载模型、不 import torch**，
所以用系统 Python 或 QGIS 的 Python 都能跑。

退出码：0 = 必需项全部就绪；1 = 有缺失。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# (相对路径, 说明, 是否必需, 最小体积字节)
# 最小体积用来抓「下了一半」和「放了个空目录」——比只判断存在更实用。
ITEMS = (
    ("vecrot/models/detector_strip_s_scene/best_mAP_iter_37620.pth",
     "Strip-R-CNN-S 场景分支权重", True, 500 * 1024 * 1024),
    ("vecrot/models/detector_strip_s_scene/strip_s_gzda_scene.py",
     "场景版推理配置", True, 1024),
    ("vecrot/models/detector_rt-detr_bbox/model.safetensors",
     "RT-DETR 权重（可选的预设 2）", False, 250 * 1024 * 1024),
    ("vecrot/models/detector_rt-detr_bbox/config.json",
     "RT-DETR 配置（可选的预设 2）", False, 256),
    ("vecrot/models/official_vocab_635/model.safetensors",
     "轮廓基座（635 词表）", True, 1400 * 1024 * 1024),
    ("vecrot/models/official_vocab_635/tokenizer.json",
     "轮廓基座 tokenizer", True, 1024),
    ("vecrot/models/official_vocab_635/config.json",
     "轮廓基座配置", True, 256),
    ("vecrot/models/detector_strip_s/Strip-R-CNN",
     "Strip-R-CNN 源码目录（MMRotate 分支）", True, 1024),
    ("vecrot/models/detector_strip_s/Strip-R-CNN/configs/gzda/gzda_dataset.py",
     "贵州数据集注册模块", True, 1),
    ("vecrot/models/detector_strip_s/Strip-R-CNN/configs/gzda/scene_branch.py",
     "场景分类模型模块", True, 1),
    ("vecrot/models/detector_strip_s/Strip-R-CNN/configs/gzda/scene_hooks.py",
     "场景分类 Hook 注册模块", True, 1),
    ("vecrot/models/adaptor_sc_055m/trainable.pt",
     "ScaleCond 适配器（网盘 models 包）", True, 500 * 1024),
    ("vecrot/lib/vectorizer.py",
     "VectorLLM 推理封装（仓库自带）", True, 1024),
    ("vecrot/lib/vectorllm/hub.py",
     "VectorLLM 模型构建代码（仓库自带）", True, 256),
    ("vecrot/models/base_v1/model.safetensors",
     "VectorLLM v1 基座原料（一般不需要）", False, 1700 * 1024 * 1024),
)

RENAME_HINTS = {
    "official_vocab_635": "vecrot/models/official_vocab_635",
    "best_mAP_iter_37620.pth": "vecrot/models/detector_strip_s_scene",
    "best_mAP_iter_90288.pth": "vecrot/models/detector_strip_s",
}


def human(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB"):
        n /= 1024.0
        if n < 1024 or unit == "GB":
            return f"{n:.2f} {unit}" if n < 10 else f"{n:.1f} {unit}"
    return f"{n:.1f} GB"


def size_of(path: str) -> int:
    """文件返回字节数；目录返回其下文件总和。"""
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def main() -> int:
    print(f"仓库根：{REPO}\n")
    missing_required = []
    missing_optional = []
    ok_count = 0

    for rel, desc, required, min_bytes in ITEMS:
        full = os.path.join(REPO, *rel.split("/"))
        exists = os.path.exists(full)
        size = size_of(full) if exists else 0
        good = exists and size >= min_bytes

        if good:
            print(f"  ✓ {rel}")
            print(f"      {desc} · {human(size)}")
            ok_count += 1
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
        (missing_required if required else missing_optional).append((rel, reason))

    print()
    if missing_required:
        print(f"{len(missing_required)} 项必需文件未就绪：\n")
        for rel, reason in missing_required:
            print(f"  · {rel}")
            print(f"    {reason}")
            for key, hint in RENAME_HINTS.items():
                if key in rel:
                    print(f"    提示：这个目录/文件应放在 {hint}/ 下，名字不要改")
        print("\n下载地址见 docs/模型下载.md。放好后重跑本脚本。")
        return 1

    extra = ""
    if missing_optional:
        # base_v1 是「只在重建基座时才需要」的原料，不属于「少了会丢功能」，
        # 单独说一句，避免和 RT-DETR 那种「少了少一个预设」混为一谈。
        lost = [r for r, _ in missing_optional if "base_v1" not in r]
        if lost:
            names = "、".join(os.path.basename(r) for r in lost)
            extra = f"\n（可选未装：{names} —— 只影响对应预设，不影响主流程）"
    print(f"全部就绪（{ok_count} 项）。{extra}")
    print("下一步：按 docs/安装教程-自建环境.md 建好三个 conda 环境。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
