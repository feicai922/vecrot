#!/usr/bin/env python
"""把现有模型目录迁移成「文件夹自带 model.json」的新约定。

    python deployment/migrate_model_folders.py            # 迁移（幂等）
    python deployment/migrate_model_folders.py --check    # 只看要做什么，不写文件
    python deployment/migrate_model_folders.py --models-root "D:/vecrot/vecrot/models"

## 为什么需要

模型列表以前写死在代码里，权重散落在 ``vecrot/models/`` 与 ``third_party/`` 两处。
现在改成「一个文件夹 = 一个模型」，由文件夹里的 ``model.json`` 声明自己。
但用户机器上已经下载好的老目录（网盘 models 包）里没有 ``model.json``，
直接升级就会出现「面板里一个模型都没有」——那不是 bug，是缺声明。

本脚本按已知的目录名补写 ``model.json``，**只写声明、不动权重**：

| 目录 | 迁移成 |
|---|---|
| ``detector_strip_s_scene`` | Strip-R-CNN-S + scene_head，配置指向源码树 |
| ``detector_strip_s`` | Strip-R-CNN-S，配置指向源码树 |
| ``detector_rt-detr_bbox`` | RT-DETR（HF 目录即模型） |
| 其他含 ``.pth`` 的目录 | **不猜**：列出来让用户自己写 model.json（模板见 docs/模型下载.md） |

幂等：已经有 ``model.json`` 的目录一律跳过（用户手改过的内容不会被覆盖）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from backend.contracts import model_registry as mr   # noqa: E402

#: 已知老目录 → model.json 内容（路径都相对模型文件夹，便于整包搬机器）
LEGACY = {
    "detector_strip_s_scene": {
        "spec_version": 1,
        "key": "strip_s_gzda_scene",
        "label": "建筑专用 · Strip-R-CNN-S + 场景分类（大范围推理推荐）",
        "backend": "strip_s_gzda",
        "num_classes": 1,
        "repo": "repo:third_party/Strip-R-CNN",
        "weights": "best_mAP_iter_37620.pth",
        "config": "src:configs/gzda/strip_s_gzda_scene.py",
        "detector_options": {"score_thr": 0.05, "nms_iou": 0.3,
                             "max_per_img": 2000, "scene_threshold": 0.5},
        "order": 10,
        "note": "带 scene_head：先判窗口有无建筑，无建筑直接跳过检测与轮廓。"
                "纯背景窗口耗时降到 1/3.5。",
    },
    "detector_strip_s": {
        "spec_version": 1,
        "key": "strip_s_gzda",
        "label": "建筑专用 · Strip-R-CNN-S（旋转框，无场景分类）",
        "backend": "strip_s_gzda",
        "num_classes": 1,
        "repo": "repo:third_party/Strip-R-CNN",
        "weights": "best_mAP_iter_90288.pth",
        "config": "src:configs/gzda/strip_s_gzda.py",
        "detector_options": {"score_thr": 0.05, "nms_iou": 0.3, "max_per_img": 2000},
        "order": 20,
        "note": "不带场景分类的旧权重；功能被场景版覆盖，仅在需要复现旧结果时选它。",
    },
    "detector_rt-detr_bbox": {
        "spec_version": 1,
        "key": "rtdetr_roof",
        "label": "卫星屋顶 · RT-DETR R101vd（水平框，单类别）",
        "backend": "rtdetr_hf",
        "num_classes": 1,
        # HF 是「目录即模型」：weights 指向文件夹自身
        "weights": ".",
        "detector_options": {"score_thr": 0.3},
        "order": 40,
        "note": "HF Transformers 版 RT-DETR，1 类屋顶，固定 640×640。"
                "AP_small 仅 0.248，高分辨率影像小目标漏检偏多。",
    },
}

#: 权重文件体积下限（抓「下了一半」）
_MIN_WEIGHT_BYTES = 200 * 1024 * 1024


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def _size(path: str) -> int:
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


def write_spec(folder: str, spec: dict, check_only: bool) -> str:
    path = os.path.join(folder, mr.SPEC_FILENAME)
    if check_only:
        return f"将写入 {path}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return f"已写入 {path}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="把老模型目录迁移成 model.json 驱动")
    ap.add_argument("--models-root", default=None,
                    help="模型根目录，默认 <仓库>/vecrot/models")
    ap.add_argument("--check", action="store_true", help="只报告，不写文件")
    args = ap.parse_args(argv)

    models_root = os.path.normpath(args.models_root or mr.default_models_root(REPO))
    print(f"仓库根：{REPO}\n模型根：{models_root}\n")
    if not os.path.isdir(models_root):
        print("模型根不存在；先按 docs/模型下载.md 放好 models 包。")
        return 1

    migrated = already = 0
    unknown: list[str] = []
    for name in sorted(os.listdir(models_root)):
        folder = os.path.join(models_root, name)
        if not os.path.isdir(folder) or name.startswith((".", "_")):
            continue
        if name in mr.ASSET_FOLDERS:
            continue        # 轮廓适配器/词表基座，不是检测模型
        spec_path = os.path.join(folder, mr.SPEC_FILENAME)
        if os.path.isfile(spec_path):
            already += 1
            print(f"  · {name}: 已有 {mr.SPEC_FILENAME}，跳过")
            continue
        spec = LEGACY.get(name)
        if spec:
            # 权重没下载/没下完时也写声明：面板会显示"缺文件"并指出缺哪个，
            # 这比"列表里干脆没有这一项"更容易排查。
            weight_path = mr.resolve_path(spec.get("weights"), folder, REPO, models_root)
            size = _size(weight_path) if weight_path and os.path.exists(weight_path) else 0
            note = "" if size >= _MIN_WEIGHT_BYTES else f"（权重未就绪或偏小：{_human(size)}）"
            print(f"  ✓ {name}: {write_spec(folder, spec, args.check)}{note}")
            migrated += 1
            continue
        # 不认识的目录：够大的 .pth 值得提醒，纯空目录忽略
        pths = [f for f in os.listdir(folder) if f.lower().endswith((".pth", ".pt", ".safetensors"))]
        if pths or os.path.isfile(os.path.join(folder, "config.json")):
            unknown.append(name)

    print()
    if unknown:
        print("以下目录有模型文件但没有 model.json，面板里不会出现（也不知道该怎么配）：")
        for name in unknown:
            print(f"  ? {name}")
        print("  新建一个 model.json 即可，字段见 docs/模型下载.md 的“模型文件夹约定”。")
    print(f"迁移完成：新写 {migrated} 个，已有 {already} 个。"
          + ("（--check 模式，未真正写文件）" if args.check else ""))

    found, skipped = mr.scan(models_root, REPO)
    print(f"\n扫描结果：{len(found)} 个模型文件夹可用/待补文件")
    for p in found:
        state = "可用" if p.is_available() else "缺文件"
        print(f"  [{'✓' if p.is_available() else '✗'}] {p.key} ({state}) -> {p.label}")
        for item in p.missing():
            print(f"        {item}")
    for item in skipped:
        print(f"  [!] {item['folder']}: {item['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
