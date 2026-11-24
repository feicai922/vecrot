#!/usr/bin/env python
"""生成「网盘模型包」目录骨架（含 model.json），方便整包压缩上传。

    python deployment/build_models_package.py --out "D:/上传/vecrot-models"
    python deployment/build_models_package.py --out "D:/上传/vecrot-models" \\
        --rio-config third_party/RiO-DETRv2/configs/rio-gzda/riov2_hgnetv2_x_gzda_12e.yml

产出**两个**目录：

* ``<out>/vecrot-models-package/``：完整包骨架。解压后应正好是用户机器上的
  ``vecrot/models/``。脚本只写 ``model.json`` / RiO 配置 / 说明文件，
  **权重由你从本机拷进来**（脚本不会去动几 GB 的权重）。
* ``<out>/vecrot-models-patch/``：增量补丁。只含各文件夹新增的 ``model.json``
  （几 KB），给"已经有老模型包"的用户升级用 —— 网盘单独传这一个就行。

设计原则：``model.json`` 的内容**复用** ``migrate_model_folders.LEGACY``，
保证"网盘包里的声明"和"迁移脚本补出来的声明"永远一致。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from backend.contracts import model_registry as mr      # noqa: E402
from deployment.migrate_model_folders import LEGACY     # noqa: E402

#: 轮廓链路资产（含旧权重），脚本只建空目录 + 说明，权重由你拷
ASSET_DIRS = {
    "adaptor_sc_055m": (
        "轮廓适配器（ScaleCond 0.55M）",
        ["trainable.pt", "tokenizer.json", "arch.json"],
    ),
    "official_vocab_635": (
        "轮廓基座（635 词表，按 prune 映射重建）",
        ["model.safetensors（约 1.5 GB）", "config.json", "tokenizer.json",
         "tokenizer_config.json", "REBUILD_INFO.json"],
    ),
    "base_v1": (
        "可选：重建基座的原料，正常推理不用",
        ["model.safetensors（约 1.7 GB）", "config.json", "tokenizer.json"],
    ),
}

#: 已有检测模型文件夹 → 权重文件名（从 LEGACY 的 weights 字段取，这里只列说明）
EXTRA_NOTES = {
    "detector_strip_s_scene": "场景分类版，面板默认项（order=10）",
    "detector_strip_s": "不带场景分类的旧权重（order=20）",
    "detector_rt-detr_bbox": "HF 目录即模型：config.json + preprocessor_config.json + model.safetensors",
}

#: 新增的检测模型（本项目这一轮接入的两个）
NEW_MODELS = {
    "detector_strip_s_fusion": {
        "spec_version": 1,
        "key": "strip_s_fusion",
        "label": "融合数据集 · Strip-R-CNN-S（旋转框）",
        "backend": "strip_s_gzda",
        "num_classes": 1,
        "repo": "repo:third_party/Strip-R-CNN",
        # 权重名固定成 best.pth：放进文件夹后改名即可（或改本字段）
        "weights": "best.pth",
        "config": "src:configs/gzda/strip_s_gzda.py",
        "detector_options": {"score_thr": 0.05, "nms_iou": 0.3, "max_per_img": 2000},
        "order": 15,
        "note": "融合数据集重训的 Strip-R-CNN-S；结构与 strip_s 相同（num_classes=1）。",
    },
    "detector_rio_gzda": {
        "spec_version": 1,
        "key": "rio_gzda",
        "label": "RiO-DETRv2-X · gzda 融合数据（旋转框）",
        "backend": "rio_detr",
        "num_classes": 1,
        "repo": "repo:third_party/RiO-DETRv2",
        "weights": "last.pth",
        "config": "riov2_x_gzda_infer.yaml",
        "detector_options": {"score_thr": 0.4},
        "order": 30,
        "note": "RiO-DETRv2-X（65.5M）。需单独环境 vecrot-rio，以及 "
                "third_party/RiO-DETRv2 源码（或其绝对路径）。",
    },
}

NEW_WEIGHT_HINTS = {
    "detector_strip_s_fusion": (
        "把融合数据集重训的 Strip-R-CNN-S 权重拷进来，命名为 best.pth\\n"
        "（或改 model.json 的 weights 字段）。约 540 MB。\\n"
        "若这次训练用了新的配置文件，把那份 .py 也放进\\n"
        "third_party/Strip-R-CNN/configs/gzda/ 并把 model.json 的 config 指向它。"),
    "detector_rio_gzda": (
        "把 RiO-DETRv2 的 checkpoint 拷进来，命名为 last.pth\\n"
        "（或改 model.json 的 weights 字段）。X 档约 1.0 GB。\\n"
        "训练产物里优先用 EMA 权重；本目录已带展开好的单文件配置\\n"
        "riov2_x_gzda_infer.yaml，不依赖源码树里的 configs。"),
}


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def build_patch(out_root: str) -> str:
    """增量补丁：只有各文件夹的 model.json（+ RiO 的单文件配置）。"""
    patch = os.path.join(out_root, "vecrot-models-patch")
    rows = []
    for folder, spec in list(LEGACY.items()) + list(NEW_MODELS.items()):
        write_json(os.path.join(patch, folder, mr.SPEC_FILENAME), spec)
        rows.append((folder, spec["key"], spec["backend"]))

    lines = [
        "# 增量补丁：只补 model.json（已有模型包的用户）",
        "",
        "从这一版开始，面板的「模型搭配」下拉框由**文件夹**驱动：",
        "每个模型文件夹里必须有一份 `model.json`，声明自己的后端/权重/配置/阈值。",
        "老模型包（网盘旧版）里没有这个文件，所以升级代码后面板会显示",
        "「没有可用的模型组合」——那不是装坏了，是缺声明。",
        "",
        "## 用法（三种，任选）",
        "",
        "1. **自动**（推荐）：在仓库根目录跑",
        "   ```",
        "   python -X utf8 deployment\\migrate_model_folders.py",
        "   ```",
        "   它会按已知目录名补写 model.json，幂等，不动权重。",
        "2. **手动合并**：把本补丁里各文件夹的 `model.json` 拷到",
        "   `<仓库>\\vecrot\\models\\<同名文件夹>\\` 下。",
        "3. **重新下载完整包**（见另一个网盘文件夹），最省事。",
        "",
        "## 本补丁包含",
        "",
        "| 文件夹 | key | 后端 |",
        "|---|---|---|",
    ]
    lines += [f"| `{folder}` | `{key}` | `{backend}` |" for folder, key, backend in rows]
    lines += [
        "",
        "> `detector_rio_gzda` 还需要 RiO-DETRv2 源码与独立 conda 环境",
        "> （`vecrot-rio`，torch 2.2.2 + timm + cv2/numba/pycocotools），",
        "> 见 docs/安装教程-自建环境.md §5.6。",
        "",
    ]
    write_text(os.path.join(patch, "README-补丁说明.md"), "\n".join(lines))
    return patch


def build_full(out_root: str, rio_config: str | None) -> str:
    """完整包骨架：目录结构 + model.json + 说明；权重留空待拷。"""
    full = os.path.join(out_root, "vecrot-models-package")

    # 1) 轮廓资产：只建目录 + 说明
    for name, (desc, files) in ASSET_DIRS.items():
        write_text(os.path.join(full, name, "放权重.txt"),
                   f"{desc}\n\n本目录应包含：\n"
                   + "".join(f"  - {f}\n" for f in files)
                   + "\n从你现有的 models 包里原样拷过来（内容没变）。\n")

    # 2) 检测模型：老的三项 + 新的两项
    for folder, spec in LEGACY.items():
        write_json(os.path.join(full, folder, mr.SPEC_FILENAME), spec)
        weight = spec.get("weights")
        if weight and weight != ".":
            write_text(os.path.join(full, folder, "放权重.txt"),
                       f"{EXTRA_NOTES.get(folder, '')}\n\n"
                       f"把权重 `{weight}` 原样拷进本目录（内容没变）。\n"
                       f"配置 `{spec.get('config')}` 指向源码树，不需要拷。\n")
        else:
            write_text(os.path.join(full, folder, "放权重.txt"),
                       f"{EXTRA_NOTES.get(folder, '')}\n\n"
                       "HF 格式：把整个模型目录的三个文件拷进本目录\n"
                       "  config.json / preprocessor_config.json / model.safetensors\n")

    for folder, spec in NEW_MODELS.items():
        write_json(os.path.join(full, folder, mr.SPEC_FILENAME), spec)
        write_text(os.path.join(full, folder, "放权重.txt"),
                   NEW_WEIGHT_HINTS[folder] + "\n")

    # 3) RiO 的单文件配置（自包含）
    rio_dir = os.path.join(full, "detector_rio_gzda")
    if rio_config:
        cfg_path = os.path.normpath(
            rio_config if os.path.isabs(rio_config) else os.path.join(REPO, rio_config))
        if not os.path.isfile(cfg_path):
            raise SystemExit(f"--rio-config 不存在：{cfg_path}")
        sys.path.insert(0, HERE)
        from flatten_rio_config import load_flattened

        cfg = load_flattened(cfg_path)
        for key in ("optimizer", "lrsheduler", "lr_gamma", "warmup_iter",
                    "lr_warmup_scheduler", "flat_epoch", "no_aug_epoch",
                    "clip_max_norm", "use_amp", "use_ema", "ema", "scaler",
                    "train_dataloader", "val_dataloader", "sync_bn",
                    "find_unused_parameters", "criterion", "epoches",
                    "checkpoint_freq", "print_freq", "output_dir"):
            cfg.pop(key, None)
        import yaml

        with open(os.path.join(rio_dir, "riov2_x_gzda_infer.yaml"), "w",
                  encoding="utf-8") as f:
            f.write("# 由 deployment/build_models_package.py 展开自：\n")
            f.write(f"#   {cfg_path}\n")
            f.write("# include 已内联；训练期字段已删除，推理只需要 model/postprocessor。\n")
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, width=120)

    # 4) 包说明
    readme = """# vecrot 模型包（文件夹约定版）

解压后应正好是仓库里的 `vecrot/models/`：

```text
<仓库根>/vecrot/models/     ← 本目录的内容
├─ adaptor_sc_055m/
├─ official_vocab_635/
├─ base_v1/                 （可选）
├─ detector_strip_s_scene/
├─ detector_strip_s/
├─ detector_rt-detr_bbox/
├─ detector_strip_s_fusion/      ← 新增
└─ detector_rio_gzda/            ← 新增
```

每个**检测模型文件夹**里都有一份 `model.json`，面板的「模型搭配」下拉框
就是按它列的（一个文件夹 = 一项）。加/换模型只要建文件夹，不用改代码。

## 打包前检查清单（维护者）

| 目录 | 需要放进去的权重 | 备注 |
|---|---|---|
| `adaptor_sc_055m/` | `trainable.pt` + tokenizer/arch | 未变，原样拷 |
| `official_vocab_635/` | `model.safetensors` 约 1.5 GB + 3 个 json | 未变，原样拷 |
| `detector_strip_s_scene/` | `best_mAP_iter_37620.pth` 约 540 MB | 未变 + 新增 model.json |
| `detector_strip_s/` | `best_mAP_iter_90288.pth` 约 540 MB | 未变 + 新增 model.json |
| `detector_rt-detr_bbox/` | HF 三个文件（safetensors 约 306 MB） | 未变 + 新增 model.json |
| `detector_strip_s_fusion/` | `best.pth`（融合数据集重训权重） | **新**，名字对不上就改 model.json |
| `detector_rio_gzda/` | `last.pth`（RiO-DETRv2-X，约 1.0 GB） | **新**，配置已随包 |
| `base_v1/` | 可选，1.7 GB | 只影响“重建基座”，可不上传 |

放完之后在仓库根跑这两条确认无误，再压缩上传：

```powershell
python -X utf8 deployment\\verify_models.py
python -X utf8 deployment\\add_model_folder.py --list
```

## 用户侧怎么装

解压本目录到 `<仓库根>\\vecrot\\models\\`（不要形成 `vecrot/models/models/`），
然后 `python install.py`。老用户若已有旧模型包，只需下载「增量补丁」
或跑一次 `python deployment\\migrate_model_folders.py`。

## 别忘的两件事

1. **源码不在这个包里**：`third_party/Strip-R-CNN` 与 `third_party/RiO-DETRv2`
   随 Git 仓库分发。跑 RiO-DETR 还额外需要 conda 环境 `vecrot-rio`
   （torch 2.2.2 + timm + cv2/numba/pycocotools）。
2. **RiO-DETRv2 是 AGPL-3.0**：若要随包/随仓库分发它的源码，请遵守其许可
   （保留 LICENSE 与来源说明）。最稳的做法是让用户自己
   `git clone https://github.com/RicePasteM/RiO-DETRv2` 到 `third_party/RiO-DETRv2/`。
"""
    write_text(os.path.join(full, "README-模型包.md"), readme)
    return full


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成网盘模型包目录骨架")
    ap.add_argument("--out", required=True, help="输出根目录（会建两个子目录）")
    ap.add_argument("--rio-config", default=None,
                    help="RiO-DETR 训练用的 .yml（会被展开成单文件放进模型文件夹）")
    ap.add_argument("--patch-only", action="store_true", help="只生成增量补丁")
    ap.add_argument("--full-only", action="store_true", help="只生成完整包骨架")
    ap.add_argument("--zip", action="store_true", help="顺手压成 zip")
    args = ap.parse_args(argv)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    made = []
    if not args.full_only:
        made.append(build_patch(out))
    if not args.patch_only:
        made.append(build_full(out, args.rio_config))

    for path in made:
        count = sum(len(files) for _r, _d, files in os.walk(path))
        print(f"已生成 {path}（{count} 个文件）")
    if args.zip:
        for path in made:
            archive = shutil.make_archive(path, "zip", root_dir=path)
            print(f"已压缩 {archive}（{os.path.getsize(archive) / 1024:.0f} KB）")
    print("\n下一步：把权重拷进 vecrot-models-package 各目录 → 压缩上传网盘。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
