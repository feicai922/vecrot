#!/usr/bin/env python
"""生成「网盘模型包」目录骨架（含 model.json），方便整包压缩上传。

    python deployment/build_models_package.py --out "D:/上传/vecrot-models"
    python deployment/build_models_package.py --out "D:/上传/vecrot-models" \\
        --rio-config third_party/RiO-DETRv2/configs/rio-gzda/riov2_hgnetv2_x_gzda_12e.yml

产出**一个**目录 ``<out>/models/``：解压后就是用户机器上的 ``vecrot/models/``。

脚本只写 ``model.json`` / RiO 单文件配置 / 各目录的 ``放权重.txt`` 说明，
**权重由你从本机拷进来**（脚本不碰几 GB 的权重），拷完压成 ``models.zip`` 上传网盘。

项目只有一条分发路线：**GitHub 给代码，网盘给模型**。用户 ``git clone`` 之后
把 ``models.zip`` 解压到 ``vecrot/models/``（或用 manifest 的 ``models_root``
指到别的盘），环境自己配好，装完插件就能用。没有"老用户补丁"这种东西。

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


def build_full(out_root: str, rio_config: str | None) -> str:
    """模型包骨架：目录结构 + model.json + 说明；权重留空待拷。"""
    full = os.path.join(out_root, "models")
    if os.path.isdir(full):
        shutil.rmtree(full)

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
    readme = """# vecrot 模型包

解压后得到 `models/` 目录，把它放到 `vecrot/models/`（或 `VECROT_MODELS_ROOT`
/manifest `models_root` 指向的位置）。面板的「模型搭配」下拉框就是按这里的
子目录列的：**一个文件夹 = 一个可选模型**。

```text
models/
├─ adaptor_sc_055m/          轮廓适配器
├─ official_vocab_635/       轮廓基座
├─ base_v1/                  可选：重建基座原料
├─ detector_strip_s_scene/   Strip-R-CNN-S + 场景分类（默认推荐）
├─ detector_strip_s/         不带场景分类的旧权重
├─ detector_rt-detr_bbox/    RT-DETR（HF 格式，水平框）
├─ detector_strip_s_fusion/  融合数据集重训的 Strip-R-CNN-S
└─ detector_rio_gzda/        RiO-DETRv2-X（旋转框）
```

## 发布前清单（维护者）

权重不在生成脚本里，按下面的表拷进对应目录：

| 目录 | 放什么 | 体积 |
|---|---|---|
| `adaptor_sc_055m/` | `trainable.pt` + tokenizer/arch | 小 |
| `official_vocab_635/` | `model.safetensors` + 3 个 json | 约 1.5 GB |
| `detector_strip_s_scene/` | `best_mAP_iter_37620.pth` | 约 540 MB |
| `detector_strip_s/` | `best_mAP_iter_90288.pth` | 约 540 MB |
| `detector_rt-detr_bbox/` | `config.json` + `preprocessor_config.json` + `model.safetensors` | 约 306 MB |
| `detector_strip_s_fusion/` | `best.pth`（融合数据重训；名字不同就改 model.json） | 约 540 MB |
| `detector_rio_gzda/` | `last.pth`（RiO-DETRv2-X；建议用 EMA 权重） | 约 1.0 GB |
| `base_v1/` | 可选，只影响“重建基座” | 约 1.7 GB |

拷完在仓库根确认，再压缩上传：

```powershell
python -X utf8 deployment\\verify_models.py
python -X utf8 deployment\\add_model_folder.py --list
```

## 用户侧

```
<仓库根>/vecrot/models/     ← 本目录内容
python install.py
```

## 源码不在这个包里

`third_party/Strip-R-CNN`、`third_party/RiO-DETRv2` 随 GitHub 分发。
跑 RiO-DETR 还需要独立 conda 环境 `vecrot-rio`（torch 2.2.2 + timm +
cv2/numba/pycocotools），见 docs/安装教程-自建环境.md §5.6。

RiO-DETRv2 上游是 AGPL-3.0，其 LICENSE 原样保留在
`third_party/RiO-DETRv2/LICENSE`，详见该目录的 `VECROT_NOTICE.md`。
"""
    write_text(os.path.join(full, "README-模型包.md"), readme)
    return full


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成网盘模型包目录（models/）")
    ap.add_argument("--out", required=True, help="输出根目录；脚本在其中建 models/")
    ap.add_argument("--rio-config", default=None,
                    help="RiO-DETR 训练用的 .yml（会被展开成单文件放进模型文件夹）")
    ap.add_argument("--zip", action="store_true", help="顺手压成 models.zip")
    args = ap.parse_args(argv)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    models_dir = build_full(out, args.rio_config)
    count = sum(len(files) for _r, _d, files in os.walk(models_dir))
    print(f"已生成 {models_dir}（{count} 个文件）")
    if args.zip:
        archive = shutil.make_archive(os.path.join(out, "models"), "zip",
                                      root_dir=out, base_dir="models")
        print(f"已压缩 {archive}（{os.path.getsize(archive) / 1024:.0f} KB，"
              "解压后得到 models/）")
    print("\n下一步：按 models/README-模型包.md 把权重拷进各目录 → 压缩上传网盘。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
