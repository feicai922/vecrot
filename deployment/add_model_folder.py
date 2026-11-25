#!/usr/bin/env python
"""把一个权重 + 配置接成一个模型文件夹（生成 model.json）。

    # 融合数据集重训的 Strip-R-CNN-S
    python deployment/add_model_folder.py --name strip_s_fusion \\
        --backend strip_s_gzda \\
        --weights "G:/gzda/best_mAP_iter_12345.pth" \\
        --config third_party/Strip-R-CNN/configs/gzda/strip_s_gzda.py \\
        --label "融合数据集 · Strip-R-CNN-S（旋转框）"

    # RiO-DETRv2
    python deployment/add_model_folder.py --name rio_gzda --backend rio_detr \\
        --weights "G:/rio/last.pth" \\
        --config third_party/RiO-DETRv2/configs/rio-gzda/riov2_hgnetv2_x_gzda_12e.yml \\
        --label "RiO-DETRv2-X（融合数据）" --score-thr 0.4 --order 30

    python deployment/add_model_folder.py --list      # 看现有文件夹与就绪状态

做的事：把权重**复制**进 ``vecrot/models/<name>/``（``--no-copy`` 则写绝对路径），
配置默认用 ``src:`` 前缀指向源码树（也可 ``--copy-config`` 复制进文件夹），
然后写 ``model.json``，最后用注册表回读一遍确认「能被扫出来且文件齐备」。

为什么用脚本而不是让人手写 JSON：路径前缀（``src:`` / ``repo:`` / ``models:``）
和必需字段很容易记错，错了的表现是"面板里没有这个模型"，不好排查。
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

from backend.contracts import model_registry as mr   # noqa: E402

#: 各后端默认源码树 + 是否必需 config
BACKEND_DEFAULTS = {
    "strip_s_gzda": {"repo": "third_party/Strip-R-CNN", "config": True},
    "mmrotate_obb": {"repo": "third_party/Strip-R-CNN", "config": True},
    "mmdet_hbb": {"repo": None, "config": True},
    "rtdetr_hf": {"repo": None, "config": False},
    "rio_detr": {"repo": "third_party/RiO-DETRv2", "config": True},
}


def _to_repo_relative(path: str) -> str | None:
    """把仓库内的绝对/相对路径转成 ``src:`` 或 ``repo:`` 写法，返回声明串。"""
    full = os.path.normpath(path if os.path.isabs(path) else os.path.join(REPO, path))
    try:
        rel = os.path.relpath(full, REPO)
    except ValueError:            # Windows 跨盘符
        return None
    if rel.startswith(".."):
        return None
    return "repo:" + rel.replace(os.sep, "/")


def _config_decl(path: str, repo_dir: str | None, folder: str,
                 copy_config: bool) -> str:
    """决定 config 字段怎么写。

    优先 ``src:``（相对源码树，最贴合"配置随源码走"的老布局）；
    在源码树里找不到就复制进模型文件夹（自包含）。
    """
    full = os.path.normpath(path if os.path.isabs(path) else os.path.join(REPO, path))
    if copy_config:
        dst = os.path.join(folder, os.path.basename(full))
        if os.path.abspath(dst) != os.path.abspath(full):
            shutil.copy2(full, dst)
            print(f"  复制配置 → {dst}")
            print("  注意：若这份配置里有 _base_/__include__ 相对路径，"
                  "复制后会失效，请改成绝对路径或改用 --no-copy-config")
        return os.path.basename(full)
    if repo_dir:
        try:
            rel = os.path.relpath(full, repo_dir)
            if not rel.startswith(".."):
                return "src:" + rel.replace(os.sep, "/")
        except ValueError:
            pass
    decl = _to_repo_relative(full)
    if decl:
        return decl
    return full            # 仓库外 → 绝对路径


def list_folders() -> int:
    root = mr.resolve_models_root(REPO, None)
    presets, skipped = mr.scan(root, REPO)
    print(f"models 根：{root}\n")
    for p in presets:
        miss = p.missing()
        print(f"[{'✓' if not miss else '✗'}] {p.key}  ({p.backend})")
        print(f"     {p.label}")
        if miss:
            for item in miss:
                print(f"     缺：{item}")
    for item in skipped:
        print(f"[!] {os.path.basename(item['folder'])}: {item['reason']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成一个模型文件夹（model.json）")
    ap.add_argument("--list", action="store_true", help="列出已有模型文件夹")
    ap.add_argument("--name", help="文件夹名（也是默认 key）")
    ap.add_argument("--backend", help="strip_s_gzda / mmrotate_obb / mmdet_hbb / rtdetr_hf / rio_detr")
    ap.add_argument("--weights", help="权重文件路径（或 HF 模型目录）")
    ap.add_argument("--config", help="配置文件路径（.py / .yml）")
    ap.add_argument("--label", help="面板上显示的名字（默认用文件夹名）")
    ap.add_argument("--num-classes", type=int, default=None)
    ap.add_argument("--score-thr", type=float, default=None, help="后端打分阈值")
    ap.add_argument("--nms-iou", type=float, default=None)
    ap.add_argument("--scene-threshold", type=float, default=None,
                    help="只有带 scene_head 的 Strip-R-CNN 生效")
    ap.add_argument("--repo", default=None, help="源码树目录（默认按 backend 推断）")
    ap.add_argument("--order", type=int, default=100, help="下拉排序，越小越前")
    ap.add_argument("--note", default="", help="面板上显示的一句说明")
    ap.add_argument("--no-copy", action="store_true",
                    help="不复制权重，model.json 里写绝对路径")
    ap.add_argument("--copy-config", action="store_true",
                    help="把配置也复制进模型文件夹（默认用 src: 指向源码树）")
    args = ap.parse_args(argv)

    if args.list:
        return list_folders()
    missing = [f for f in ("name", "backend", "weights") if not getattr(args, f)]
    if missing:
        ap.error("缺少参数：" + "、".join("--" + m for m in missing))
    if args.backend not in mr.BACKEND_REQUIRED_FILES:
        ap.error(f"未知 backend {args.backend!r}（可选 {sorted(BACKEND_REQUIRED_FILES)}）")

    defaults = BACKEND_DEFAULTS.get(args.backend, {})
    repo_dir = None
    repo_decl = args.repo or defaults.get("repo")
    if repo_decl:
        repo_dir = os.path.normpath(
            repo_decl if os.path.isabs(repo_decl) else os.path.join(REPO, repo_decl))
        if not os.path.isdir(repo_dir):
            print(f"⚠ 源码树不存在：{repo_dir}\n"
                  f"  {args.backend} 需要它；请先放到该位置或用 --repo 指定")

    root = mr.resolve_models_root(REPO, None)
    folder = os.path.join(root, args.name)
    weights_src = os.path.normpath(
        args.weights if os.path.isabs(args.weights) else os.path.join(REPO, args.weights))
    if not os.path.exists(weights_src):
        ap.error(f"权重不存在：{weights_src}")
    os.makedirs(folder, exist_ok=True)

    spec: dict = {
        "spec_version": 1,
        "key": args.name,
        "label": args.label or args.name,
        "backend": args.backend,
        "order": int(args.order),
        "enabled": True,
    }
    if args.num_classes:
        spec["num_classes"] = int(args.num_classes)
    if repo_decl:
        spec["repo"] = ("repo:" + os.path.relpath(repo_dir, REPO).replace(os.sep, "/")
                        if os.path.isdir(repo_dir) and _is_under(repo_dir, REPO)
                        else repo_dir)

    # 权重
    if args.backend == "rtdetr_hf" and os.path.isdir(weights_src):
        # 目录即模型：整目录复制或直接写绝对路径
        if args.no_copy:
            spec["weights"] = weights_src
            print(f"  权重目录（绝对路径）→ {weights_src}")
        else:
            dst = os.path.join(folder, os.path.basename(weights_src.rstrip("\\/")))
            if os.path.abspath(dst) != os.path.abspath(weights_src):
                print(f"  复制权重目录 → {dst}（{_human_dir(weights_src)}）")
                shutil.copytree(weights_src, dst, dirs_exist_ok=True)
            spec["weights"] = "."
    else:
        if args.no_copy:
            spec["weights"] = weights_src
            print(f"  权重（绝对路径）→ {weights_src}")
        else:
            dst = os.path.join(folder, os.path.basename(weights_src))
            if os.path.abspath(dst) != os.path.abspath(weights_src):
                size = os.path.getsize(weights_src) / (1024 ** 2)
                print(f"  复制权重 → {dst}（{size:.0f} MB）")
                shutil.copy2(weights_src, dst)
            spec["weights"] = os.path.basename(weights_src)

    # 配置
    need_config = defaults.get("config", True)
    if args.config:
        cfg_full = os.path.normpath(
            args.config if os.path.isabs(args.config) else os.path.join(REPO, args.config))
        if not os.path.isfile(cfg_full):
            ap.error(f"配置不存在：{cfg_full}")
        spec["config"] = _config_decl(cfg_full, repo_dir, folder, args.copy_config)
    elif need_config:
        ap.error(f"{args.backend} 需要 --config（.py 或 .yml）")

    # 阈值
    opts = {}
    for key, value in (("score_thr", args.score_thr), ("nms_iou", args.nms_iou),
                       ("scene_threshold", args.scene_threshold)):
        if value is not None:
            opts[key] = float(value)
    if opts:
        spec["detector_options"] = opts
    if args.note:
        spec["note"] = args.note

    path = os.path.join(folder, mr.SPEC_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\n已写入 {path}")

    print("\n=== 回读校验 ===")
    ok, messages = mr.validate_folder(folder, REPO)
    for msg in messages:
        print(("  ✓ " if ok else "  ✗ ") + msg)
    if not ok:
        print("\n按上面的提示补齐文件；面板里暂时不会出现这个模型。")
        return 1
    preset = mr.load_spec(folder, REPO, root)
    print(f"\n面板将显示：{preset.label}  [{preset.backend}]")
    for role, p in preset.files.items():
        print(f"  {role}: {p}")
    print("\n重启 QGIS 后在「模型搭配」下拉框里选它。")
    return 0


def _is_under(path: str, root: str) -> bool:
    try:
        return not os.path.relpath(path, root).startswith("..")
    except ValueError:
        return False


def _human_dir(path: str) -> str:
    total = 0
    for r, _d, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return f"{total / 1024 ** 2:.0f} MB"


if __name__ == "__main__":
    sys.exit(main())
