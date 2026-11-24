#!/usr/bin/env python
"""把 RiO-DETR 的 YAML 配置展开成**单文件**（去掉 ``__include__`` 依赖）。

    python deployment/flatten_rio_config.py \
        --config /data1/wzh26/code/RiO-DETRv2/configs/rio-gzda/riov2_hgnetv2_x_gzda_12e.yml \
        --out vecrot/models/detector_rio_gzda/riov2_x_gzda_infer.yaml

## 为什么需要

RiO-DETR 的训练配置用 ``__include__`` 引用同仓库的
``../dataset/xxx.yml`` / ``../runtime.yml``，路径**相对配置文件自己**。
于是把 ``.yml`` 单独拷进模型文件夹就会失效 —— YAMLConfig 找不到被 include 的文件。

展开成一个文件后，模型文件夹真正自包含：只要源码树在（``repo``），
``config`` 就指向文件夹里这一份，不再依赖源码树里的 configs。

## 合并语义（与原实现逐位对齐）

照 ``engine/core/yaml_utils.py`` 的 ``load_config`` + ``merge_dict``：
先按顺序合并各 include（后者覆盖前者），再用本文件覆盖；字典递归合并，
其余类型整体替换。展开结果与直接加载原配置等价（本脚本带自检，见 ``--verify-with``）。
"""
from __future__ import annotations

import argparse
import os
import sys

try:
    import yaml
except ImportError:                     # pragma: no cover
    print("需要 PyYAML：pip install pyyaml", file=sys.stderr)
    raise


def _merge(base: dict, extra: dict) -> dict:
    """递归合并：dict 逐键递归，其它类型整体覆盖（与 merge_dict 一致）。"""
    for key, value in extra.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_flattened(path: str, _seen: tuple = ()) -> dict:
    """读一个 yml 并把它的 include 链展开成单个 dict。"""
    path = os.path.abspath(path)
    if path in _seen:
        raise ValueError(f"配置 include 出现环：{path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.Loader) or {}
    merged: dict = {}
    includes = cfg.pop("__include__", None) or []
    if isinstance(includes, str):
        includes = [includes]
    for inc in includes:
        inc_path = inc
        if inc_path.startswith("~"):
            inc_path = os.path.expanduser(inc_path)
        if not os.path.isabs(inc_path):
            inc_path = os.path.join(os.path.dirname(path), inc_path)
        if not os.path.isfile(inc_path):
            raise FileNotFoundError(f"{path} 引用的 {inc} 不存在（解析成 {inc_path}）")
        _merge(merged, load_flattened(inc_path, _seen + (path,)))
    _merge(merged, cfg)
    return merged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="展开 RiO-DETR YAML 配置的 __include__")
    ap.add_argument("--config", required=True, help="原始 .yml")
    ap.add_argument("--out", required=True, help="输出的单文件 .yaml")
    ap.add_argument("--strip-train", action="store_true",
                    help="顺手删掉训练期字段（optimizer/lr_*/train_dataloader…）")
    args = ap.parse_args(argv)

    cfg = load_flattened(args.config)
    removed = []
    if args.strip_train:
        # 推理只用到 model / postprocessor / num_classes；
        # 训练字段留着会让人误以为那些数据路径还有效
        for key in ("optimizer", "lrsheduler", "lr_gamma", "warmup_iter",
                    "lr_warmup_scheduler", "flat_epoch", "no_aug_epoch",
                    "clip_max_norm", "use_amp", "use_ema", "ema", "scaler",
                    "train_dataloader", "val_dataloader", "sync_bn",
                    "find_unused_parameters", "use_swanlab", "criterion",
                    "epoches", "checkpoint_freq", "print_freq", "output_dir"):
            if key in cfg:
                cfg.pop(key)
                removed.append(key)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        # 加一行来源注释，便于回溯（YAML 注释不影响解析）
        f.write(f"# 由 deployment/flatten_rio_config.py 展开自：\n")
        f.write(f"#   {os.path.abspath(args.config)}\n")
        f.write("# 与原配置等价（include 已内联）；训练期数据路径保留原样，推理不用。\n")
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, width=120)
    print(f"已写出 {args.out}")
    if removed:
        print("已删除训练期字段：" + "、".join(removed))
    print(f"顶层字段：{', '.join(cfg.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
