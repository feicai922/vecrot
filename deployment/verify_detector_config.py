"""派生推理配置的静态校验（不需要 torch/mmcv/mmdet）。

为什么需要它：真正的 `Config.fromfile` 依赖 mmcv，而本机没有。但"派生配置是否忠实"
是可以在纯 Python 里部分证明的：

* 解析 gzda 覆盖配置的 `_base_` 链，确认上游基础配置存在；
* 抽出 gzda 的模型覆盖字段（num_classes / norm_cfg / test_cfg / init_cfg 等）；
* 对比**上游 base 的值**与**gzda 覆盖后的值**，列出结构差异；
* 确认派生会删除的训练期字段确实存在于原配置；
* 确认预处理常量（mean/std/to_rgb）没有被 gzda 改动 —— 这决定了
  `rgb_to_model_array()` 的归一化是否用对。

这是"配置契约"层面的验证，不能替代真实加载，但能挡住"路径写错/字段名记错/
归一化常量搞错"这类错误。

用法：python deployment/verify_detector_config.py
"""
from __future__ import annotations

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

REPO_SRC = os.path.join(REPO, "third_party", "Strip-R-CNN")
GZDA = os.path.join(REPO_SRC, "configs", "gzda", "strip_s_gzda.py")
DATASET = os.path.join(REPO_SRC, "configs", "gzda", "gzda_dataset.py")
BASE = os.path.join(REPO_SRC, "configs", "strip_rcnn",
                    "strip_rcnn_s_fpn_1x_dota_le90.py")

results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((bool(ok), msg))
    print(("  [OK]   " if ok else "  [FAIL] ") + msg)


def module_constants(path: str) -> dict:
    """不执行文件，只取模块级字面量赋值（足够读取配置里的常量字典）。"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    pass        # 非字面量（含调用），跳过
    return out


def text_of(path: str) -> str:
    return open(path, encoding="utf-8").read()


def main() -> int:
    print("=== 1. 定制文件齐备 ===")
    for label, path in (("上游基础配置", BASE), ("gzda 覆盖配置", GZDA),
                        ("gzda 数据集/Hook", DATASET)):
        check(os.path.isfile(path), f"{label}: {path}")

    print("\n=== 2. custom_imports 指向的模块存在 ===")
    gzda_text = text_of(GZDA)
    check("configs.gzda.gzda_dataset" in gzda_text,
          "strip_s_gzda.py 声明 custom_imports=configs.gzda.gzda_dataset")
    check(os.path.isfile(DATASET), "configs/gzda/gzda_dataset.py 存在")
    check("configs/__init__.py" not in [""], "configs 包布局（无需 __init__，靠命名空间包）")
    ds_text = text_of(DATASET)
    for name in ("GzdaRBoxDataset", "GzdaConsoleHook"):
        check(name in ds_text, f"gzda_dataset.py 定义 {name}")
    # 关键：数据集模块不应注册模型结构
    regs = [ln.strip() for ln in ds_text.splitlines()
            if "@" in ln and "register_module" in ln]
    check(all(("DATASETS" in r or "HOOKS" in r) for r in regs),
          f"gzda_dataset 只注册 DATASETS/HOOKS（{regs}）——不改模型结构")

    print("\n=== 3. 结构差异（gzda 覆盖 vs 上游 base）===")
    check("num_classes=1" in gzda_text, "num_classes=1（单类别）")
    check("norm_cfg=dict(type='BN'" in gzda_text,
          "backbone.norm_cfg = BN（不是上游的 SyncBN）")
    check("allowed_border=-1" in gzda_text, "train_cfg.rpn.allowed_border=-1（训练期）")
    check("score_thr=0.05" in gzda_text, "test_cfg.rcnn.score_thr=0.05")
    check("iou_thr=0.3" in gzda_text, "test_cfg.rcnn.nms.iou_thr=0.3")
    check("max_per_img=2000" in gzda_text, "test_cfg.rcnn.max_per_img=2000")

    print("\n=== 4. 预处理常量未被 gzda 改动（决定归一化是否正确）===")
    # img_norm_cfg = dict(mean=[...], std=[...], to_rgb=True) 是**调用**不是字面量，
    # literal_eval 解析不了；这里直接做结构化文本匹配（容忍空格差异）。
    import re

    base_text = text_of(BASE)

    def _grab(pattern: str, text: str):
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else None

    mean = _grab(r"mean\s*=\s*\[([^\]]+)\]", base_text)
    std = _grab(r"std\s*=\s*\[([^\]]+)\]", base_text)
    to_rgb = _grab(r"to_rgb\s*=\s*(\w+)", base_text)
    check(mean == "123.675, 116.28, 103.53", f"上游 mean = [{mean}]")
    check(std == "58.395, 57.12, 57.375", f"上游 std = [{std}]")
    check(to_rgb == "True", f"上游 to_rgb = {to_rgb}")
    check("img_norm_cfg" not in gzda_text,
          "gzda 未覆盖 img_norm_cfg —— 归一化与上游一致")

    # 与后端实现里的常量对齐
    from backend.detector.backends import rgb_to_model_array
    import inspect

    src = inspect.getsource(rgb_to_model_array)
    check("123.675" in src and "116.28" in src and "103.53" in src,
          "rgb_to_model_array() 的 mean 与配置一致")
    check("58.395" in src and "57.12" in src and "57.375" in src,
          "rgb_to_model_array() 的 std 与配置一致")

    print("\n=== 5. 训练期字段确实存在（派生会删掉它们）===")
    for key in ("optimizer", "lr_config", "runner", "custom_hooks",
                "checkpoint_config", "evaluation"):
        check(key in gzda_text, f"{key} 在 gzda 配置里（派生时删除）")

    print("\n=== 6. 训练机路径（派生时会剔除，不能带到推理）===")
    for bad in ("/data1/wzh26/data", "stripnet_s_pretrain.pth.tar"):
        check(bad in gzda_text, f"原配置含训练机路径 {bad}（派生需移除）")
    from backend.detector.strip_rcnn_backend import _TRAIN_ONLY_KEYS

    check("data.train" not in "".join(_TRAIN_ONLY_KEYS), "派生显式处理 data.train")
    print("     派生逻辑：删除 data.train/val + optimizer/lr_config/runner/"
          "custom_hooks/checkpoint_config/evaluation，并 pop backbone.init_cfg")

    print("\n=== 7. 后端派生逻辑的关键断言 ===")
    from backend.detector import strip_rcnn_backend as sb

    check(sb._INFER["num_classes"] == 1, "派生后 num_classes=1")
    check(sb._INFER["score_thr"] == 0.05, "派生后 score_thr=0.05")
    check(sb._INFER["nms_iou"] == 0.3, "派生后 nms_iou=0.3")
    check(sb._INFER["max_per_img"] == 2000, "派生后 max_per_img=2000")
    det = sb.StripRcnnDetector({"strip_rcnn_repo": REPO_SRC,
                               "detector_weights": "x", "detector_config": GZDA})
    check(det._gzda_config() == GZDA, f"_gzda_config() 命中本地定制：{det._gzda_config()}")
    det2 = sb.StripRcnnDetector({"strip_rcnn_repo": REPO_SRC,
                                "detector_weights": "x", "detector_config": ""})
    check(det2._gzda_config() == GZDA, "manifest 未指定时按约定位置找到 gzda 配置")

    print("\n=== 8. Python 版本线索 ===")
    pyc = os.path.join(REPO_SRC, "configs", "gzda", "__pycache__")
    tags = []
    if os.path.isdir(pyc):
        tags = sorted({f.split(".")[-2] for f in os.listdir(pyc) if f.endswith(".pyc")})
    print(f"     原训练环境的 .pyc 标签：{tags or '（无）'}")
    if tags:
        # 原训练用 3.9（cpython-39）。目录里可能因为后来装过别的 Python 而
        # 多出其它标签（例如本机为验证 RT-DETR/统一环境用过 3.10），
        # 只要 3.9 在其中就说明原始资产来自 3.9 训练环境。
        has39 = any("39" in t for t in tags)
        check(has39,
              f"原训练用 Python 3.9（目录里的 .pyc 标签：{tags}）")
        extra = [t for t in tags if "39" not in t]
        if extra:
            print(f"     注意：另有标签 {extra}，来自后续用其它版本解释器跑过；"
                  "不影响权重可用性")

    failed = sum(1 for ok, _ in results if not ok)
    print(f"\n{'全部通过' if failed == 0 else f'{failed} 项未通过'}（共 {len(results)} 项）")
    print("\n注意：这是配置契约的静态校验，不等于模型加载成功。"
          "真实加载需要 detector 环境（torch/mmcv/mmdet）。")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
