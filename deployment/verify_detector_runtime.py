"""检测后端实机验证（需要 detector 环境）。

    deployment/envs/detector/python.exe deployment/verify_detector_runtime.py

依次验证：
1. 依赖与 CUDA
2. mmcv 旋转算子（nms_rotated / box_iou_rotated / roi_align_rotated）真实调用
3. 派生纯推理配置（从本地 gzda 出发）
4. build_detector + 加载 541MB 权重，报告 missing/unexpected
5. 单窗口前向：合成 1024² 影像 → 输出旋转框
6. T400 4GB 的峰值显存
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((bool(ok), msg))
    print(("  [OK]   " if ok else "  [FAIL] ") + msg)


def _load_detector_section() -> dict:
    """取 manifest 的 models.detector。

    分发包里默认只有 ``manifest.template.json``（python 路径是占位）。
    本脚本其实**不需要**解释器路径（它自己就是在 detector 环境里跑的），
    只需要权重/配置/源码的相对路径。所以这里退化为读模板 —— 让用户
    在填 manifest 之前就能验证环境是否装好。
    """
    for name in ("manifest.json", "manifest.template.json"):
        path = os.path.join(HERE, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        det = (m.get("models") or {}).get("detector")
        if det:
            if name != "manifest.json":
                print("（提示：用的是 manifest.template.json）")
            return det
    raise SystemExit(
        "找不到 manifest.json 或 manifest.template.json；"
        "请确认 deployment/ 目录完整")


def main() -> int:
    from backend.detector.strip_rcnn_backend import StripRcnnDetector, import_mmrotate

    det = _load_detector_section()

    def resolve(rel):
        return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(REPO, rel))

    repo = resolve(det["repo"])
    weights = resolve(det["weights"])
    config = resolve(det["config"])

    print("=== 1. 依赖 ===")
    import torch
    check(True, f"torch {torch.__version__} / cuda {torch.version.cuda}")
    check(torch.cuda.is_available(), "CUDA 可用")
    props = torch.cuda.get_device_properties(0)
    check(True, f"{props.name} / {props.total_memory/1024**3:.1f} GB / sm_{props.major}{props.minor}")

    print("\n=== 2. mmcv 旋转算子真实调用 ===")
    from mmcv.ops import box_iou_rotated, nms_rotated, roi_align_rotated

    dev = torch.device("cuda:0")
    boxes = torch.tensor([[50., 50., 40., 20., 0.1],
                          [52., 51., 40., 20., 0.1],
                          [200., 200., 30., 30., 0.5]], device=dev)
    scores = torch.tensor([0.9, 0.8, 0.7], device=dev)
    try:
        iou = box_iou_rotated(boxes, boxes)
        check(True, f"box_iou_rotated 在 GPU 上运行，对角线={iou.diag().tolist()}")
        check(abs(float(iou[0, 0]) - 1.0) < 1e-5, "自 IoU = 1")
        check(float(iou[0, 1]) > 0.5, f"相邻框 IoU={float(iou[0,1]):.3f} > 0.5")
        # mmcv 的 nms_rotated 返回 (dets, keep)：dets 是保留的框，keep 是索引
        dets, keep = nms_rotated(boxes, scores, 0.3)
        n_keep = int(dets.shape[0])
        check(n_keep == 2, f"nms_rotated 保留 {n_keep} 个（前两个重叠应被抑制为一个）")
        check(keep.tolist() == [0, 2], f"NMS 保留索引 {keep.tolist()}")
        feats = torch.randn(1, 1, 32, 32, device=dev)
        rois = torch.tensor([[0, 16., 16., 10., 6., 0.3]], device=dev)
        pooled = roi_align_rotated(feats, rois, (7, 7), 1.0, 2, True)
        check(tuple(pooled.shape) == (1, 1, 7, 7), f"roi_align_rotated 输出 {tuple(pooled.shape)}")
    except Exception as exc:                    # noqa: BLE001
        check(False, f"旋转算子失败：{type(exc).__name__}: {exc}")
        return 1

    print("\n=== 3. 导入 mmrotate（含兼容补丁）===")
    mm, patches = import_mmrotate(repo)
    check(True, f"mmrotate {mm.__version__}")
    for p in patches:
        check(True, p)
    from mmrotate.models.backbones import StripNet
    from mmrotate.models.detectors import StripRCNN

    check(True, f"StripRCNN={StripRCNN.__name__} / StripNet={StripNet.__name__}")

    print("\n=== 4. 派生推理配置 + 构建模型 ===")
    detector = StripRcnnDetector({
        "strip_rcnn_repo": repo,
        "detector_weights": weights,
        "detector_config": config,
        "device": "cuda:0",
        "derived_dir": os.path.join(REPO, "work", "detector_check", "derived"),
    })
    t0 = time.time()
    try:
        info = detector.prepare()
    except Exception as exc:                    # noqa: BLE001
        import traceback

        traceback.print_exc()
        check(False, f"prepare() 失败：{type(exc).__name__}: {exc}")
        return 1
    elapsed = time.time() - t0
    check(True, f"模型加载完成（{elapsed:.1f}s）")

    deriv = info["config_derivation"]
    check(deriv["derivation_source"] == "local_gzda",
          f"派生来源：{deriv['derivation_source']}")
    check(os.path.isfile(deriv.get("derived_config", "")),
          f"派生配置已写出：{deriv.get('derived_config')}")
    check(info["num_classes"] == 1, "num_classes=1")
    check(deriv["overrides"].get("num_classes") == 1, "派生覆盖 num_classes=1")
    check("BN" in str(deriv["overrides"].get("norm_cfg")), f"norm_cfg：{deriv['overrides'].get('norm_cfg')}")

    loading = info["weight_loading"]
    print(f"     missing={len(loading['missing_keys'])} unexpected={len(loading['unexpected_keys'])}")
    if loading["missing_keys"][:5]:
        print(f"     missing 示例：{loading['missing_keys'][:5]}")
    if loading["unexpected_keys"][:5]:
        print(f"     unexpected 示例：{loading['unexpected_keys'][:5]}")
    check(loading["matched_all_structural"],
          "权重结构全部匹配（无 serious missing key）")

    print("\n=== 5. 单窗口前向 ===")
    import numpy as np

    # 合成一张 1024² 影像：白底 + 几个深色矩形（模拟建筑）
    img = np.full((1024, 1024, 3), 210, dtype=np.uint8)
    for (x0, y0, w, h) in ((120, 150, 90, 60), (400, 300, 120, 80), (700, 600, 70, 100)):
        img[y0:y0 + h, x0:x0 + w] = 60
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    try:
        boxes = detector.predict_windows([{"window_id": 0, "rgb": img}])[0]
    except Exception as exc:                    # noqa: BLE001
        import traceback

        traceback.print_exc()
        check(False, f"前向失败：{type(exc).__name__}: {exc}")
        return 1
    dt = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 1024**3
    check(True, f"前向完成（{dt:.2f}s），T400 峰值显存 {peak:.2f} GB")
    check(isinstance(boxes, list), f"返回 {len(boxes)} 个候选框")
    for b in boxes[:5]:
        print(f"     框: cx={b.cx:.1f} cy={b.cy:.1f} w={b.w:.1f} h={b.h:.1f} "
              f"ang={b.angle:.3f} score={b.score:.3f}")
    # 合成图是纯色块，模型未必检出；这里不断言数量，只报告
    if boxes:
        check(True, "至少检出一个框（合成图上的真实模型输出）")
        scores = [b.score for b in boxes]
        check(all(0 <= s <= 1 for s in scores), "分数在 [0,1]")
        check(all(b.w > 0 and b.h > 0 for b in boxes), "宽高为正")
    else:
        print("     （合成图未检出：纯色矩形与真实建筑差异大，属正常；"
              "真实影像上再验证召回）")

    detector.close()
    failed = sum(1 for ok, _ in results if not ok)
    print(f"\n{'全部通过' if failed == 0 else f'{failed} 项未通过'}（共 {len(results)} 项）")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
