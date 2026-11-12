#!/usr/bin/env python
"""分层自检（开发文档 §14）。

    python deployment/selfcheck.py --layer paths
    python deployment/selfcheck.py --layer all

层：paths → import → cuda → model → window → polygon → geopackage → qgis
每层失败都给具体原因，不输出笼统的「环境错误」。
``geometry`` 层不需要 GPU，可在任何有 numpy 的环境跑。
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

LAYERS = ("paths", "import", "cuda", "model", "geometry", "geopackage")


def _manifest() -> dict:
    """读部署 manifest。

    分发包里默认只有 ``manifest.template.json``（路径是占位），
    需要接收方填好三个 python 路径后另存为 ``manifest.json``。
    这里给出明确提示而不是抛 FileNotFoundError —— 否则对方只看到一段
    traceback，不知道下一步该做什么。
    """
    path = os.path.join(HERE, "manifest.json")
    if not os.path.isfile(path):
        tpl = os.path.join(HERE, "manifest.template.json")
        print("=" * 70)
        print("缺少 deployment/manifest.json")
        print()
        if os.path.isfile(tpl):
            print("包里只有模板：deployment/manifest.template.json")
            print()
            print("请按以下步骤操作：")
            print("  1. 打开 deployment/manifest.template.json")
            print("  2. 把 environments 里每一项的 \"python\" 改成你的解释器绝对路径")
            print("     （detector_hf 与 vector 相同，都指向 vecrot-vllm 环境）")
            print("  3. 另存为 deployment/manifest.json（同一目录）")
            print()
            print("更省事的做法：直接运行")
            print("  python install.py --skip-envs")
            print("它会按模板生成 manifest 并自动填入检测到的解释器。")
        else:
            print("manifest.template.json 也不存在，包可能不完整。")
        print("=" * 70)
        raise SystemExit(2)
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    # manifest 里的模型路径是相对**仓库根**写的（manifest 自己在 deployment/ 下）
    m["_root"] = REPO
    m["_repo_root"] = REPO
    return m


def _resolve(manifest: dict, rel: str | None) -> str | None:
    if not rel:
        return None
    return rel if os.path.isabs(rel) else os.path.normpath(os.path.join(manifest["_root"], rel))


def check_paths(m: dict) -> list[tuple[bool, str]]:
    out: list[tuple[bool, str]] = []
    models = m.get("models") or {}
    det = models.get("detector") or {}
    vec = models.get("vectorizer") or {}
    for label, rel in (
        ("vecrot 目录", vec.get("vecrot_dir")),
        ("轮廓适配器 ckpt", vec.get("ckpt")),
        ("VectorLLM 基座", vec.get("base")),
        ("检测权重", det.get("weights")),
        ("检测配置", det.get("config")),
    ):
        p = _resolve(m, rel)
        if p and os.path.exists(p):
            out.append((True, f"{label}: {p}"))
        else:
            out.append((False, f"{label} 缺失: {p}"))
    for role, env in (m.get("environments") or {}).items():
        exe = env.get("python")
        if exe and os.path.isfile(exe):
            out.append((True, f"环境 {role} python: {exe}"))
        else:
            out.append((False, f"环境 {role} python 不存在: {exe}"))
    base_cfg = _resolve(m, (models.get("vectorizer") or {}).get("base"))
    if base_cfg and os.path.isfile(os.path.join(base_cfg, "config.json")):
        out.append((True, "基座 config.json 存在"))
    else:
        out.append((False, "基座缺 config.json"))
    return out


def check_import() -> list[tuple[bool, str]]:
    out = []
    for mod in ("numpy", "rasterio", "pyproj", "shapely", "pyogrio", "geopandas"):
        try:
            m = importlib.import_module(mod)
            out.append((True, f"{mod} {getattr(m, '__version__', '?')}"))
        except Exception as exc:                    # noqa: BLE001
            out.append((False, f"{mod} 导入失败: {exc}"))
    return out


def check_cuda() -> list[tuple[bool, str]]:
    try:
        import torch
    except Exception as exc:                        # noqa: BLE001
        return [(False, f"torch 不可用（应在 detector/vector 环境运行）: {exc}")]
    out = [(True, f"torch {torch.__version__}")]
    if not torch.cuda.is_available():
        return out + [(False, "CUDA 不可用")]
    out.append((True, f"CUDA {torch.version.cuda}, 设备数 {torch.cuda.device_count()}"))
    try:
        import torchvision

        x = torch.randn(1, 3, 64, 64, device="cuda")
        w = torch.randn(4, 3, 3, 3, device="cuda")
        torch.nn.functional.conv2d(x, w, padding=1)
        out.append((True, f"CUDA 卷积实测通过（torchvision {torchvision.__version__}）"))
    except Exception as exc:                        # noqa: BLE001
        out.append((False, f"CUDA 算子实测失败: {exc}"))
    return out


def check_model(m: dict) -> list[tuple[bool, str]]:
    """尝试加载 VectorLLM（需要 vector 环境的 torch/transformers）。"""
    out = []
    vec = (m.get("models") or {}).get("vectorizer") or {}
    vecrot = _resolve(m, vec.get("vecrot_dir"))
    if not vecrot:
        return [(False, "未配置 vecrot_dir")]
    sys.path.insert(0, os.path.join(vecrot, "lib"))
    sys.path.insert(0, REPO)
    try:
        from backend.vectorizer.backends import VecrotBackend

        b = VecrotBackend({
            "vecrot_dir": vecrot,
            "vector_ckpt": _resolve(m, vec.get("ckpt")),
            "vector_base": _resolve(m, vec.get("base")),
            "device": "cuda:0",
        })
        info = b.prepare()
        out.append((True, f"VectorLLM 加载成功：embedding={info['embedding_rows']} "
                          f"need_scale={info['need_scale']}"))
        b.close()
    except Exception as exc:                        # noqa: BLE001
        out.append((False, f"VectorLLM 加载失败: {type(exc).__name__}: {exc}"))
    return out


def check_geometry() -> list[tuple[bool, str]]:
    """纯 numpy 的坐标链 + 去重自检（不需要 GPU、不需要 GDAL）。"""
    out = []
    sys.path.insert(0, REPO)
    from backend.geo.boxes import Detection, dedup_detections, rbox_corners, rbox_iou
    from backend.geo.crop import CropTransform

    import math

    tf = CropTransform(100.0, 200.0, 40.0, 20.0, math.radians(30), 58, 29, 1.45)
    pts = [[0.0, 0.0], [tf.crop_w, 0.0], [tf.crop_w, tf.crop_h], [0.0, tf.crop_h]]
    back = tf.source_to_crop(tf.crop_to_source(pts))
    err = max(abs(a - b) for p, q in zip(pts, back) for a, b in zip(p, q))
    out.append((err <= 1e-9, f"裁剪坐标往返误差 {err:.3e} px"))

    a = Detection(0, 0.9, (100.0, 100.0, 40.0, 20.0, 0.0))
    b = Detection(1, 0.8, (102.0, 101.0, 40.0, 20.0, 0.0))
    iou = rbox_iou(a.rbox, b.rbox)
    res = dedup_detections([a, b], iou_threshold=0.5)
    out.append((len(res.kept) == 1 and res.suppressed, f"IoU={iou:.3f} 去重保留 {len(res.kept)}"))

    # 紧邻但不重叠的两栋楼不能被合并
    c = Detection(0, 0.9, (100.0, 100.0, 40.0, 20.0, 0.0))
    d = Detection(0, 0.9, (140.5, 100.0, 40.0, 20.0, 0.0))
    res2 = dedup_detections([c, d], iou_threshold=0.5)
    out.append((len(res2.kept) == 2, f"紧邻建筑保留 {len(res2.kept)}/2"))
    return out


def check_geopackage() -> list[tuple[bool, str]]:
    out = []
    sys.path.insert(0, REPO)
    import tempfile

    from backend.geo.exporter import export_outputs

    tmp = tempfile.mkdtemp(prefix="vecrot_selfcheck_")
    try:
        report = {"schema_version": "1.0", "pixel_area_m2": None,
                  "completed_at_utc": "2026-01-01T00:00:00Z"}
        res = export_outputs(
            job_id="selfcheck",
            detections=[{"det_id": "d000000", "det_score": 0.9,
                         "corners": [[0, 0], [10, 0], [10, 10], [0, 10]]}],
            buildings=[],
            crs="EPSG:32648",
            output_dir=os.path.join(tmp, "out"),
            src_crs="EPSG:32648",
            aoi_policy="intersect_keep_complete",
            report=report,
        )
        out.append((os.path.exists(res.boxes_path), f"boxes.gpkg: {res.boxes_path}"))
        out.append((os.path.exists(res.buildings_path),
                    f"buildings.gpkg（空层）: {res.buildings_path}"))
        out.append((res.counts["boxes"] == 1 and res.counts["buildings"] == 0,
                    f"计数 {res.counts}"))
    except Exception as exc:                        # noqa: BLE001
        import traceback

        out.append((False, f"GPKG 导出失败: {type(exc).__name__}: {exc}"))
        out.append((False, traceback.format_exc()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


CHECKERS = {
    "paths": lambda m: check_paths(m),
    "import": lambda m: check_import(),
    "cuda": lambda m: check_cuda(),
    "model": lambda m: check_model(m),
    "geometry": lambda m: check_geometry(),
    "geopackage": lambda m: check_geopackage(),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="all", choices=list(LAYERS) + ["all"])
    args = ap.parse_args()
    m = _manifest()
    layers = LAYERS if args.layer == "all" else (args.layer,)
    failed = 0
    for name in layers:
        print(f"\n=== {name} ===")
        for ok, msg in CHECKERS[name](m):
            print(("  [OK]   " if ok else "  [FAIL] ") + msg)
            if not ok:
                failed += 1
    print(f"\n{'全部通过' if failed == 0 else f'{failed} 项未通过'}（已检查层：{', '.join(layers)}）")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
