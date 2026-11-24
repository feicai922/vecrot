#!/usr/bin/env python
"""用真实瓦片验证 rio_detr 后端（需要 vecrot-rio / riodetr 环境）。

    python deployment/verify_rio_runtime.py \
        --repo  /data1/wzh26/code/RiO-DETRv2 \
        --config /data1/wzh26/code/RiO-DETRv2/configs/rio-gzda/riov2_hgnetv2_x_gzda_12e.yml \
        --weights /path/last.pth \
        --image  某张瓦片.tif

它做的事与作业里完全一致：``build_detector("rio_detr", options).prepare()`` →
``predict_windows()``，所以能提前暴露"配置/权重不配套""预处理方向错"
"输出解析错"这类只有在真机上才会炸的问题（本仓库在这三类问题上都踩过）。

可选 ``--ann`` 给 DOTA 格式标注，会顺带算一个 **IoU≥0.5 的瓦片级 recall/precision**，
用于判断这个 checkpoint 是不是真的学到了东西（不是精度评测，只是冒烟）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((bool(ok), msg))
    print(("  [OK]   " if ok else "  [FAIL] ") + msg, flush=True)


def load_window(path: str):
    """读成 (H,W,3) uint8 RGB —— 与作业里 worker 拿到的窗口一致。"""
    import numpy as np
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def _rbox_poly(r):
    """(cx,cy,w,h,angle) → 4×2 角点（逆时针），与 backend.geo.boxes 的约定一致。"""
    import numpy as np

    cx, cy, w, h, ang = [float(v) for v in r]
    c, s = np.cos(ang), np.sin(ang)
    dx, dy = w / 2.0, h / 2.0
    local = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]], dtype=np.float64)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + np.array([cx, cy])


def _poly_area(poly) -> float:
    import numpy as np

    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs((x * np.roll(y, -1) - np.roll(x, -1) * y).sum()) / 2.0)


def _clip_convex(subject, clip):
    """Sutherland–Hodgman：把凸多边形 subject 裁到凸多边形 clip 内。"""
    import numpy as np

    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def intersect(p, q, a, b):
        x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
        x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return q
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])

    output = list(subject)
    n = len(clip)
    for i in range(n):
        a, b = clip[i], clip[(i + 1) % n]
        if not output:
            return np.zeros((0, 2))
        current, output = output, []
        prev = current[-1]
        for cur in current:
            if inside(cur, a, b):
                if not inside(prev, a, b):
                    output.append(intersect(prev, cur, a, b))
                output.append(cur)
            elif inside(prev, a, b):
                output.append(intersect(prev, cur, a, b))
            prev = cur
    return np.array(output) if output else np.zeros((0, 2))


def rbox_iou(a, b) -> float:
    """两个 (cx,cy,w,h,angle) 旋转矩形的 IoU。

    自己算凸多边形裁剪，不引 shapely —— RiO-DETR 的推理环境里没有 GIS 依赖，
    这个脚本要能在 vecrot-rio 里直接跑。
    """
    import numpy as np

    pa, pb = _rbox_poly(a), _rbox_poly(b)
    aa, ab = _poly_area(pa), _poly_area(pb)
    if aa <= 0 or ab <= 0:
        return 0.0
    inter = _poly_area(_clip_convex(pa, pb))
    union = aa + ab - inter
    return float(inter / union) if union > 0 else 0.0


def read_dota_ann(path: str):
    """DOTA 标注 → [(cx,cy,w,h,angle), …]。"""
    import numpy as np

    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9:
                continue
            pts = np.array([[float(parts[i]), float(parts[i + 1])]
                            for i in range(0, 8, 2)], dtype=np.float64)
            cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
            edge1 = np.linalg.norm(pts[1] - pts[0])
            edge2 = np.linalg.norm(pts[2] - pts[1])
            ang = float(np.arctan2(*(pts[1] - pts[0])[::-1]))
            out.append((float(cx), float(cy), float(edge1), float(edge2), ang))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="RiO-DETRv2 源码目录")
    ap.add_argument("--config", required=True, help="推理用 .yml")
    ap.add_argument("--weights", required=True, help="checkpoint (.pth)")
    ap.add_argument("--image", help="一张测试瓦片（不给就用随机噪声）")
    ap.add_argument("--ann", help="DOTA 标注，给了就算 recall/precision")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score-thr", type=float, default=0.4)
    ap.add_argument("--expect-recall", type=float, default=None,
                    help="给了才把 recall 当硬指标（1 epoch 的 checkpoint 召回必然很低，"
                         "不适合当失败条件）")
    args = ap.parse_args(argv)

    print(f"仓库根：{REPO}")
    from backend.detector.backends import DetectorNotAvailable, build_detector

    backend = build_detector("rio_detr", {
        "detector_repo": args.repo,
        "detector_config": args.config,
        "detector_weights": args.weights,
        "device": args.device,
        "score_thr": args.score_thr,
    })

    print("\n=== 1. 加载 ===")
    t0 = time.time()
    try:
        info = backend.prepare()
    except DetectorNotAvailable as exc:
        check(False, f"加载失败：{exc.code}: {exc.message}")
        return 1
    except Exception as exc:                        # noqa: BLE001
        check(False, f"加载抛异常：{type(exc).__name__}: {exc}")
        return 1
    load_s = time.time() - t0
    check(True, f"加载完成 {load_s:.1f}s  backend={info.get('backend')}")
    wl = info.get("weight_loading") or {}
    check(wl.get("matched_all_structural") is True,
          f"权重结构匹配：missing={len(wl.get('missing_keys') or [])} "
          f"unexpected={len(wl.get('unexpected_keys') or [])} "
          f"source={info.get('weights_source')}")
    check(bool(info.get("input_size")), f"输入尺寸 {info.get('input_size')}")
    check(info.get("num_classes") == 1, f"类别数 {info.get('num_classes')}")

    print("\n=== 2. 推理 ===")
    if args.image:
        rgb = load_window(args.image)
        check(rgb.ndim == 3 and rgb.shape[2] == 3,
              f"窗口读取 {rgb.shape}（{os.path.basename(args.image)}）")
    else:
        import numpy as np
        rgb = (np.random.default_rng(0).random((1024, 1024, 3)) * 255).astype("uint8")
        check(True, "使用合成噪声窗口（未给 --image）")

    t0 = time.time()
    boxes = backend.predict_windows([{"window_id": 0, "rgb": rgb}])[0]
    infer_s = time.time() - t0
    check(True, f"推理 {infer_s * 1000:.0f} ms，{len(boxes)} 个框（score≥{args.score_thr}）")

    h, w = rgb.shape[:2]
    inside = [b for b in boxes
              if 0 <= b.cx <= w and 0 <= b.cy <= h and b.w > 0 and b.h > 0]
    check(len(inside) == len(boxes), f"框中心/宽高都在窗口内：{len(inside)}/{len(boxes)}")
    finite = all(abs(b.angle) <= 3.15 for b in boxes)
    check(finite, "角度是弧度且落在 ±π 内")
    if boxes:
        sizes = sorted(b.w * b.h for b in boxes)
        med = sizes[len(sizes) // 2]
        print(f"         面积中位数 {med:.0f} px²（窗口 {w}×{h}）")

    if args.ann and os.path.isfile(args.ann):
        print("\n=== 3. 与标注粗略对照（冒烟，不是精度评测）===")
        gts = read_dota_ann(args.ann)
        matched = 0
        for gt in gts:
            if any(rbox_iou(gt, (b.cx, b.cy, b.w, b.h, b.angle)) >= 0.5 for b in boxes):
                matched += 1
        tp = sum(1 for b in boxes
                 if any(rbox_iou(gt, (b.cx, b.cy, b.w, b.h, b.angle)) >= 0.5 for gt in gts))
        recall = matched / len(gts) if gts else 0.0
        precision = tp / len(boxes) if boxes else 0.0
        check(len(gts) > 0, f"GT {len(gts)} 框 / 检出 {len(boxes)} 框")
        check(recall >= (args.expect_recall or 0.0),
              f"Recall {recall:.3f}  Precision {precision:.3f}（IoU≥0.5，窗口级）"
              + (f"，期望 ≥{args.expect_recall}" if args.expect_recall else "，仅观察"))

    backend.close()
    print()
    failed = [m for ok, m in results if not ok]
    if failed:
        print(f"{len(failed)} 项未通过：")
        for m in failed:
            print(f"  · {m}")
        return 1
    print(f"全部 {len(results)} 项通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
