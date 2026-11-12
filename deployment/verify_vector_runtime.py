"""VectorLLM 轮廓生成实机验证。

    deployment/envs/vector/python.exe deployment/verify_vector_runtime.py

验证：
1. 后端 prepare：词表一致性、ScaleCond 挂载、need_scale
2. 用真实裁剪跑轮廓（真实瓦片 + 真实检测框）
3. 状态分布：ok / abstained / empty / error
4. 坐标回代：裁剪坐标 → 源像素，检查落在框附近
5. 显存与耗时
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((bool(ok), msg))
    print(("  [OK]   " if ok else "  [FAIL] ") + msg)


def main() -> int:
    from backend.geo import crop as cgeo
    from backend.vectorizer.backends import VecrotBackend

    manifest = json.load(open(os.path.join(REPO, "deployment", "manifest.json"),
                              encoding="utf-8"))
    vec = manifest["models"]["vectorizer"]
    res = lambda r: r if os.path.isabs(r) else os.path.normpath(os.path.join(REPO, r))

    print("=== 1. 后端 prepare ===")
    backend = VecrotBackend({
        "vecrot_dir": res(vec["vecrot_dir"]),
        "vector_ckpt": res(vec["ckpt"]),
        "vector_base": res(vec["base"]),
        "device": "cuda:0",
        "vector_batch": 2,
    })
    t0 = time.time()
    try:
        info = backend.prepare()
    except Exception as exc:                   # noqa: BLE001
        import traceback

        traceback.print_exc()
        check(False, f"prepare 失败：{type(exc).__name__}: {exc}")
        return 1
    dt = time.time() - t0
    check(True, f"加载完成（{dt:.1f}s）")
    check(info["vocab_consistent"], f"词表一致：tokenizer={info['tokenizer_len']} "
                                    f"embedding={info['embedding_rows']}")
    check(info["pixel_idx_tokenizer"] == info["pixel_idx_model"],
          f"pixel_idx 对齐：{info['pixel_idx_tokenizer']}")
    check(info["need_scale"], "need_scale=True（尺度条件已启用）")
    check(not info["hr_size"], f"hr_size={info['hr_size']}（arch.json 里 hr_edge=false）")
    print(f"     arch: scale_cond={info['arch'].get('scale_cond')} "
          f"code_dim={info['arch'].get('code_dim')} "
          f"hidden={info['arch'].get('scale_cond_hidden')}")

    print("\n=== 2. 准备真实裁剪 ===")
    from PIL import Image

    TEST = r"G:\贵州省数据集\images\test"
    fname = "rural_anshunshi-4_tile0000.tif"
    rgb = np.asarray(Image.open(os.path.join(TEST, fname)).convert("RGB"))
    # 用之前实测检出的高分框（真实模型输出）
    det_boxes = [
        (246.06, 214.21, 51.68, 39.54, 0.11179, 0.94536),
        (569.07, 343.61, 109.46, 58.65, -0.20756, 0.87710),
        (187.01, 362.95, 137.90, 44.64, 1.38090, 0.81025),
        (498.89, 332.96, 130.86, 41.07, -1.55570, 0.75755),
    ]
    check(len(det_boxes) == 4, f"用 {len(det_boxes)} 个真实检测框")

    tasks = []
    transforms = {}
    H, W = rgb.shape[:2]
    for i, (cx, cy, w, h, ang, score) in enumerate(det_boxes):
        tf = cgeo.plan_crop((cx, cy, w, h, ang), expand=1.45)
        if tf is None:
            check(False, f"det{i} 裁剪规划失败")
            continue
        # 读取包含整个裁剪框的区域（越界补 0）
        import math

        x0 = int(math.floor(tf.cx - tf.crop_w / 2))
        y0 = int(math.floor(tf.cy - tf.crop_h / 2))
        x1 = x0 + tf.crop_w
        y1 = y0 + tf.crop_h
        rx0, ry0, rx1, ry1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        sub = rgb[ry0:ry1, rx0:rx1]
        pad = ((ry0 - y0, y1 - ry1), (rx0 - x0, x1 - rx1), (0, 0))
        if any(p > 0 for pair in pad for p in pair):
            sub = np.pad(sub, pad, mode="constant", constant_values=0)
        local_tf = cgeo.offset_transform(tf, x0, y0)
        cropped = cgeo.crop_rotated(sub, local_tf)
        tasks.append({
            "det_id": f"d{i:06d}",
            "crop_rgb": cropped,
            "crop_meta": {
                "obj_box_in_crop": list(tf.obj_box_in_crop()),
                "crop_box_in_crop": list(tf.crop_box_in_crop()),
                "transform": tf.to_dict(),
            },
        })
        transforms[f"d{i:06d}"] = tf
        print(f"     d{i:06d}: 框 ({cx:.0f},{cy:.0f}) {w:.0f}×{h:.0f} ang={ang:+.2f} "
              f"→ 裁剪 {tf.crop_w}×{tf.crop_h}")

    print("\n=== 3. 轮廓生成 ===")
    import torch

    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    out = backend.predict(tasks)
    elapsed = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 1024**3

    statuses = {}
    for r in out:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    print(f"     耗时 {elapsed:.2f}s（{len(tasks)/max(elapsed,1e-9):.2f} 框/s），"
          f"峰值显存 {peak:.2f} GB")
    print(f"     状态分布：{statuses}")
    check(len(out) == len(tasks), f"返回 {len(out)} 条结果 = {len(tasks)} 个输入")
    check(statuses.get("error", 0) == 0, f"无 error（{statuses}）")

    print("\n=== 4. 坐标回代检查 ===")
    for r, task in zip(out, tasks):
        det_id = task["det_id"]
        tf = transforms[det_id]
        if r.status != "ok" or not r.polygon_crop_px:
            print(f"     {det_id}: {r.status} ({r.reason})")
            continue
        src_pts = tf.crop_to_source(r.polygon_crop_px)
        xs = [p[0] for p in src_pts]
        ys = [p[1] for p in src_pts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        w = max(xs) - min(xs)
        h = max(ys) - min(ys)
        print(f"     {det_id}: {len(src_pts)} 点 → 源像素中心 ({cx:.1f},{cy:.1f}) "
              f"跨度 {w:.1f}×{h:.1f}（裁剪内 {len(r.polygon_crop_px)} 点）")
        # 轮廓中心应贴近检测框中心（expand=1.45，允许一定偏差）
        dbox = transforms[det_id]
        dist = ((cx - dbox.cx) ** 2 + (cy - dbox.cy) ** 2) ** 0.5
        max_side = max(dbox.w, dbox.h)
        check(dist < max_side * 1.2,
              f"{det_id} 轮廓中心距框中心 {dist:.1f}px < {max_side*1.2:.0f}px")
        check(all(np.isfinite([cx, cy])), f"{det_id} 坐标有限")

    backend.close()
    failed = sum(1 for ok, _ in results if not ok)
    print(f"\n{'全部通过' if failed == 0 else f'{failed} 项未通过'}（共 {len(results)} 项）")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
