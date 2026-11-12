"""检测精度验证：用真实 COCO 标注算召回与匹配率。

    deployment/envs/detector/python.exe deployment/eval_detection.py [--limit N]

指标（都在瓦片像素空间，与 pipeline 口径一致）：
* GT 数 / 检出数 / 匹配数（旋转框 IoU ≥ 阈值）
* recall（匹配 GT / 总 GT）
* precision（匹配检出 / 总检出）
* 按 score 阈值分组
* has_building=False 的瓦片是否误检（假阳性率）

不追求刷分，只回答一个问题：**权重在这台机器上加载出来的检测是否正常。**
参考值来自 vecrot/README.md：端到端 mask AP 0.2766（Strip-R-CNN 框 score≥0.3）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from backend.detector.backends import rgb_to_model_array  # noqa: E402
from backend.detector.strip_rcnn_backend import StripRcnnDetector  # noqa: E402
from backend.geo import boxes as bgeo  # noqa: E402




def xywh_to_rbox(b):
    """COCO [x, y, w, h]（左上+宽高）→ 我们的 (cx, cy, w, h, 0)。"""
    x, y, w, h = [float(v) for v in b[:4]]
    return (x + w / 2, y + h / 2, w, h, 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="本机测试影像目录")
    ap.add_argument("--annotations", required=True, help="本机 COCO 标注 JSON")
    ap.add_argument("--limit", type=int, default=40, help="最多评估多少张瓦片")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(REPO, "deployment", "manifest.json"),
                              encoding="utf-8"))
    det = manifest["models"]["detector"]
    res = lambda r: r if os.path.isabs(r) else os.path.normpath(os.path.join(REPO, r))

    coco = json.load(open(args.annotations, encoding="utf-8"))
    by_file = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        b = ann["bbox"]
        if len(b) != 4 or b[2] <= 0 or b[3] <= 0:
            continue
        by_file.setdefault(ann["image_id"], []).append(b)
    img_by_id = {im["id"]: im for im in coco["images"]}

    # 优先挑有建筑的瓦片，再补一些无建筑的看误检
    with_b = [im for im in coco["images"] if im.get("has_building")]
    no_b = [im for im in coco["images"] if not im.get("has_building")]
    chosen = with_b[:args.limit]
    chosen += no_b[:max(3, args.limit // 10)]

    d = StripRcnnDetector({
        "strip_rcnn_repo": res(det["repo"]),
        "detector_weights": res(det["weights"]),
        "detector_config": res(det["config"]),
        "device": "cuda:0", "score_thr": 0.05,
        "derived_dir": os.path.join(REPO, "work", "detector_check", "derived_eval"),
    })
    d.prepare()

    from PIL import Image
    import time

    tot_gt = tot_det = tot_match = 0
    empty_tiles = 0
    empty_tiles_fp = 0
    per_tile = []
    t0 = time.time()

    for im in chosen:
        path = os.path.join(args.images, im["file_name"])
        if not os.path.isfile(path):
            continue
        rgb = np.asarray(Image.open(path).convert("RGB"))
        boxes = d.predict_windows([{"window_id": 0, "rgb": rgb}])[0]
        dets = [b for b in boxes if b.score >= args.score_thr]
        gts = [xywh_to_rbox(b) for b in by_file.get(im["id"], [])]

        # 贪心匹配（按分数降序，每个 GT 只匹配一次）
        used = set()
        matched = 0
        det_rboxes = [(b.cx, b.cy, b.w, b.h, b.angle) for b in dets]
        order = sorted(range(len(dets)), key=lambda i: -dets[i].score)
        for i in order:
            best, best_iou = -1, args.iou_thr
            for j, g in enumerate(gts):
                if j in used:
                    continue
                iou = bgeo.rbox_iou(det_rboxes[i], g)
                if iou >= best_iou:
                    best, best_iou = j, iou
            if best >= 0:
                used.add(best)
                matched += 1

        tot_gt += len(gts)
        tot_det += len(dets)
        tot_match += matched
        if not gts:
            empty_tiles += 1
            empty_tiles_fp += len(dets)
        per_tile.append((im["file_name"], len(gts), len(dets), matched))

    elapsed = time.time() - t0
    d.close()

    recall = tot_match / tot_gt if tot_gt else 0.0
    precision = tot_match / tot_det if tot_det else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0

    print(f"\n=== 评估结果（{len(per_tile)} 张瓦片，{elapsed:.1f}s）===")
    print(f"  阈值: score≥{args.score_thr}, 旋转 IoU≥{args.iou_thr}")
    print(f"  GT 框      : {tot_gt}")
    print(f"  检出框     : {tot_det}")
    print(f"  匹配       : {tot_match}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  F1         : {f1:.4f}")
    print(f"  无建筑瓦片 : {empty_tiles} 张，误检 {empty_tiles_fp} 个框")
    print(f"  吞吐       : {tot_det/max(elapsed,1e-9):.1f} 框/s，"
          f"{len(per_tile)/max(elapsed,1e-9):.2f} 瓦片/s")

    # 最差/最好的几张，便于人工抽查
    per_tile.sort(key=lambda x: (x[3] / x[1]) if x[1] else 1.0)
    print("\n  召回最低的 5 张（GT/检出/匹配）:")
    for name, g, dd, m in per_tile[:5]:
        print(f"    {name}: {g}/{dd}/{m}")
    print("\n  注：这是瓦片级目标检测口径，不是 mask AP。")
    print("  vecrot/README 报告的端到端 mask AP 0.2766 由 tools/eval_mask_ap.py 算出，")
    print("  需要轮廓结果与 pycocotools，属下一步。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
