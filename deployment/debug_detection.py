"""检测链路逐层诊断：定位「真实瓦片 0 框」究竟发生在哪一层。

跑法：
    deployment/envs/detector/python.exe deployment/debug_detection.py

依次报告：
1. 输入预处理（通道均值、归一化范围）
2. Backbone 特征
3. RPN 前景概率分布与提案数
4. RoI 分类概率分布（绕过 NMS 与阈值）
5. 最终 NMS 后的框
"""
from __future__ import annotations

import json
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from backend.detector.backends import rgb_to_model_array
from backend.detector.strip_rcnn_backend import StripRcnnDetector




def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="本机待诊断影像路径")
    args = ap.parse_args()
    manifest = json.load(open(os.path.join(REPO, "deployment", "manifest.json"),
                              encoding="utf-8"))
    det = manifest["models"]["detector"]
    res = lambda r: r if os.path.isabs(r) else os.path.normpath(os.path.join(REPO, r))

    d = StripRcnnDetector({
        "strip_rcnn_repo": res(det["repo"]),
        "detector_weights": res(det["weights"]),
        "detector_config": res(det["config"]),
        "device": "cuda:0", "score_thr": 0.05,
        "derived_dir": os.path.join(REPO, "work", "detector_check", "derived_dbg"),
    })
    d.prepare()
    m = d._model

    from PIL import Image

    path = os.path.abspath(args.image)
    rgb = np.asarray(Image.open(path).convert("RGB"))
    print(f"=== 1. 输入 ===\n{os.path.basename(path)}: {rgb.shape} dtype={rgb.dtype}")
    print(f"RGB 通道均值: {rgb.reshape(-1, 3).mean(0).round(1)}")
    print(f"  若均值顺序为 R<G<B 说明是 RGB；反之可能是 BGR 被误当 RGB")

    arr = rgb_to_model_array(rgb)
    print(f"\n归一化后 shape={arr.shape} range=[{arr.min():.2f}, {arr.max():.2f}]")
    print(f"  通道均值: {arr.reshape(3, -1).mean(1).round(3)}")

    tensor = torch.from_numpy(arr).unsqueeze(0).cuda()
    meta = dict(img_shape=(1024, 1024, 3), ori_shape=(1024, 1024, 3),
                pad_shape=(1024, 1024, 3),
                scale_factor=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                flip=False, filename="dbg")
    metas = [[meta]]

    with torch.inference_mode():
        feats = m.extract_feat(tensor)
        print(f"\n=== 2. Backbone ===\n{[tuple(f.shape) for f in feats]}")
        for i, f in enumerate(feats):
            print(f"  lvl{i}: mean={float(f.mean()):+.3f} std={float(f.std()):.3f}")

        print("\n=== 3. RPN ===")
        rpn_outs_pair = m.rpn_head(feats)
        rpn_cls, rpn_reg = rpn_outs_pair
        probs = [c.sigmoid() for c in rpn_cls]
        allp = torch.cat([p.reshape(-1) for p in probs])
        print(f"前景概率 max={float(allp.max()):.4f} "
              f"p99.9={float(allp.quantile(0.999)):.4f} mean={float(allp.mean()):.6f}")
        for t in (0.9, 0.5, 0.3, 0.1, 0.05):
            print(f"  >{t}: {int((allp > t).sum())} 锚点")

        # 分层看每层锚点分数
        for i, p in enumerate(probs):
            print(f"  lvl{i} max={float(p.max()):.4f} >0.5={int((p > 0.5).sum())}")
            break        # 只看第一层，避免输出太长

        # 注意：simple_test_rpn(x, img_metas) 的 x 是 **backbone 特征**，
        # 它内部会再跑一次 rpn_head。直接把上面算好的 (rpn_cls, rpn_reg) 传进去
        # 会让它把 list 当张量去做 conv2d（TypeError）。
        # get_bboxes 的 img_metas 是**扁平列表**（每张图一个 dict），不是嵌套
        proposals = m.rpn_head.get_bboxes(*rpn_outs_pair, img_metas=[meta])
        print(f"提案: {[tuple(p.shape) for p in proposals]}")

        print("\n=== 4. RoI 分类（绕过 NMS/阈值）===")
        rois = proposals[0]
        if rois.numel() == 0:
            print("  没有提案，后续无法评估")
            return 1
        roi_feats = m.roi_head.bbox_roi_extractor(
            feats[:m.roi_head.bbox_roi_extractor.num_inputs], rois)
        cls_score, bbox_pred = m.roi_head.bbox_head(roi_feats)
        prob = cls_score.softmax(-1)
        fg = prob[:, 1] if prob.shape[1] > 1 else prob[:, 0]
        print(f"前景概率 max={float(fg.max()):.4f} p99={float(fg.quantile(0.99)):.4f} "
              f"mean={float(fg.mean()):.4f}")
        for t in (0.9, 0.5, 0.3, 0.1, 0.05):
            print(f"  >{t}: {int((fg > t).sum())} 个提案")

        # 看分数最高的那几个提案都在哪
        top = torch.topk(fg, min(5, fg.numel()))
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            r = rois[idx].tolist()
            print(f"  提案#{idx}: x={r[1]:.0f} y={r[2]:.0f} w={r[3]:.0f} h={r[4]:.0f} "
                  f"ang={r[5]:+.2f} fg={score:.4f}")

        print("\n=== 5. 最终输出 ===")
        boxes = d.predict_windows([{"window_id": 0, "rgb": rgb}])[0]
        print(f"{len(boxes)} 个框（score>={d.score_thr}）")
        for b in sorted(boxes, key=lambda x: -x.score)[:8]:
            print(f"  cx={b.cx:.1f} cy={b.cy:.1f} w={b.w:.1f} h={b.h:.1f} "
                  f"ang={b.angle:+.3f} score={b.score:.3f}")

    d.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
