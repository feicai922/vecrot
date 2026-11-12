"""Count TP, FP and duplicates at a fixed, preselected score threshold."""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.ops import box_iou_rotated
from mmrotate.core.evaluation.eval_map import tpfp_default
from mmrotate.datasets import build_dataset


def score_one(item):
    dets, gt = item
    dets = np.asarray(dets, dtype=np.float32).reshape(-1, 6)
    dets = dets[dets[:, 5] >= 0.30]
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 5)
    if not len(dets):
        return (len(gt), 0, 0, 0, 0)
    tp, fp = tpfp_default(dets, gt, np.empty((0, 5), dtype=np.float32), iou_thr=0.5)
    duplicate = 0
    if len(gt):
        overlaps = box_iou_rotated(torch.from_numpy(dets[:, :5].copy()), torch.from_numpy(gt)).numpy()
        duplicate = int(((fp[0] > 0) & (overlaps.max(axis=1) >= 0.5)).sum())
    return (len(gt), len(dets), int(tp.sum()), int(fp.sum()), duplicate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    cfg = Config.fromfile('configs/gzda/strip_s_gzda.py')
    data = build_dataset(cfg.data.test, dict(test_mode=True))
    results = mmcv.load(os.path.join(args.output_dir, 'predictions.pkl'))
    assert len(data) == len(results)
    inputs = [(result[0], info['ann']['bboxes']) for info, result in zip(data.data_infos, results)]
    totals = np.zeros(5, dtype=np.int64)
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as pool:
        for i, item in enumerate(pool.map(score_one, inputs, chunksize=32), 1):
            totals += item
            if i % 500 == 0:
                print('Matching %d/%d' % (i, len(inputs)), flush=True)
    gt, det, tp, fp, dup = [int(x) for x in totals]
    report = {'score_threshold':0.3, 'matching_iou':0.5, 'gt_boxes':gt, 'detections':det, 'true_positives':tp, 'false_positives':fp, 'duplicate_false_positives':dup, 'other_false_positives':fp-dup, 'false_negatives':gt-tp, 'precision':tp/(tp+fp) if tp+fp else None, 'recall':tp/gt if gt else None, 'note':'Per-image one-to-one rotated-IoU matching against existing labels; duplicates are detections matching an already assigned ground-truth box at IoU >= 0.5. At fixed score threshold, not selected on test set.'}
    with open(os.path.join(args.output_dir, 'precision_at_0.3.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
