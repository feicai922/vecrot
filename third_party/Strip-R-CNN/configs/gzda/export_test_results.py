"""Export test detections, representative previews, and oriented AP scores."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import mmcv
import numpy as np
from mmcv import Config
from mmrotate.datasets import build_dataset


def preview(task):
    index, info, detections, source, target = task
    path = os.path.join(source, info['filename'])
    for attempt in range(3):
        try:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError('cannot decode ' + path)
            for row in detections:
                if float(row[5]) < 0.3:
                    continue
                poly = cv2.boxPoints(((float(row[0]), float(row[1])), (float(row[2]), float(row[3])), float(np.degrees(row[4]))))
                poly = np.rint(poly).astype(np.int32)
                cv2.polylines(image, [poly], True, (0, 255, 0), 2, cv2.LINE_AA)
                xy = tuple(int(x) for x in poly[0])
                cv2.putText(image, '%.2f' % row[5], xy, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
            dst = os.path.join(target, '%04d_%s.jpg' % (index, os.path.splitext(info['filename'])[0]))
            if not cv2.imwrite(dst, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
                raise RuntimeError('cannot write ' + dst)
            return dst
        except Exception:
            if attempt == 2:
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    cfg = Config.fromfile('configs/gzda/strip_s_gzda.py')
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    results = mmcv.load(os.path.join(args.output_dir, 'predictions.pkl'))
    if len(results) != len(dataset):
        raise RuntimeError('prediction count %d differs from dataset %d' % (len(results), len(dataset)))
    jsonl = os.path.join(args.output_dir, 'detections.jsonl')
    totals = {'images': len(results), 'detections_score_ge_005': 0, 'detections_score_ge_03': 0, 'empty_prediction_images_score_ge_03': 0}
    with open(jsonl, 'w') as f:
        for info, classes in zip(dataset.data_infos, results):
            dets = np.asarray(classes[0]).reshape(-1, 6)
            records = []
            for row in dets:
                poly = cv2.boxPoints(((float(row[0]), float(row[1])), (float(row[2]), float(row[3])), float(np.degrees(row[4]))))
                records.append({'category': 'building', 'score': round(float(row[5]), 6), 'rbox_xywha_rad': [round(float(v), 5) for v in row[:5]], 'polygon_xy': [[round(float(v), 2) for v in pt] for pt in poly]})
            n03 = sum(float(row[5]) >= 0.3 for row in dets)
            totals['detections_score_ge_005'] += len(dets)
            totals['detections_score_ge_03'] += n03
            totals['empty_prediction_images_score_ge_03'] += int(n03 == 0)
            f.write(json.dumps({'image': info['filename'], 'width': info['width'], 'height': info['height'], 'detections': records}, ensure_ascii=False, separators=(',', ':')) + '\n')
    print('Exported %d image records to %s' % (len(results), jsonl), flush=True)
    preview_dir = os.path.join(args.output_dir, 'previews_score_0.3')
    os.makedirs(preview_dir, exist_ok=True)
    selected = np.linspace(0, len(results)-1, min(64, len(results)), dtype=int).tolist()
    tasks = [(i, dataset.data_infos[i], np.asarray(results[i][0]).reshape(-1, 6), cfg.data.test.img_prefix, preview_dir) for i in selected]
    failures = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(preview, t): t[0] for t in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                failures.append({'index': futures[future], 'error': str(exc)})
            if done % 16 == 0 or done == len(tasks):
                print('Previews %d/%d, failures %d' % (done, len(tasks), len(failures)), flush=True)
    if failures or len(os.listdir(preview_dir)) != len(tasks):
        raise RuntimeError('preview export incomplete: %s' % failures)
    totals['previews'] = len(tasks)
    totals['preview_score_threshold'] = 0.3
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(totals, f, indent=2, ensure_ascii=False)
    print('Summary:', totals, flush=True)
    scores = {}
    for value in range(50, 100, 5):
        threshold = round(value / 100, 2)
        print('Evaluating oriented IoU %.2f' % threshold, flush=True)
        scores['%.2f' % threshold] = float(dataset.evaluate(results, metric='mAP', iou_thr=threshold, nproc=4)['mAP'])
    metrics = {'rotated_ap_by_iou': scores, 'rotated_map_50_95': sum(scores.values()) / len(scores), 'score_threshold': 0.05, 'nms_iou': 0.3, 'max_detections_per_image': 2000, 'note': 'Single-class rotated AP averaged over IoU 0.50:0.05:0.95; MMRotate evaluation, not COCO API.'}
    with open(os.path.join(args.output_dir, 'rotated_ap_50_95.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Metrics:', metrics, flush=True)

if __name__ == '__main__':
    main()
