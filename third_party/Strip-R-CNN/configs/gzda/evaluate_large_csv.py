"""Evaluate merged big-image OBB predictions against source building polygons.

GT cleaning mirrors /data1/wzh26/data/build_gzda_v0.py: repair invalid
geometries, transform CRS, filter <100 pixel², exact and IoU>0.5 polygon
annotation deduplication. Full-image boundary clipping is used rather than
the training dataset's 1024-pixel tile clipping.
"""
import argparse
import csv
import glob
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import fiona
import numpy as np
import rasterio
from rasterio.warp import transform_geom
from shapely import STRtree, box
from shapely.geometry import Polygon, shape
import torch
from mmcv.ops import box_iou_rotated

ROOT = '/data1/wzh26/data/building_img_shp/output'
CSV = '/data1/wzh26/vectorllm_v3_obb/input/demo/分类结果.csv'
DATA_STATS = '/data1/wzh26/data/gzda_v3_human_del/dataset_stats.json'
THRESHOLDS = [round(i / 100, 2) for i in range(50, 100, 5)]


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(tmp, path)


def signature(p):
    pts = [(round(float(x), 3), round(float(y), 3)) for x, y in p.exterior.coords]
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if not pts:
        return None
    k = min(range(len(pts)), key=lambda i: pts[i])
    return tuple(pts[k:] + pts[:k])


def clean_annotations(scene, raster):
    choices = sorted(glob.glob(os.path.join(ROOT, scene, '*.shp')))
    if len(choices) != 1:
        raise ValueError('Expected exactly one shapefile for %s, got %s' % (scene, choices))
    with rasterio.open(raster) as im:
        inv = ~im.transform
        width, height, image_crs = im.width, im.height, im.crs
    polys = []
    stats = dict(source_features=0, invalid=0, small=0, duplicate=0, clipped_out=0, crs_reprojected=False)
    with fiona.open(choices[0]) as src:
        if not src.crs_wkt or image_crs is None:
            raise ValueError('Missing CRS in ' + scene)
        source_crs = rasterio.crs.CRS.from_wkt(src.crs_wkt)
        convert = source_crs != image_crs
        stats['crs_reprojected'] = convert
        for feature in src:
            stats['source_features'] += 1
            g = feature['geometry']
            if not g:
                continue
            if convert:
                g = transform_geom(source_crs, image_crs, g)
            geom = shape(g)
            if geom.is_empty:
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
                if geom.is_empty or not geom.is_valid:
                    stats['invalid'] += 1
                    continue
            parts = [geom] if geom.geom_type == 'Polygon' else (list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [])
            for part in parts:
                try:
                    p = Polygon([inv * (float(x), float(y)) for x, y in part.exterior.coords])
                except Exception:
                    stats['invalid'] += 1
                    continue
                if p.is_empty or not p.is_valid:
                    stats['invalid'] += 1
                elif p.area < 100:
                    stats['small'] += 1
                else:
                    polys.append(p)
    seen, unique = set(), []
    for p in polys:
        key = signature(p)
        if key is not None and key in seen:
            stats['duplicate'] += 1
        else:
            if key is not None:
                seen.add(key)
            unique.append(p)
    if unique:
        tree = STRtree(unique)
        kept = np.zeros(len(unique), dtype=bool)
        for i in sorted(range(len(unique)), key=lambda k: -unique[k].area):
            p = unique[i]
            duplicate = False
            for j in tree.query(p):
                j = int(j)
                if j == i or not kept[j]:
                    continue
                q = unique[j]
                inter = p.intersection(q).area
                union = p.area + q.area - inter
                if union > 0 and inter / union > 0.5:
                    duplicate = True
                    break
            if duplicate:
                stats['duplicate'] += 1
            else:
                kept[i] = True
        unique = [p for i, p in enumerate(unique) if kept[i]]
    img_bounds = box(0, 0, width, height)
    gt = []
    for p in unique:
        if img_bounds.contains(p):
            parts = [p]
        else:
            clipped = p.intersection(img_bounds)
            if clipped.is_empty or clipped.area / p.area < 0.3:
                stats['clipped_out'] += 1
                continue
            if clipped.geom_type == 'Polygon':
                parts = [clipped]
            elif clipped.geom_type == 'MultiPolygon':
                parts = list(clipped.geoms)
            elif clipped.geom_type == 'GeometryCollection':
                parts = [x for x in clipped.geoms if x.geom_type == 'Polygon']
            else:
                parts = []
        for part in parts:
            if part.is_empty or part.area < 100 or not part.is_valid:
                continue
            (cx, cy), (w, h), angle = cv2.minAreaRect(np.asarray(part.exterior.coords, dtype=np.float32))
            if w > 0 and h > 0:
                gt.append([cx, cy, w, h, np.deg2rad(angle)])
    return np.asarray(gt, dtype=np.float32).reshape(-1, 5), stats


def envelopes(b):
    c, s = np.abs(np.cos(b[:, 4])), np.abs(np.sin(b[:, 4]))
    dx = 0.5 * (c * b[:, 2] + s * b[:, 3])
    dy = 0.5 * (s * b[:, 2] + c * b[:, 3])
    return box(b[:, 0]-dx, b[:, 1]-dy, b[:, 0]+dx, b[:, 1]+dy)


def sparse_best(dets, gt):
    n = len(dets)
    best_iou = np.zeros(n, dtype=np.float32)
    best_gt = np.full(n, -1, dtype=np.int32)
    if not n or not len(gt):
        return best_iou, best_gt
    tree = STRtree(envelopes(gt))
    for start in range(0, n, 4096):
        stop = min(start + 4096, n)
        pairs = tree.query(envelopes(dets[start:stop]))
        if pairs.shape[1] == 0:
            continue
        di, gi = pairs[0] + start, pairs[1]
        vals = np.empty(len(di), dtype=np.float32)
        for j in range(0, len(di), 100000):
            sl = slice(j, min(j + 100000, len(di)))
            vals[sl] = box_iou_rotated(torch.from_numpy(dets[di[sl], :5].copy()), torch.from_numpy(gt[gi[sl]].copy()), aligned=True).numpy()
        # Highest overlap per prediction; lowest GT index breaks exact ties.
        order = np.lexsort((gi, -vals, di))
        di, gi, vals = di[order], gi[order], vals[order]
        first = np.r_[True, di[1:] != di[:-1]]
        best_iou[di[first]] = vals[first]
        best_gt[di[first]] = gi[first]
    return best_iou, best_gt


def calc_tp(scores, best_iou, best_gt, n_gt, threshold):
    order = np.argsort(-scores)
    covered = np.zeros(n_gt, dtype=bool)
    tp = np.zeros(len(scores), dtype=np.uint8)
    for i in order:
        g = int(best_gt[i])
        if g >= 0 and best_iou[i] >= threshold and not covered[g]:
            covered[g] = True
            tp[i] = 1
    return tp


def eval_scene(job):
    meta, outdir, check = job
    torch.set_num_threads(1)
    cv2.setNumThreads(0)
    name = meta['name']
    cache = Path(outdir) / 'cache' / (name + '.npz')
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as d:
            return {'name': name, 'gt': int(d['n_gt']), 'detections': len(d['scores']), 'cached': True}
    dpath = Path(outdir).parent / 'scenes' / name / 'detections_pixels.npz'
    det = np.load(dpath)['boxes'].astype(np.float32)
    if det.ndim != 2 or det.shape[1] != 6 or not np.all(np.isfinite(det)):
        raise ValueError('invalid detections for ' + name)
    gt, gt_stats = clean_annotations(name, meta['image'])
    iou, gt_idx = sparse_best(det, gt)
    tps = np.stack([calc_tp(det[:, 5], iou, gt_idx, len(gt), t) for t in THRESHOLDS], axis=0)
    if check and len(det) and len(gt):
        from mmrotate.core.evaluation.eval_map import tpfp_default
        sub = det[:min(50, len(det))]
        siou, sgt = sparse_best(sub, gt)
        for t in (0.5, 0.75):
            expected, _ = tpfp_default(sub, gt, np.empty((0, 5), np.float32), iou_thr=t)
            actual = calc_tp(sub[:, 5], siou, sgt, len(gt), t)
            if not np.array_equal(actual, expected[0]):
                raise AssertionError('Sparse matching differs from MMRotate at IoU %.2f' % t)
    tmp = cache.with_name(cache.stem + '.tmp.npz')
    np.savez_compressed(tmp, scores=det[:, 5], best_iou=iou, best_gt=gt_idx, tps=tps, n_gt=np.int64(len(gt)), stats=np.array(json.dumps(gt_stats, ensure_ascii=False)))
    os.replace(tmp, cache)
    return {'name': name, 'gt': len(gt), 'detections': len(det), 'stats': gt_stats}


def ap_area(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))


def summarize(items):
    n_gt = sum(x['n_gt'] for x in items)
    if not items or not n_gt:
        return None
    scores = np.concatenate([x['scores'] for x in items])
    tp = np.concatenate([x['tps'] for x in items], axis=1)
    ious = np.concatenate([x['ious'] for x in items])
    order = np.argsort(-scores)
    tp = tp[:, order]
    ordered_scores = scores[order]
    ap = {}
    for k, threshold in enumerate(THRESHOLDS):
        cumulative = np.cumsum(tp[k], dtype=np.int64)
        recall = cumulative / n_gt
        precision = cumulative / np.arange(1, len(scores)+1)
        ap['%.2f' % threshold] = ap_area(recall, precision)
    chosen = ordered_scores >= 0.30
    n_det = int(chosen.sum())
    n_tp = int(tp[0, :n_det].sum())
    n_fp = n_det - n_tp
    ordered_ious = ious[order]
    dup = int(((tp[0, :n_det] == 0) & (ordered_ious[:n_det] >= 0.5)).sum())
    return {'scenes': len(items), 'ground_truth': n_gt, 'raw_predictions_score_ge_0.05': len(scores), 'ap50': ap['0.50'], 'ap75': ap['0.75'], 'rotated_map50_95': sum(ap.values()) / len(ap), 'ap_by_iou': ap, 'at_score_0.3': {'detections': n_det, 'true_positive': n_tp, 'false_positive': n_fp, 'false_negative': n_gt-n_tp, 'duplicate_fp_iou_ge_0.5': dup, 'precision': n_tp/n_det if n_det else None, 'recall': n_tp/n_gt, 'f1': 2*n_tp/(2*n_tp+n_fp+n_gt-n_tp) if (2*n_tp+n_fp+n_gt-n_tp) else None}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out.parent / 'manifest.json', encoding='utf-8') as f:
        scenes = json.load(f)['scenes']
    if args.limit:
        scenes = scenes[:args.limit]
    with open(DATA_STATS, encoding='utf-8') as f:
        dataset = json.load(f)
    split_map = {k: next(iter(v)) for k, v in dataset['per_scene'].items()}
    exclusions = {x['base_scene']: x['reason'] for x in dataset['excluded']}
    all_names = {x['name'] for x in scenes}
    if not all_names.issubset(set(split_map) | set(exclusions)):
        raise ValueError('Some scenes lack split/exclusion mapping')
    jobs = [(meta, str(out), i == 0) for i, meta in enumerate(scenes)]
    pending, failures, done = {}, {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job in jobs:
            pending[pool.submit(eval_scene, job)] = (job, 1)
        while pending:
            for future in as_completed(list(pending)):
                job, attempt = pending.pop(future)
                try:
                    result = future.result()
                    done.append(result)
                    print('Completed %d/%d %s GT=%s detections=%s cached=%s' % (len(done), len(jobs), result['name'], result['gt'], result['detections'], result.get('cached', False)), flush=True)
                except Exception as exc:
                    print('FAILED %s attempt %d: %s\n%s' % (job[0]['name'], attempt, exc, traceback.format_exc()), flush=True)
                    if attempt < 3:
                        pending[pool.submit(eval_scene, job)] = (job, attempt+1)
                    else:
                        failures[job[0]['name']] = str(exc)
                atomic_json(out / 'progress.json', {'status': 'running', 'completed': len(done), 'total': len(jobs), 'failed': failures, 'updated_at': time.time()})
                break
    if failures or len(done) != len(jobs):
        atomic_json(out / 'progress.json', {'status': 'failed', 'completed': len(done), 'total': len(jobs), 'failed': failures, 'updated_at': time.time()})
        raise RuntimeError('Scene evaluations did not all complete')
    entries = []
    for meta in scenes:
        name = meta['name']
        with np.load(out / 'cache' / (name + '.npz'), allow_pickle=False) as d:
            entries.append({'name': name, 'category': meta['category'], 'split': split_map.get(name, 'excluded'), 'exclusion_reason': exclusions.get(name, ''), 'scores': d['scores'], 'ious': d['best_iou'], 'tps': d['tps'], 'n_gt': int(d['n_gt']), 'cleaning': json.loads(str(d['stats']))})
    groups = {'all_85': entries, 'kept_74': [e for e in entries if e['split'] != 'excluded']}
    for group in ('train', 'val', 'test', 'excluded'):
        groups[group] = [e for e in entries if e['split'] == group]
    for category in sorted({e['category'] for e in entries}):
        groups['category_' + category] = [e for e in entries if e['category'] == category]
    report = {'method': {'prediction': 'merged whole-scene oriented boxes, scores >= 0.05, class building', 'ground_truth': 'original source shapefile, deduped with gzda_v0 rules, minimum-area rotated rectangles in image pixel space', 'gt_cleaning': 'repair invalid, transform source CRS to raster CRS, discard area < 100 pixels², exact and polygon IoU > 0.5 dedup, clip to image', 'matching': 'score-ranked one-to-one per scene with max rotated IoU, MMRotate area AP', 'precision_score_threshold': 0.3, 'iou_thresholds': THRESHOLDS, 'limitation': 'Full scenes, unlike tile-based evaluation; includes training scenes and scenes excluded due to annotation issues. AP50-95 uses MMRotate area AP, not COCO API.'}, 'groups': {k: summarize(v) for k, v in groups.items()}}
    atomic_json(out / 'metrics.json', report)
    with open(out / 'per_scene.csv', 'w', encoding='utf-8-sig', newline='') as f:
        fields = ['name','category','split','exclusion_reason','ground_truth','detections_at_0.3','true_positive','false_positive','false_negative','duplicate_fp_iou_ge_0.5','precision','recall','f1','ap50','ap75','rotated_map50_95','source_features','invalid','small','duplicate','clipped_out','crs_reprojected']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in entries:
            m = summarize([e])
            if m is None:
                n_det = int((e['scores'] >= 0.3).sum())
                m = {'ground_truth': 0, 'ap50': None, 'ap75': None, 'rotated_map50_95': None,
                     'at_score_0.3': {'detections': n_det, 'true_positive': 0, 'false_positive': n_det,
                                      'false_negative': 0, 'duplicate_fp_iou_ge_0.5': 0,
                                      'precision': 0.0 if n_det else None, 'recall': None, 'f1': None}}
            c = m['at_score_0.3']
            row = {'name': e['name'], 'category': e['category'], 'split': e['split'], 'exclusion_reason': e['exclusion_reason'], 'ground_truth': m['ground_truth'], 'detections_at_0.3': c['detections'], **{k:c[k] for k in ('true_positive','false_positive','false_negative','duplicate_fp_iou_ge_0.5','precision','recall','f1')}, 'ap50':m['ap50'], 'ap75':m['ap75'], 'rotated_map50_95':m['rotated_map50_95'], **e['cleaning']}
            w.writerow(row)
    atomic_json(out / 'progress.json', {'status': 'completed', 'completed': len(done), 'total': len(jobs), 'failed': {}, 'updated_at': time.time()})
    print('FINISHED', json.dumps(report['groups'], ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
