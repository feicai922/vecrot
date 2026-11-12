"""Streaming inference for the CSV-selected georeferenced rasters."""
import argparse
import csv
import json
import math
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import fiona
import numpy as np
import rasterio
import torch
from mmcv.ops import nms_rotated
from mmdet.apis import init_detector, inference_detector
from rasterio.enums import Resampling
from rasterio.windows import Window

CSV = '/data1/wzh26/vectorllm_v3_obb/input/demo/分类结果.csv'
ROOT = '/data1/wzh26/data/building_img_shp/output'
CONFIG = 'configs/gzda/strip_s_gzda.py'
SIZE = 1024
STRIDE = 824
MERGE_IOU = 0.3
OUTPUT_SCORE = 0.3
RAW_SCORE = 0.05
_local = threading.local()


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def starts(length):
    if length <= SIZE:
        return [0]
    items = list(range(0, length - SIZE + 1, STRIDE))
    if items[-1] != length - SIZE:
        items.append(length - SIZE)
    return items


def describe(row):
    name = row['name']
    path = os.path.join(ROOT, name, name + '.tif')
    for attempt in range(3):
        try:
            with rasterio.open(path) as src:
                if src.count != 3 or src.crs is None or any(v != 'uint8' for v in src.dtypes):
                    raise ValueError('Expected 3-band uint8 raster with CRS: ' + path)
                return dict(name=name, category=row['category'], image=path, width=src.width, height=src.height,
                            crs=src.crs.to_string(), transform=list(src.transform)[:6],
                            tiles=len(starts(src.width)) * len(starts(src.height)))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def read_window(item):
    path, x, y, w, h = item
    for attempt in range(3):
        try:
            if getattr(_local, 'path', None) != path:
                old = getattr(_local, 'src', None)
                if old is not None:
                    old.close()
                _local.src = rasterio.open(path)
                _local.path = path
            data = _local.src.read((1, 2, 3), window=Window(x, y, w, h))
            if data.shape != (3, h, w):
                raise ValueError('Unexpected tile shape %s' % (data.shape,))
            if not np.any(data):
                return None
            return np.ascontiguousarray(data[::-1].transpose(1, 2, 0))
        except Exception:
            if getattr(_local, 'src', None) is not None:
                _local.src.close()
                _local.path = None
                _local.src = None
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def bounded_map(pool, fn, items, prefetch=16):
    iterator = iter(items)
    queue = []
    for _ in range(prefetch):
        try:
            item = next(iterator)
        except StopIteration:
            break
        queue.append((item, pool.submit(fn, item)))
    while queue:
        item, future = queue.pop(0)
        result = future.result()
        try:
            nxt = next(iterator)
            queue.append((nxt, pool.submit(fn, nxt)))
        except StopIteration:
            pass
        yield item, result


def make_outputs(meta, arr, output, tiled, skipped, preview):
    name = meta['name']
    scene_dir = Path(output) / 'scenes' / name
    scene_dir.mkdir(parents=True, exist_ok=True)
    npz_tmp = scene_dir / 'detections.tmp.npz'
    np.savez_compressed(npz_tmp, boxes=arr, score_floor=RAW_SCORE, nms_iou=MERGE_IOU)
    os.replace(npz_tmp, scene_dir / 'detections_pixels.npz')
    with rasterio.open(meta['image']) as src:
        transform = src.transform
        crs_wkt = src.crs.to_wkt()
    ph, pw = preview.shape[:2]
    tmp_gpkg = scene_dir / 'detections.tmp.gpkg'
    if tmp_gpkg.exists():
        tmp_gpkg.unlink()
    schema = {'geometry': 'Polygon', 'properties': {'score': 'float', 'cx': 'float', 'cy': 'float', 'width_px': 'float', 'height_px': 'float', 'angle_rad': 'float', 'scene': 'str', 'category': 'str'}}
    count = 0
    with fiona.open(tmp_gpkg, mode='w', driver='GPKG', layer='buildings', schema=schema, crs_wkt=crs_wkt, encoding='UTF-8') as dst:
        records = []
        bounds = np.array([[0, 0], [meta['width'], 0], [meta['width'], meta['height']], [0, meta['height']]], dtype=np.float32)
        for row in arr:
            if float(row[5]) < OUTPUT_SCORE:
                continue
            poly = cv2.boxPoints(((float(row[0]), float(row[1])), (float(row[2]), float(row[3])), float(np.degrees(row[4]))))
            _, clipped = cv2.intersectConvexConvex(poly, bounds)
            if clipped is None or len(clipped) < 3:
                continue
            clipped = clipped.reshape(-1, 2)
            world = [tuple(float(v) for v in (transform * (float(x), float(y)))) for x, y in clipped]
            world.append(world[0])
            records.append({'geometry': {'type': 'Polygon', 'coordinates': [world]}, 'properties': {'score': float(row[5]), 'cx': float(row[0]), 'cy': float(row[1]), 'width_px': float(row[2]), 'height_px': float(row[3]), 'angle_rad': float(row[4]), 'scene': name, 'category': meta['category']}})
            draw = np.rint(poly * np.array([pw / meta['width'], ph / meta['height']])).astype(np.int32)
            cv2.polylines(preview, [draw], True, (0, 255, 0), 1, cv2.LINE_AA)
            count += 1
            if len(records) >= 1000:
                dst.writerecords(records)
                records.clear()
        if records:
            dst.writerecords(records)
    with fiona.open(tmp_gpkg, layer='buildings') as check:
        if len(check) != count or not check.crs:
            raise RuntimeError('GeoPackage verification failed for ' + name)
    os.replace(tmp_gpkg, scene_dir / 'detections.gpkg')
    thumbnail_tmp = scene_dir / 'overview.tmp.jpg'
    if not cv2.imwrite(str(thumbnail_tmp), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
        raise RuntimeError('Could not write scene preview')
    os.replace(thumbnail_tmp, scene_dir / 'overview.jpg')
    result = dict(name=name, category=meta['category'], source=meta['image'], crs=meta['crs'], image_width=meta['width'], image_height=meta['height'], tiles_total=tiled, blank_tiles_skipped=skipped, raw_merged_detections=len(arr), published_detections=count, raw_score_threshold=RAW_SCORE, output_score_threshold=OUTPUT_SCORE, nms_iou=MERGE_IOU, completed_at=time.strftime('%Y-%m-%d %H:%M:%S %z'))
    atomic_json(scene_dir / 'done.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--limit-scenes', type=int, default=0)
    parser.add_argument('--limit-patches', type=int, default=0)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(CSV, encoding='gb18030', newline='') as f:
        rows = list(csv.DictReader(f))
    if len({r['name'] for r in rows}) != len(rows):
        raise ValueError('Duplicate scene names in CSV')
    if args.limit_scenes:
        rows = rows[:args.limit_scenes]
    with ThreadPoolExecutor(max_workers=8) as pool:
        manifest = list(pool.map(describe, rows))
    atomic_json(out / 'manifest.json', {'source_csv': CSV, 'checkpoint': os.path.abspath(args.checkpoint), 'patch_size': SIZE, 'patch_stride': STRIDE, 'merge_rotated_nms_iou': MERGE_IOU, 'output_score_threshold': OUTPUT_SCORE, 'scenes': manifest})
    print('Manifest: %d scenes, %d windows' % (len(manifest), sum(x['tiles'] for x in manifest)), flush=True)
    model = init_detector(CONFIG, args.checkpoint, device='cuda:0')
    model.CLASSES = ('building',)
    failures = []
    for scene_index, meta in enumerate(manifest, 1):
        name = meta['name']
        done = out / 'scenes' / name / 'done.json'
        if done.exists():
            with open(done, encoding='utf-8') as f:
                previous = json.load(f)
            if all((done.parent / f).exists() for f in ('detections.gpkg', 'detections_pixels.npz', 'overview.jpg')) and previous.get('tiles_total') == meta['tiles'] and previous.get('source') == meta['image']:
                print('[%d/%d] %s already complete, skipping' % (scene_index, len(manifest), name), flush=True)
                continue
        try:
            windows = [(meta['image'], x, y, min(SIZE, meta['width']-x), min(SIZE, meta['height']-y)) for y in starts(meta['height']) for x in starts(meta['width'])]
            if args.limit_patches:
                windows = windows[:args.limit_patches]
            results, blank = [], 0
            ratio = min(1.0, 2048.0 / max(meta['height'], meta['width']))
            ph, pw = max(1, round(meta['height'] * ratio)), max(1, round(meta['width'] * ratio))
            preview = np.zeros((ph, pw, 3), dtype=np.uint8)
            print('[%d/%d] %s starting (%d windows)' % (scene_index, len(manifest), name, len(windows)), flush=True)
            with ThreadPoolExecutor(max_workers=6) as pool:
                for index, (window, image) in enumerate(bounded_map(pool, read_window, windows), 1):
                    if image is None:
                        blank += 1
                    else:
                        x0 = round(window[1] * pw / meta['width'])
                        y0 = round(window[2] * ph / meta['height'])
                        x1 = round((window[1] + window[3]) * pw / meta['width'])
                        y1 = round((window[2] + window[4]) * ph / meta['height'])
                        if x1 > x0 and y1 > y0:
                            preview[y0:y1, x0:x1] = cv2.resize(image, (x1-x0, y1-y0), interpolation=cv2.INTER_AREA)
                        result = np.asarray(inference_detector(model, image)[0], dtype=np.float32).reshape(-1, 6)
                        if len(result):
                            result[:, 0] += window[1]
                            result[:, 1] += window[2]
                            results.append(result)
                    if index % 50 == 0 or index == len(windows):
                        progress = {'status': 'running', 'scene': name, 'scene_index': scene_index, 'scenes_total': len(manifest), 'tiles_done': index, 'tiles_total': len(windows), 'blank_tiles': blank, 'updated_at': time.time()}
                        atomic_json(out / 'progress.json', progress)
                        print('[%d/%d] %s tiles %d/%d blank %d' % (scene_index, len(manifest), name, index, len(windows), blank), flush=True)
            detections = np.concatenate(results, axis=0) if results else np.zeros((0, 6), dtype=np.float32)
            if len(detections):
                boxes = torch.as_tensor(detections[:, :5], device='cuda:0')
                scores = torch.as_tensor(detections[:, 5], device='cuda:0')
                _, keep = nms_rotated(boxes, scores, MERGE_IOU)
                detections = detections[keep.cpu().numpy()]
            details = make_outputs(meta, detections, out, len(windows), blank, preview)
            print('[%d/%d] %s done, kept %d scored >= %.2f' % (scene_index, len(manifest), name, details['published_detections'], OUTPUT_SCORE), flush=True)
        except Exception as exc:
            print('[%d/%d] %s FAILED: %s\n%s' % (scene_index, len(manifest), name, exc, traceback.format_exc()), flush=True)
            failures.append({'scene': name, 'error': str(exc)})
    done_files = [out / 'scenes' / x['name'] / 'done.json' for x in manifest]
    successful = sum(p.exists() for p in done_files)
    report = {'status': 'completed' if not failures and successful == len(manifest) else 'failed', 'scenes_total': len(manifest), 'scenes_completed': successful, 'failures': failures, 'updated_at': time.time()}
    atomic_json(out / 'progress.json', report)
    print('Final: ' + json.dumps(report, ensure_ascii=False), flush=True)
    if report['status'] != 'completed':
        sys.exit(1)

if __name__ == '__main__':
    main()
