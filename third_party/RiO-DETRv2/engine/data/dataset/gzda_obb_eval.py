"""单类(c building)旋转框验证器：按我们的统一评测标准算 mAP。

口径（与 rule-24 一致）：
  * 分数阈值 0.3，每图最多 100 个框（COCO maxDets）
  * IoU 阈值 0.50:0.05:0.95 共 10 点，全测试集 micro（全局按分数排序、每图一对一贪心匹配）
  * 指标：mAP@[.5:.95]、AP50、AP75

不依赖 pycocotools（它不支持旋转框），用 shapely 算旋转多边形 IoU + AABB 预筛。
"""
import os
import os.path as osp

import cv2
import numpy as np
from shapely.geometry import Polygon

from ...core import register

IOU_THRS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def _rbox_poly(cx, cy, w, h, ang_rad):
    c, s = np.cos(ang_rad), np.sin(ang_rad)
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    return [(cx + c * u - s * v, cy + s * u + c * v) for u, v in pts]


def _parse_ann(txt_path):
    """DOTA labelTxt -> list of (cx,cy,w,h,ang_rad)；与数据集解析逻辑保持一致。"""
    out = []
    if not osp.exists(txt_path):
        return out
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                poly = np.array(list(map(float, parts[:8])), dtype=np.float32).reshape(4, 2)
            except ValueError:
                continue
            (cx, cy), (bw, bh), ang = cv2.minAreaRect(poly)
            if bw <= 1 or bh <= 1:
                continue
            if bw < bh:
                bw, bh = bh, bw
                ang += 90.0
            ang = ang % 180.0
            out.append((cx, cy, bw, bh, np.deg2rad(ang)))
    return out


def _ap(scores, tps, n_gt):
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps)[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mr = np.concatenate([[0.0], rec, [1.0]])
    mp = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mp) - 2, -1, -1):
        mp[i] = max(mp[i], mp[i + 1])
    idx = np.where(mr[1:] != mr[:-1])[0]
    return float(np.sum((mr[idx + 1] - mr[idx]) * mp[idx + 1]))


@register()
class GzdaObbEvaluator(object):
    __share__ = ['output_dir']

    def __init__(self, dataset, score_thr=0.3, max_dets=100, output_dir=None):
        self.dataset = dataset
        self.score_thr = float(score_thr)
        self.max_dets = int(max_dets)
        self.output_dir = output_dir
        self.iou_types = ('bbox',)
        self.predictions = {}
        self.stats = {}
        self.current_epoch = None
        self._gt_cache = {}

    # ---- CocoEvaluator 兼容接口 ----
    def cleanup(self):
        self.predictions = {}

    def update(self, res):
        for img_id, out in res.items():
            boxes = out['boxes']
            scores = out['scores']
            if hasattr(boxes, 'detach'):
                boxes = boxes.detach().cpu().numpy()
            if hasattr(scores, 'detach'):
                scores = scores.detach().cpu().numpy()
            boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 5)
            scores = np.asarray(scores, dtype=np.float64).reshape(-1)
            keep = scores >= self.score_thr
            boxes, scores = boxes[keep], scores[keep]
            if len(scores) > self.max_dets:
                top = np.argsort(-scores)[:self.max_dets]
                boxes, scores = boxes[top], scores[top]
            self.predictions[int(img_id)] = (boxes, scores)

    def synchronize_between_processes(self):
        return

    def _image_name(self, img_id):
        try:
            return self.dataset.img_ids[img_id]
        except Exception:
            return None

    def _gt(self, img_id):
        if img_id in self._gt_cache:
            return self._gt_cache[img_id]
        name = self._image_name(img_id)
        ann_dir = getattr(self.dataset, 'ann_folder', None)
        g = []
        if name is not None and ann_dir is not None:
            g = _parse_ann(osp.join(ann_dir, name + '.txt'))
        self._gt_cache[img_id] = g
        return g

    def accumulate(self):
        # 先算 GT 总数，并缓存 GT 多边形
        self._gt_polys = {}
        n_gt = 0
        for img_id in self.predictions:
            polys = []
            for (cx, cy, w, h, ang) in self._gt(img_id):
                try:
                    p = Polygon(_rbox_poly(cx, cy, w, h, ang))
                    if not p.is_valid:
                        p = p.buffer(0)
                    if p.area > 0:
                        polys.append(p)
                except Exception:
                    continue
            self._gt_polys[img_id] = polys
            n_gt += len(polys)
        self._n_gt = n_gt
        self._results = None

    def _eval_thr(self, thr):
        scores_all, tp_all = [], []
        for img_id, (boxes, scores) in self.predictions.items():
            gt_polys = self._gt_polys.get(img_id, [])
            if len(boxes) == 0:
                continue
            det_polys, det_ab = [], []
            for (cx, cy, w, h, ang) in boxes:
                poly = _rbox_poly(cx, cy, w, h, ang)
                det_polys.append(Polygon(poly))
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                det_ab.append((min(xs), min(ys), max(xs), max(ys)))
            used = np.zeros(len(gt_polys), bool)
            gt_ab = []
            for g in gt_polys:
                x0, y0, x1, y1 = g.bounds
                gt_ab.append((x0, y0, x1, y1))
            order = np.argsort(-scores)
            tp = np.zeros(len(order), dtype=np.float64)
            for k, i in enumerate(order):
                a = det_ab[i]
                best, bj = 0.0, -1
                for j, b in enumerate(gt_ab):
                    if used[j]:
                        continue
                    if a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]:
                        continue
                    try:
                        inter = det_polys[i].intersection(gt_polys[j]).area
                    except Exception:
                        continue
                    if inter <= 0:
                        continue
                    u = det_polys[i].area + gt_polys[j].area - inter
                    v = inter / u if u > 0 else 0.0
                    if v > best:
                        best, bj = v, j
                if best >= thr and bj >= 0:
                    used[bj] = True
                    tp[k] = 1.0
            scores_all.append(np.asarray(scores)[order])
            tp_all.append(tp)
        if not scores_all:
            return 0.0
        return _ap(np.concatenate(scores_all), np.concatenate(tp_all), self._n_gt)

    def summarize(self):
        if self._results is None:
            aps = [self._eval_thr(t) for t in IOU_THRS]
            i75 = int(np.where(IOU_THRS == 0.75)[0][0])
            self._results = {'mAP': float(np.mean(aps)),
                             'AP50': float(aps[0]),
                             'AP75': float(aps[i75])}
        self.stats = dict(self._results)
        print('[GzdaObbEvaluator] epoch=%s  n_gt=%d  mAP=%.4f  AP50=%.4f  AP75=%.4f  (score>=%.2f)'
              % (self.current_epoch, getattr(self, '_n_gt', 0), self.stats['mAP'],
                 self.stats['AP50'], self.stats['AP75'], self.score_thr), flush=True)
        return self.stats
