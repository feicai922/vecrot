import json
import os
import time
import numpy as np
from mmcv.runner import HOOKS, Hook
from mmrotate.datasets.builder import ROTATED_DATASETS
from mmrotate.datasets.dota import DOTADataset

@ROTATED_DATASETS.register_module()
class GzdaRBoxDataset(DOTADataset):
    CLASSES = ('building',)
    PALETTE = [(0, 255, 0)]

    def load_annotations(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        boxes = {im['id']: [] for im in data['images']}
        for ann in data['annotations']:
            b = ann['bbox']
            if ann.get('iscrowd', 0) or len(b) != 5 or b[2] <= 0 or b[3] <= 0:
                continue
            boxes[ann['image_id']].append(b)
        infos = []
        for im in data['images']:
            b = np.asarray(boxes[im['id']], dtype=np.float32).reshape(-1, 5)
            infos.append(dict(filename=im['file_name'], width=im['width'], height=im['height'], ann=dict(
                bboxes=b, labels=np.zeros(len(b), dtype=np.int64),
                bboxes_ignore=np.zeros((0, 5), dtype=np.float32), labels_ignore=np.zeros(0, dtype=np.int64))))
        self.img_ids = [os.path.splitext(x['filename'])[0] for x in infos]
        return infos

@HOOKS.register_module()
class GzdaConsoleHook(Hook):
    def after_train_iter(self, runner):
        if (runner.iter + 1) % 50:
            return
        values = dict(runner.outputs.get('log_vars', {}))
        record = {'step': runner.iter + 1, 'epoch': runner.epoch + 1, 'ts': time.time()}
        for key in ('loss', 'loss_cls', 'loss_bbox', 'loss_rpn_cls', 'loss_rpn_bbox'):
            if key in values:
                record[key] = float(values[key])
        record['lr'] = float(runner.current_lr()[0])
        with open(os.path.join(runner.work_dir, 'console.jsonl'), 'a') as f:
            f.write(json.dumps(record) + '\n')
    def after_run(self, runner):
        with open(os.path.join(runner.work_dir, 'console.jsonl'), 'a') as f:
            f.write(json.dumps({'step': runner.iter, 'event': 'done', 'ts': time.time()}) + '\n')
