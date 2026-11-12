import os
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector

def main():
    cfg = Config.fromfile('configs/gzda/strip_s_gzda.py')
    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    original = dataset.data_infos
    indices = list(range(12))
    indices += [next(i for i, info in enumerate(original) if min(info['width'], info['height']) <= 1)]
    indices += [next(i for i, info in enumerate(original) if len(info['ann']['bboxes']) == 0)]
    indices = list(dict.fromkeys(indices))
    dataset.data_infos = [original[i] for i in indices]
    dataset.img_ids = [os.path.splitext(info['filename'])[0] for info in dataset.data_infos]
    print('Smoke validation:', [(i, original[i]['filename'], original[i]['width'], original[i]['height'], len(original[i]['ann']['bboxes'])) for i in indices], flush=True)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, 'work_dirs/gzda_strip_s/iter_7000.pth', map_location='cpu')
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    results = single_gpu_test(model, loader, show=False)
    print('Prediction counts:', [sum(len(a) for a in result) for result in results], flush=True)
    print('Evaluation:', dataset.evaluate(results, metric='mAP', nproc=1), flush=True)

if __name__ == '__main__':
    main()
