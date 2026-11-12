_base_ = '../strip_rcnn/strip_rcnn_s_fpn_1x_dota_le90.py'
custom_imports = dict(imports=['configs.gzda.gzda_dataset'], allow_failed_imports=False)
root = '/data1/wzh26/data/gzda_v3_human_del/'
model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='/data1/wzh26/model_zoo/stripnet_s_pretrain.pth.tar'), norm_cfg=dict(type='BN', requires_grad=True)),
    roi_head=dict(bbox_head=dict(num_classes=1)),
    train_cfg=dict(rpn=dict(allowed_border=-1)),
    test_cfg=dict(rcnn=dict(score_thr=0.05, nms=dict(iou_thr=0.3), max_per_img=2000)))
data = dict(
    samples_per_gpu=2, workers_per_gpu=4,
    train=dict(type='GzdaRBoxDataset', ann_file=root+'annotations/instances_train_rbox.json', img_prefix=root+'images/train/', filter_empty_gt=False),
    val=dict(type='GzdaRBoxDataset', ann_file=root+'annotations/instances_val_rbox.json', img_prefix=root+'images/val/', filter_empty_gt=False),
    test=dict(type='GzdaRBoxDataset', ann_file=root+'annotations/instances_test_rbox.json', img_prefix=root+'images/test/', filter_empty_gt=False))
optimizer = dict(lr=0.000025)
checkpoint_config = dict(interval=1000, by_epoch=False, max_keep_ckpts=2)
evaluation = dict(interval=1, metric='mAP', save_best='mAP', rule='greater', nproc=4)
custom_hooks = [dict(type='GzdaConsoleHook')]
# Preserve empty negative tiles even when random rotation is applied.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=[0.25, 0.25, 0.25], direction=['horizontal', 'vertical', 'diagonal'], version='le90'),
    dict(type='PolyRandomRotate', rotate_ratio=0.5, angles_range=180, auto_bound=False, allow_negative=True, version='le90'),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data['train']['pipeline'] = train_pipeline
# An iteration runner saves and resumes exact optimizer/iteration progress.
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=90288)
lr_config = dict(_delete_=True, policy='step', by_epoch=False, warmup='linear', warmup_iters=500, warmup_ratio=1.0/3, step=[60192, 82764])
evaluation = dict(interval=7524, metric='mAP', save_best='mAP', rule='greater', nproc=4)
