_base_ = './strip_s_gzda.py'
custom_imports = dict(imports=['configs.gzda.gzda_dataset', 'configs.gzda.continue_hooks'], allow_failed_imports=False)
# Resume iteration 90,288 and train approximately six more full passes.
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=135432)
optimizer = dict(lr=0.000001)
lr_config = dict(_delete_=True, policy='step', by_epoch=False, warmup=None, step=[112860, 127908], gamma=0.5)
checkpoint_config = dict(type='GzdaRetentionCheckpointHook', interval=1000, by_epoch=False, max_keep_ckpts=2)
evaluation = dict(interval=7524, metric='mAP', save_best='mAP', rule='greater', nproc=4)
custom_hooks = [dict(type='GzdaContinueSetupHook', base_lr=0.000001, priority='HIGHEST'), dict(type='GzdaContinueConsoleHook')]
