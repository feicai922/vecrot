"""Hooks for the six-epoch continuation of the gzda Strip R-CNN run."""
import glob
import json
import os
import re
import time

from mmcv.runner import HOOKS, Hook
from mmcv.runner.hooks import CheckpointHook

OFFSET = 90288
STEPS_PER_EPOCH = 7524

@HOOKS.register_module()
class GzdaContinueSetupHook(Hook):
    def __init__(self, base_lr=1e-6):
        self.base_lr = float(base_lr)

    def before_run(self, runner):
        # MMCV saves after applying the update but stores the zero-based
        # iteration. Advance once so resume neither repeats a step nor
        # triggers the just-completed validation at an epoch boundary.
        runner._iter += 1
        runner._inner_iter = runner._iter
        # AdamW's moments are restored by runner.resume; override the old
        # initial_lr before MMCV's LR hook captures its base learning rate.
        optimizers = runner.optimizer.values() if isinstance(runner.optimizer, dict) else [runner.optimizer]
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group['initial_lr'] = self.base_lr
                group['lr'] = self.base_lr
        runner.logger.info('Continuation optimizer moments preserved; base LR reset to %.8g', self.base_lr)

@HOOKS.register_module()
class GzdaContinueConsoleHook(Hook):
    def __init__(self, log_subdir='continue_6ep'):
        self.log_subdir = log_subdir

    def _write(self, runner, record):
        path = os.path.join(runner.work_dir, self.log_subdir)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'console.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

    def after_train_iter(self, runner):
        step = runner.iter + 1 - OFFSET
        if step <= 0 or step % 50:
            return
        values = dict(runner.outputs.get('log_vars', {}))
        rec = {'step': step, 'global_step': runner.iter + 1, 'epoch': (step-1)//STEPS_PER_EPOCH+1, 'ts': time.time()}
        for key in ('loss', 'loss_cls', 'loss_bbox', 'loss_rpn_cls', 'loss_rpn_bbox'):
            if key in values:
                rec[key] = float(values[key])
        rec['lr'] = float(runner.current_lr()[0])
        self._write(runner, rec)

    def after_run(self, runner):
        self._write(runner, {'step': max(0, runner.iter-OFFSET), 'global_step': runner.iter, 'event': 'done', 'ts': time.time()})

@HOOKS.register_module()
class GzdaRetentionCheckpointHook(CheckpointHook):
    def _save_checkpoint(self, runner):
        super()._save_checkpoint(runner)
        # MMCV's interval-based cleanup misses a final checkpoint whose
        # iteration isn't divisible by the checkpoint interval.
        candidates = []
        for path in glob.glob(os.path.join(self.out_dir, 'iter_*.pth')):
            match = re.fullmatch(r'iter_(\d+)\.pth', os.path.basename(path))
            if match:
                candidates.append((int(match.group(1)), path))
        for _, path in sorted(candidates, reverse=True)[2:]:
            os.remove(path)
            runner.logger.info('Removed older process checkpoint %s', path)
