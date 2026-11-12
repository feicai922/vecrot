"""场景分支的训练期日志 Hook（重建版，仅让配置能解析）。

原始 ``scene_hooks.py`` 不在仓库与上游中。配置里的 ``custom_hooks`` 引用了
``SceneConsoleHook``，而 ``custom_imports`` 是 ``allow_failed_imports=False``，
所以这个模块**必须存在**，否则整个配置无法 ``Config.fromfile``。

推理部署不需要任何训练 Hook，因此这里的实现只保证：

* 类名与注册名匹配（``@HOOKS.register_module()`` + ``SceneConsoleHook``）
* 有 ``after_train_iter`` / ``after_run`` 等 Hook 生命周期方法
* 行为是最小化的控制台日志（与 ``GzdaConsoleHook`` 同风格）

**这不是原训练的数值复现**：原实现可能统计了场景分支的 loss/acc 等指标。
训练期指标与部署推理无关，故此处不复现。
"""
from __future__ import annotations

import json
import os
import time

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SceneConsoleHook(Hook):
    """每个打印周期把训练进度（含场景分支 loss）追加到 console.jsonl。"""

    def __init__(self, interval: int = 50, out_name: str = "console.jsonl"):
        self.interval = int(interval)
        self.out_name = out_name

    def after_train_iter(self, runner):
        if (runner.iter + 1) % self.interval:
            return
        values = dict(getattr(runner, "outputs", {}).get("log_vars", {}))
        record = {
            "step": runner.iter + 1,
            "epoch": runner.epoch + 1,
            "ts": time.time(),
        }
        for key in ("loss", "loss_cls", "loss_bbox", "loss_rpn_cls",
                    "loss_rpn_bbox", "loss_scene", "acc_scene"):
            if key in values:
                record[key] = float(values[key])
        try:
            record["lr"] = float(runner.current_lr()[0])
        except (IndexError, TypeError):
            pass
        path = os.path.join(runner.work_dir, self.out_name)
        os.makedirs(runner.work_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def after_run(self, runner):
        path = os.path.join(runner.work_dir, self.out_name)
        os.makedirs(runner.work_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": runner.iter, "event": "done",
                                "ts": time.time()}) + "\n")
