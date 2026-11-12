"""pytest 全局设置。

核心职责：**保证 ``backend`` 与 ``vecrot_qgis`` 从本仓库加载**。

为什么需要：本仓库可能被嵌在另一个同构仓库里（开发布局
``<dev_repo>/github_repo/``）。两个仓库都有 ``backend/`` 和 ``vecrot_qgis/``，
而 ``sys.path`` 里外层仓库的条目可能排在前面 —— 于是 ``import backend``
命中外层那份代码，测出来的结论和本仓库的实际代码无关。

症状是「每个测试文件单独跑都过，合起来跑就失败」，且报错里出现
本仓库之外的绝对路径。把本仓库根始终插到 ``sys.path[0]``，
并在进程启动时把已有的同名模块清掉，就能消除这种跨仓库串味。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# 本仓库根 + tests 目录（tests 里用 `from helpers import ...`）都要在搜索路径最前
for p in (REPO, HERE):
    while p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

# 清掉可能已被外层仓库加载的同名包，强制从本仓库重新导入
for name in ("backend", "vecrot_qgis"):
    mod = sys.modules.get(name)
    if mod is None:
        continue
    mod_file = getattr(mod, "__file__", "") or ""
    if not os.path.normcase(os.path.abspath(mod_file)).startswith(
            os.path.normcase(REPO)):
        for key in [k for k in sys.modules
                    if k == name or k.startswith(name + ".")]:
            del sys.modules[key]
