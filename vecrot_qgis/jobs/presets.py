"""插件侧的模型搭配列表（**不依赖 backend 包**）。

为什么要有这份东西
------------------
插件安装在 QGIS profile 的 ``python/plugins`` 下，那里没有 ``backend`` 包，
`sys.path` 也不包含仓库根，所以插件不能 ``from backend.contracts.presets import ...``。

现在模型列表由**模型文件夹**驱动（``vecrot/models/<名>/model.json``），
扫描逻辑在 ``jobs/model_registry.py``——它与
``backend/contracts/model_registry.py`` **逐字节相同**，由
``tests/test_registry_parity.py`` 守住。本文件只做两件插件特有的事：

1. **定位仓库根**：插件在 profile 里时只能靠 manifest 的 ``repo_root``；
2. 把扫描结果暴露成与 backend 侧同名的函数（``available_presets`` / ``describe`` …）。

这样"加一个模型"只需要建一个文件夹，两边都不用改代码。
"""
from __future__ import annotations

import os

from . import model_registry as _mr
from .model_registry import (          # noqa: F401 - 供面板直接 import
    BACKEND_LABELS,
    SPEC_FILENAME,
    ModelPreset,
    ModelSpecError,
    default_models_root,
    load_spec,
    resolve_path,
    scan,
    validate_folder,
)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root() -> str:
    """确定模型所在仓库根。

    查找顺序（实测：插件装在 profile 里时前两条会失败，必须靠第三、四条）：

    1. 向上找含 ``backend/`` 的目录（开发时插件就在仓库内）
    2. 同上，找含 ``deployment/`` 的目录
    3. QGIS 设置 ``vecrot/manifest_path`` 指向的 manifest 里的 ``repo_root``
    4. 环境变量 ``VECROT_REPO_ROOT``

    都找不到时返回 ``_HERE`` —— 此时所有模型会显示为"缺文件"，
    用户能看到明确提示而不是静默失败。
    """
    cur = _HERE
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, "backend")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    cur = _HERE
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, "deployment")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # 3) 从 QGIS 设置里的 manifest 副本读 repo_root
    root = _root_from_manifest_setting()
    if root:
        return root

    # 4) 环境变量兜底
    env = os.environ.get("VECROT_REPO_ROOT")
    if env and os.path.isdir(env):
        return env

    return _HERE


def _manifest_candidates() -> list[str]:
    """候选 manifest 路径，按可靠性排序。

    为什么不能只依赖 QgsSettings：``QgsSettings()`` 的存储位置取决于
    QGIS 应用名与组织名（实测在 headless + o4w_env 下会落到
    ``.../Roaming/python/profiles/default/Unknown Organization/python.ini``），
    而正常启动的 QGIS 写的是 ``.../QGIS/QGIS3/profiles/default/QGIS/QGIS3.ini``。
    两者不是同一个文件 —— 安装脚本写进前者、QGIS 读后者时就读不到。

    所以这里同时读多个来源，并对**已知部署位置**做文件系统扫描兜底。
    """
    out: list[str] = []

    # 1) 环境变量
    env = os.environ.get("VECROT_MANIFEST")
    if env:
        out.append(env)

    # 2) QgsSettings，两种作用域都试
    try:
        from qgis.core import QgsSettings

        for ctor_args in ((), ("QGIS", "QGIS3")):
            try:
                s = QgsSettings(*ctor_args)
                v = s.value("vecrot/manifest_path", "", type=str)
                if v:
                    out.append(v)
            except Exception:              # noqa: BLE001
                pass
    except Exception:                      # noqa: BLE001 - 无 QGIS
        pass

    # 3) 文件系统扫描：安装脚本放置的固定位置 + 仓库内
    home = os.path.expanduser("~")
    out += [
        os.path.join(home, "vecrot", "deployment", "manifest.json"),
        os.path.join("C:\\", "vecrot", "deployment", "manifest.json"),
        os.path.join("D:\\", "vecrot", "deployment", "manifest.json"),
    ]
    cur = _HERE
    for _ in range(8):
        out.append(os.path.join(cur, "deployment", "manifest.json"))
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return out


def _root_from_manifest_setting() -> str | None:
    """从候选 manifest 里读 ``repo_root``（第一个有效的胜出）。"""
    import json

    for mp in _manifest_candidates():
        if not mp or not os.path.isfile(mp):
            continue
        try:
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, ValueError):
            continue
        root = m.get("repo_root")
        if isinstance(root, str) and root:
            cand = root if os.path.isabs(root) else os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(mp)), root))
            if os.path.isdir(cand):
                return cand
    return None


#: 当前使用的仓库根 / 模型根。模块 import 时先算一次，之后可 refresh。
_REPO = _repo_root()
_MODELS_ROOT = _mr.default_models_root(_REPO)
PRESETS: tuple = ()
_BY_KEY: dict = {}
_SKIPPED: list = []


def _rebuild(repo_root: str | None = None, models_root: str | None = None) -> None:
    global _REPO, _MODELS_ROOT, PRESETS, _BY_KEY, _SKIPPED
    if repo_root:
        _REPO = os.path.normpath(repo_root)
    if models_root:
        _MODELS_ROOT = os.path.normpath(models_root)
    else:
        _MODELS_ROOT = _mr.default_models_root(_REPO)
    found, dropped = _mr.scan(_MODELS_ROOT, _REPO)
    PRESETS = tuple(found) + tuple(_mr.builtin_presets(_MODELS_ROOT, _REPO))
    _BY_KEY = {p.key: p for p in PRESETS}
    _SKIPPED = dropped


def refresh_repo_root(root: str | None = None) -> str:
    """重新解析仓库根并**重扫**模型文件夹。

    为什么需要：模块级常量在 import 时求值，而 QGIS 插件 import 发生在
    manifest 定位之前。若 manifest 里有 repo_root（插件装在 profile 时的唯一
    可靠线索），必须等 manifest 定位后再解析一次，否则所有路径会拼到插件目录下
    （实战报错：全部显示"缺文件"）。

    ``root`` 传了就**直接用它**，跳过自动发现。
    """
    _rebuild(repo_root=root or _repo_root())
    return _REPO


def refresh_models_root(models_root: str) -> str:
    _rebuild(models_root=models_root)
    return _MODELS_ROOT


def models_root() -> str:
    return _MODELS_ROOT


def repo_root() -> str:
    return _REPO


def skipped() -> list[dict]:
    """没被列进来的模型文件夹及原因（面板要显示）。"""
    return list(_SKIPPED)


def available_presets(only_available: bool = True) -> list:
    """预设列表。

    默认只返回**文件齐备**的：精简分发包可能不含某些权重，此时在面板上列出
    「缺文件」的选项只会让用户以为装坏了。排查缺件时用 ``only_available=False``。
    """
    if not only_available:
        return list(PRESETS)
    return [p for p in PRESETS if p.is_available()]


def get_preset(key: str):
    return _BY_KEY.get(key)


def describe() -> list:
    """给面板/自检用的可读清单（含缺件说明）。"""
    return [p.describe() for p in PRESETS]


_rebuild()
