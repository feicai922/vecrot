"""固定模型搭配（preset registry）—— **兼容层**。

历史沿革：本文件曾经把每个模型的权重/配置/类别数写成 Python 常量，理由是
"权重、配置、预处理、类别数必须成套"。这个理由仍然成立，但载体换了：

* 现在权威来源是**模型文件夹**：``vecrot/models/<任意名>/model.json``
  声明自己的 backend / 权重 / 配置 / 类别数 / 阈值，见 ``model_registry``；
* 加一个模型不再改代码，拷贝文件夹即可，面板自动多一项。

本文件只保留旧的对外函数名（``available_presets`` / ``get_preset`` /
``describe`` / ``refresh_repo_root`` / ``PRESETS``），避免 UI、测试与部署脚本
全线改名。真正的逻辑全在 ``model_registry.py``。

## 为什么它不是唯一实现

插件装在 QGIS profile 的 ``python/plugins`` 下，那里没有 ``backend`` 包。
所以插件自带一份**逐字节相同**的 ``model_registry.py``，并用
``tests/test_registry_parity.py`` 守住两份不漂移。
"""
from __future__ import annotations

import os

from . import model_registry as _mr
from .model_registry import (          # noqa: F401 - 供调用方直接 import
    BACKEND_LABELS,
    SPEC_FILENAME,
    ModelPreset,
    ModelSpecError,
    default_models_root,
    find_repo_root,
    load_spec,
    resolve_path,
    scan,
    validate_folder,
)

#: 仓库根（本文件在 backend/contracts/ 下，向上三级）
_REPO = _mr.find_repo_root(os.path.dirname(os.path.abspath(__file__)))

#: 模型文件夹根目录
_MODELS_ROOT = _mr.default_models_root(_REPO)


def _rebuild(repo_root: str | None = None, models_root: str | None = None) -> None:
    """重新扫描并替换模块级常量（PRESETS / _BY_KEY / _SKIPPED）。"""
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
    """重新解析仓库根并重扫模型文件夹，返回解析到的仓库根。

    插件侧需要在定位到 manifest 之后再调一次（模块级常量在 import 时求值，
    那时可能还不知道仓库在哪）。
    """
    _rebuild(repo_root=root)
    return _REPO


def refresh_models_root(models_root: str) -> str:
    """显式指定模型根目录（自检脚本 / 用户自定义部署用）。"""
    _rebuild(models_root=models_root)
    return _MODELS_ROOT


def models_root() -> str:
    return _MODELS_ROOT


def repo_root() -> str:
    return _REPO


def skipped() -> list[dict]:
    """没被列进来的文件夹及原因（面板上要显示，否则用户以为装坏了）。"""
    return list(_SKIPPED)


def available_presets(only_available: bool = True) -> list[ModelPreset]:
    """预设列表；默认只返回文件齐备的（精简包会缺部分权重）。"""
    if not only_available:
        return list(PRESETS)
    return [p for p in PRESETS if p.is_available()]


def get_preset(key: str) -> ModelPreset | None:
    return _BY_KEY.get(key)


def describe() -> list[dict]:
    """给面板/自检用的可读清单（含缺件说明）。"""
    return [p.describe() for p in PRESETS]


# 首次 import 时扫一遍
_rebuild()


if __name__ == "__main__":
    import json

    print(json.dumps({"models_root": _MODELS_ROOT, "presets": describe(),
                      "skipped": skipped()}, ensure_ascii=False, indent=2))
