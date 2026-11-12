"""守卫两份模型预设的一致性。

插件侧（``qgis_plugin/vecrot_qgis/jobs/presets.py``）必须能在没有 backend 包的环境里
独立工作，所以它是 ``backend/contracts/presets.py`` 的**镜像**。这种冗余的风险是
两边逐渐漂移：加了新模型只改一边、或者改了权重路径忘了同步。

本测试把漂移变成红灯：两边的 key 集合、backend、num_classes、文件路径必须完全一致。
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import plugin_parent, plugin_dir   # noqa: E402

sys.path.insert(0, plugin_parent())


def _backend_presets():
    from backend.contracts import presets as bp

    return {p.key: p for p in bp.PRESETS}


def _plugin_presets():
    from vecrot_qgis.jobs import presets as pp

    # 插件侧 ``_REPO`` 在 import 时求值。如果本仓库恰好放在另一个同构仓库里
    # （开发布局 ``<dev_repo>/github_repo/``），自动发现会先命中外层仓库，
    # 于是和 backend 侧的路径对不上 —— 那是环境问题，不是两份预设漂移。
    # 显式指定本仓库根，测的才是「预设本身是否一致」。
    pp.refresh_repo_root(REPO)
    return {p.key: p for p in pp.PRESETS}


def test_same_keys():
    a, b = _backend_presets(), _plugin_presets()
    assert set(a) == set(b), (
        "两份预设的 key 不一致：backend="
        f"{sorted(set(a) - set(b))} plugin={sorted(set(b) - set(a))}")


@pytest.mark.parametrize("key", list(_backend_presets()))
def test_same_metadata(key):
    a = _backend_presets()[key]
    b = _plugin_presets()[key]
    assert a.backend == b.backend, f"{key}: backend 不一致"
    assert a.num_classes == b.num_classes, f"{key}: num_classes 不一致"
    assert a.label == b.label, f"{key}: label 不一致"


@pytest.mark.parametrize("key", list(_backend_presets()))
def test_same_files(key):
    a = _backend_presets()[key]
    b = _plugin_presets()[key]
    assert set(a.files) == set(b.files), (
        f"{key}: 文件角色不一致 backend={sorted(a.files)} plugin={sorted(b.files)}")
    for role in a.files:
        pa, pb = a.files[role], b.files[role]
        assert os.path.normcase(os.path.normpath(pa)) == \
            os.path.normcase(os.path.normpath(pb)), (
            f"{key}.{role}: 路径不一致\n  backend={pa}\n  plugin={pb}")


def test_default_preset_is_scene():
    """默认选中的必须是场景版（大范围推理推荐）。只查顺序，不要求文件已下载。"""
    from backend.contracts import presets as bp

    p = bp.PRESETS[0]
    assert p.key == "strip_s_gzda_scene", f"第一个预设应是场景版，实际 {p.key}"


def test_default_preset_available_when_files_present():
    """文件下载齐全时，默认预设必须可用。

    文件没下载（新 clone 还没拉模型）时跳过：那是环境状态，不是代码缺陷。
    """
    from backend.contracts import presets as bp

    p = bp.PRESETS[0]
    miss = p.missing()
    if miss:
        pytest.skip("模型未下载，见 docs/模型下载.md：" + "；".join(miss))
    assert p.is_available()


def test_placeholder_preset_is_not_available():
    """空 files 的占位条目不能被判为可用。"""
    p = _backend_presets().get("rtdetr_r50")
    if p is not None and not p.files:
        assert not p.is_available(), "占位预设不应显示为可用"


def test_preset_paths_are_absolute():
    """所有预设的文件路径必须是绝对路径。

    历史上出过一次：插件安装在 QGIS profile 里时，向上找不到仓库根，
    就用插件目录拼接相对路径，得到
    ``.../plugins/vecrot_qgis/jobs/vecrot/models/...`` 这种不存在的路径，
    面板上所有文件都显示 ✗，且长路径把面板撑到超屏。
    """
    for key, p in _plugin_presets().items():
        for role, path in p.files.items():
            assert os.path.isabs(path), f"{key}.{role} 不是绝对路径：{path}"


def test_plugin_can_locate_repo_root():
    """插件侧必须能定位到仓库根（否则所有预设都不可用）。"""
    from vecrot_qgis.jobs import presets as pp

    root = pp._repo_root()
    assert os.path.isdir(os.path.join(root, "backend")), (
        f"插件推断的仓库根不对：{root}（应含 backend/ 目录）")


def test_default_preset_resolves_under_repo():
    """默认预设的文件路径应在仓库根之下（防止拼到插件目录里）。"""
    from vecrot_qgis.jobs import presets as pp

    root = os.path.normcase(pp._repo_root())
    p = pp.PRESETS[0]
    for role, path in p.files.items():
        assert os.path.normcase(path).startswith(root), (
            f"{role} 解析到仓库外：{path}（仓库根 {root}）")


def test_refresh_repo_root_rebuilds_paths():
    """refresh_repo_root() 必须真能重建预设路径。

    历史事故：面板初始化时先调 _reload_presets() 再定位 manifest，
    导致 _REPO 在模块 import 时就被定成插件目录，所有预设显示"缺文件"。
    修复后 refresh_repo_root() 会重新解析并重建 PRESETS。
    """
    from vecrot_qgis.jobs import presets as pp

    before = pp._REPO
    try:
        after = pp.refresh_repo_root()
        assert os.path.isdir(os.path.join(after, "backend")), (
            f"refresh 后的仓库根不含 backend/：{after}")
        # 重建后 PRESETS / _BY_KEY 必须一致
        assert len(pp.PRESETS) == len(pp._BY_KEY)
        for p in pp.PRESETS:
            assert pp.get_preset(p.key) is p, f"{p.key} 未进 _BY_KEY"
    finally:
        # 必须复原：pytest 里所有测试共用一个进程，这个测试调的是**无参**
        # refresh（走自动发现），在嵌套仓库布局下会解析到外层仓库。
        # 不复原就污染后续 test_same_files 的路径比对。
        pp.refresh_repo_root(before)


def test_init_order_manifest_before_presets():
    """面板初始化必须先生成 manifest 配置，再加载预设。

    这是个顺序依赖：预设路径要靠 manifest 里的 repo_root 解析。
    用源码静态检查守住——运行时很难在单测里复现 QGIS 启动顺序。
    """
    src = open(
        os.path.join(plugin_dir(), "ui", "dock_widget.py"),
        encoding="utf-8").read()
    i_manifest = src.index("self._load_default_manifest()")
    i_presets = src.index("self._reload_presets()          # 必须在 manifest 之后")
    assert i_manifest < i_presets, (
        "_load_default_manifest() 必须在 _reload_presets() 之前调用，"
        "否则预设路径会按插件目录解析（历史 bug）")


def test_preset_functions_exist():
    """三个对外函数必须存在（曾被误删导致面板卡死）。"""
    from vecrot_qgis.jobs import presets as pp

    for fn in ("available_presets", "get_preset", "describe",
               "refresh_repo_root"):
        assert callable(getattr(pp, fn, None)), f"presets 缺少 {fn}()"
