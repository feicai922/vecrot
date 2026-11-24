"""插件侧模型列表（presets）的契约测试。

历史：这个文件原本守的是"backend 与插件两份**手写常量**不漂移"。现在模型列表
由**模型文件夹**驱动（``vecrot/models/<名>/model.json``），常量不存在了。
两件事分开守：

* 扫描逻辑只有一份实现，插件的副本必须**逐字节相同**
  → ``tests/test_model_registry.py::test_registry_mirror_is_byte_identical``
  与 ``test_both_sides_scan_identically``；
* 本文件守**插件侧薄适配层**：仓库根解析、刷新语义、面板要用的函数与字段。

这些点都是实际踩过的坑（插件装在 QGIS profile 里时找不到仓库根 → 全部显示
"缺文件"；重写文件时误删函数 → 面板初始化卡死），所以单独留一组回归测试。
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import make_model_folder, plugin_parent, plugin_dir   # noqa: E402

sys.path.insert(0, plugin_parent())


@pytest.fixture()
def models_root(tmp_path):
    """一个临时 models 根：一个齐备模型 + 一个缺权重的 + 一个没 model.json 的。"""
    root = str(tmp_path)
    make_model_folder(root, "good", {
        "spec_version": 1, "key": "good_model", "label": "齐备模型",
        "backend": "strip_s_gzda", "num_classes": 1,
        "weights": "best.pth", "config": "cfg.py",
        "detector_options": {"score_thr": 0.05},
    }, {"best.pth": 32, "cfg.py": 8})
    make_model_folder(root, "incomplete", {
        "spec_version": 1, "key": "incomplete_model", "label": "缺权重模型",
        "backend": "strip_s_gzda", "num_classes": 1, "weights": "gone.pth",
    })
    os.makedirs(os.path.join(root, "no_spec_here"))
    return root


def _plugin_presets():
    from vecrot_qgis.jobs import presets as pp

    return pp


def test_preset_functions_exist():
    """对外函数必须存在（曾被误删导致面板初始化卡死）。"""
    pp = _plugin_presets()
    for fn in ("available_presets", "get_preset", "describe",
               "refresh_repo_root", "refresh_models_root", "skipped",
               "models_root", "repo_root"):
        assert callable(getattr(pp, fn, None)), f"presets 缺少 {fn}()"


def test_plugin_can_locate_repo_root():
    """插件侧必须能定位到仓库根（否则所有模型都不可用）。"""
    root = _plugin_presets()._repo_root()
    assert os.path.isdir(os.path.join(root, "backend")), (
        f"插件推断的仓库根不对：{root}（应含 backend/ 目录）")


def test_refresh_models_root_rebuilds(models_root):
    """refresh_models_root() 必须真能重建列表（面板靠它切换/重扫）。"""
    pp = _plugin_presets()
    saved = pp.models_root()
    try:
        assert pp.refresh_models_root(models_root) == os.path.normpath(models_root)
        keys = [p.key for p in pp.PRESETS]
        assert "good_model" in keys and "incomplete_model" in keys
        assert "vector_only" in keys, "内置 entry 不能因为没扫到而消失"
        for p in pp.PRESETS:
            assert pp.get_preset(p.key) is p, f"{p.key} 未进 _BY_KEY"
    finally:
        pp.refresh_models_root(saved)


def test_available_presets_filters_missing(models_root):
    pp = _plugin_presets()
    saved = pp.models_root()
    try:
        pp.refresh_models_root(models_root)
        avail = [p.key for p in pp.available_presets()]
        every = [p.key for p in pp.available_presets(only_available=False)]
        assert "good_model" in avail
        assert "incomplete_model" not in avail
        assert "incomplete_model" in every
    finally:
        pp.refresh_models_root(saved)


def test_skipped_explains_folders_without_spec(models_root):
    """没 model.json 的文件夹必须给出原因，否则用户以为模型包坏了。"""
    pp = _plugin_presets()
    saved = pp.models_root()
    try:
        pp.refresh_models_root(models_root)
        dropped = pp.skipped()
        assert any("no_spec_here" in d["folder"] for d in dropped)
        assert all(d["reason"] for d in dropped)
    finally:
        pp.refresh_models_root(saved)


def test_describe_exposes_panel_fields(models_root):
    pp = _plugin_presets()
    saved = pp.models_root()
    try:
        pp.refresh_models_root(models_root)
        info = {d["key"]: d for d in pp.describe()}
        good = info["good_model"]
        for field in ("key", "label", "backend", "backend_label",
                      "available", "missing", "num_classes", "note"):
            assert field in good, f"describe() 缺字段 {field}"
        assert good["available"] is True and good["missing"] == []
        assert info["incomplete_model"]["available"] is False
    finally:
        pp.refresh_models_root(saved)


def test_paths_are_absolute(models_root):
    """所有解析后的路径必须是绝对路径。

    历史事故：插件装在 profile 里时向上找不到仓库根，就用插件目录拼相对路径，
    得到 ``.../plugins/vecrot_qgis/jobs/vecrot/models/...`` 这种不存在的路径，
    面板上所有文件都显示 ✗，长路径还把面板撑到超屏。
    """
    pp = _plugin_presets()
    saved = pp.models_root()
    try:
        pp.refresh_models_root(models_root)
        for p in pp.PRESETS:
            for role, path in p.files.items():
                assert os.path.isabs(path), f"{p.key}.{role} 不是绝对路径：{path}"
    finally:
        pp.refresh_models_root(saved)


def test_model_ordering_comes_from_order_field(tmp_path):
    """面板第一项由 model.json 的 order 决定（默认推荐模型排最前）。"""
    root = str(tmp_path)
    make_model_folder(root, "zzz_recommended", {
        "spec_version": 1, "key": "recommended", "label": "推荐",
        "backend": "strip_s_gzda", "num_classes": 1,
        "weights": "best.pth", "order": 10}, {"best.pth": 8})
    make_model_folder(root, "aaa_other", {
        "spec_version": 1, "key": "other", "label": "其他",
        "backend": "strip_s_gzda", "num_classes": 1,
        "weights": "best.pth", "order": 20}, {"best.pth": 8})
    pp = _plugin_presets()
    saved = pp.models_root()
    try:
        pp.refresh_models_root(root)
        assert [p.key for p in pp.PRESETS][:2] == ["recommended", "other"]
    finally:
        pp.refresh_models_root(saved)


def test_init_order_manifest_before_presets():
    """面板初始化必须先生成 manifest 配置，再加载模型列表。

    这是个顺序依赖：模型路径要靠 manifest 里的 repo_root 解析。
    用源码静态检查守住——运行时很难在单测里复现 QGIS 启动顺序。
    """
    src = open(os.path.join(plugin_dir(), "ui", "dock_widget.py"),
               encoding="utf-8").read()
    i_manifest = src.index("self._load_default_manifest()")
    i_presets = src.index("self._reload_presets()          # 必须在 manifest 之后")
    assert i_manifest < i_presets, (
        "_load_default_manifest() 必须在 _reload_presets() 之前调用，"
        "否则模型路径会按插件目录解析（历史 bug）")


def test_job_fields_shape(models_root):
    """to_job_fields() 必须产出 job.json 认识的字段名。"""
    pp = _plugin_presets()
    saved = pp.models_root()
    try:
        pp.refresh_models_root(models_root)
        fields = pp.get_preset("good_model").to_job_fields()
        assert fields["detector_profile"] == "strip_s_gzda"
        assert fields["detector_num_classes"] == 1
        assert os.path.isabs(fields["detector_weights"])
        assert fields["detector_options"] == {"score_thr": 0.05}
    finally:
        pp.refresh_models_root(saved)
