"""场景分类分支的契约测试（不需要 GPU / mmcv）。

真实推理在 ``deployment/verify_scene_branch.py`` 里跑（需要 detector 环境）。
这里只守「配置契约」——那是上一次 RT-DETR / gzda 集成时最容易出错的地方。
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import plugin_parent   # noqa: E402

sys.path.insert(0, plugin_parent())

SCENE_DIR = os.path.join(REPO, "vecrot", "models", "detector_strip_s_scene")
SCENE_CFG = os.path.join(SCENE_DIR, "strip_s_gzda_scene.py")
SCENE_W = os.path.join(SCENE_DIR, "best_mAP_iter_37620.pth")
GZDA_DIR = os.path.join(REPO, "vecrot", "models", "detector_strip_s",
                        "Strip-R-CNN", "configs", "gzda")

# 场景版推理配置随仓库分发（10 KB，是源码不是权重）；权重和 Strip-R-CNN
# 源码要靠网盘下载。没下载时跳过依赖它们的用例，而不是报失败 ——
# 否则新 clone 的人跑 pytest 会以为代码有问题。
HAS_WEIGHTS = os.path.isfile(SCENE_W)
HAS_SOURCE = os.path.isdir(GZDA_DIR)
needs_weights = pytest.mark.skipif(
    not HAS_WEIGHTS, reason="未下载场景权重，见 docs/模型下载.md")
needs_source = pytest.mark.skipif(
    not HAS_SOURCE, reason="未下载 Strip-R-CNN 源码，见 docs/模型下载.md")


@needs_weights
def test_scene_assets_exist():
    assert os.path.isfile(SCENE_CFG), f"缺场景配置：{SCENE_CFG}"
    assert os.path.isfile(SCENE_W), f"缺场景权重：{SCENE_W}"


def test_scene_config_declares_scene_head():
    src = open(SCENE_CFG, encoding="utf-8").read()
    assert "StripRCNNScene" in src, "配置应使用 StripRCNNScene 模型类"
    assert "scene_head" in src, "配置应声明 scene_head"
    # 与重建的 SceneHead 形状对应
    assert "num_levels=5" in src.replace(" ", "")
    assert "hidden_channels=512" in src.replace(" ", "")


@needs_source
@pytest.mark.parametrize("mod", ["scene_branch.py", "scene_hooks.py"])
def test_reconstructed_modules_exist(mod):
    """配置的 custom_imports 是 allow_failed_imports=False。

    缺任何一个模块，Config.fromfile 直接失败 —— 这正是最初拿不到
    scene_branch.py 时的症状。这两个文件是按权重形状重建的。
    """
    path = os.path.join(GZDA_DIR, mod)
    assert os.path.isfile(path), f"缺 {mod}（custom_imports 会失败）：{path}"


@needs_source
def test_scene_branch_defines_expected_class():
    src = open(os.path.join(GZDA_DIR, "scene_branch.py"), encoding="utf-8").read()
    assert "class StripRCNNScene" in src
    assert "class SceneHead" in src
    assert "ROTATED_DETECTORS.register_module" in src.lower() or \
        "@ROTATED_DETECTORS.register_module()" in src
    # 5 层 * 256 = 1280：与权重里的 fc1.weight (512, 1280) 对应
    assert "num_levels" in src and "in_channels" in src


@needs_source
def test_scene_hooks_defines_hook():
    src = open(os.path.join(GZDA_DIR, "scene_hooks.py"), encoding="utf-8").read()
    assert "SceneConsoleHook" in src


def test_scene_preset_registered_both_sides():
    from backend.contracts import presets as bp
    from vecrot_qgis.jobs import presets as pp

    keys_b = {p.key for p in bp.PRESETS}
    keys_p = {p.key for p in pp.PRESETS}
    assert "strip_s_gzda_scene" in keys_b, "backend 侧缺少场景预设"
    assert "strip_s_gzda_scene" in keys_p, "插件侧缺少场景预设"
    assert keys_b == keys_p


def test_scene_preset_is_first():
    """场景模型应排在第一位（大范围推理推荐）。

    只查顺序与参数，不查文件是否下载 —— 后者由下面的 availability 用例覆盖。
    """
    from backend.contracts import presets as bp

    first = bp.PRESETS[0]
    assert first.key == "strip_s_gzda_scene", f"第一个预设是 {first.key}"
    assert first.detector_options.get("scene_threshold") == 0.5


@needs_weights
@needs_source
def test_scene_preset_is_available():
    """下载齐全时，默认预设必须真的可用，否则面板一打开就跑不了。"""
    from backend.contracts import presets as bp

    first = bp.PRESETS[0]
    assert first.is_available(), "默认选中的场景预设必须可用：" + str(first.missing())


def test_scene_preset_uses_scene_weights():
    from backend.contracts import presets as bp

    p = bp.get_preset("strip_s_gzda_scene")
    assert p is not None
    assert "detector_strip_s_scene" in p.files["weights"]
    assert "strip_s_gzda_scene.py" in p.files["config"]


def test_jobspec_accepts_scene_threshold():
    """job.json 必须能传 scene_threshold，且越界被拒。"""
    from backend.contracts.jobspec import JobSpecError, parse_job_spec

    base = {
        "schema_version": "1.0",
        "job_id": "scene_t",
        "input": {"raster_path": "C:/x/scene.tif", "bands_rgb": [1, 2, 3],
                  "aoi_geojson": {"type": "Polygon",
                                  "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                  "aoi_crs": "EPSG:4326"},
        "inference": {"scene_threshold": 0.35},
        "output": {"directory": "C:/x/out", "crs_policy": "source"},
    }
    spec = parse_job_spec(base)
    assert spec.inference.scene_threshold == pytest.approx(0.35)

    bad = dict(base)
    bad["inference"] = {"scene_threshold": 1.5}
    with pytest.raises(JobSpecError):
        parse_job_spec(bad)


def test_scene_threshold_default_is_none():
    """未指定时应为 None，让协调器回退到 manifest / 0.5。"""
    from backend.contracts.jobspec import parse_job_spec

    spec = parse_job_spec({
        "schema_version": "1.0", "job_id": "n",
        "input": {"raster_path": "C:/x/a.tif", "bands_rgb": [1, 2, 3],
                  "aoi_geojson": {"type": "Polygon",
                                  "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}},
        "inference": {},
        "output": {"directory": "C:/x/o"},
    })
    assert spec.inference.scene_threshold is None
