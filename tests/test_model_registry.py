"""模型文件夹注册表的测试（不需要 GPU / torch / QGIS）。

守的是「一个文件夹 = 一个模型」这套新约定的每一环：

* ``model.json`` 的字段校验（非法内容要报出来，而不是猜）
* 路径写法（相对文件夹 / ``repo:`` / ``models:`` / 绝对）
* 扫描与排序、跳过与原因
* 迁移脚本能把老目录补成新约定
* ``to_job_fields()`` 会把 repo / options 带进 job.json
* 两份镜像逐字节一致
* 后端工厂与环境角色能被 rio_detr 正确路由
"""
from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import make_model_folder, plugin_parent   # noqa: E402

sys.path.insert(0, plugin_parent())

from backend.contracts import model_registry as mr   # noqa: E402


# ------------------------------------------------------------------ 构造


def _strip_spec(**over):
    spec = {
        "spec_version": 1,
        "key": "strip_s_fusion",
        "label": "融合数据集 · Strip-R-CNN-S",
        "backend": "strip_s_gzda",
        "num_classes": 1,
        "repo": "repo:third_party/Strip-R-CNN",
        "weights": "best.pth",
        "config": "src:configs/gzda/strip_s_gzda.py",
        "detector_options": {"score_thr": 0.05, "nms_iou": 0.3},
    }
    spec.update(over)
    return spec


def _rio_spec(**over):
    spec = {
        "spec_version": 1,
        "key": "rio_gzda",
        "label": "RiO-DETRv2-X（gzda 融合数据）",
        "backend": "rio_detr",
        "num_classes": 1,
        "repo": "repo:third_party/RiO-DETRv2",
        "weights": "last.pth",
        "config": "riov2_hgnetv2_x_gzda.yml",
        "detector_options": {"score_thr": 0.4},
    }
    spec.update(over)
    return spec


# ------------------------------------------------------------------ 字段校验


def test_minimal_spec_parses(tmp_path):
    folder = make_model_folder(str(tmp_path), "strip_s_fusion", _strip_spec(),
                               {"best.pth": 16})
    preset = mr.load_spec(folder, REPO, str(tmp_path))
    assert preset.key == "strip_s_fusion"
    assert preset.backend == "strip_s_gzda"
    assert preset.num_classes == 1
    assert preset.detector_options["score_thr"] == 0.05
    assert preset.spec_error == ""


def test_key_defaults_to_folder_name(tmp_path):
    spec = _strip_spec()
    spec.pop("key")
    folder = make_model_folder(str(tmp_path), "my_strip", spec, {"best.pth": 16})
    assert mr.load_spec(folder, REPO, str(tmp_path)).key == "my_strip"


@pytest.mark.parametrize("bad,needle", [
    ({"backend": "not_a_backend"}, "未知 backend"),
    ({"backend": ""}, "缺少 backend"),
    ({"num_classes": 0}, "num_classes"),
    ({"num_classes": "1"}, "num_classes"),
    ({"spec_version": 99}, "spec_version"),
    ({"detector_options": {"score_thr": {"nested": 1}}}, "detector_options"),
    ({"label": 42}, "label"),
    ({"weights": 123}, "路径必须是字符串"),
    ({"order": "first"}, "order"),
    ({"enabled": "yes"}, "enabled"),
])
def test_bad_spec_is_rejected(tmp_path, bad, needle):
    spec = _strip_spec(**bad)
    folder = make_model_folder(str(tmp_path), "bad", spec, {"best.pth": 16})
    with pytest.raises(mr.ModelSpecError) as exc:
        mr.load_spec(folder, REPO, str(tmp_path))
    assert needle in str(exc.value)


def test_missing_required_role_reported_not_crashed(tmp_path):
    spec = _strip_spec()
    spec.pop("weights")
    folder = make_model_folder(str(tmp_path), "no_weights", spec)
    preset = mr.load_spec(folder, REPO, str(tmp_path))
    assert not preset.is_available()
    assert any("权重" in m for m in preset.missing())


def test_missing_file_on_disk_is_reported(tmp_path):
    folder = make_model_folder(str(tmp_path), "no_pth", _strip_spec())
    preset = mr.load_spec(folder, REPO, str(tmp_path))
    assert not preset.is_available()
    assert any("权重" in m for m in preset.missing())


# ------------------------------------------------------------------ 路径解析


def test_path_forms(tmp_path):
    folder = os.path.join(str(tmp_path), "f")
    os.makedirs(folder)
    assert mr.resolve_path("best.pth", folder, REPO, str(tmp_path)) == \
        os.path.normpath(os.path.join(folder, "best.pth"))
    assert mr.resolve_path(".", folder, REPO, str(tmp_path)) == os.path.normpath(folder)
    assert mr.resolve_path("repo:configs/gzda/x.py", folder, REPO, str(tmp_path)) == \
        os.path.normpath(os.path.join(REPO, "configs/gzda/x.py"))
    assert mr.resolve_path("models:other/best.pth", folder, REPO, str(tmp_path)) == \
        os.path.normpath(os.path.join(str(tmp_path), "other/best.pth"))
    abs_path = os.path.join(os.sep, "tmp", "abs.pth")
    assert mr.resolve_path(abs_path, folder, REPO, str(tmp_path)) == \
        os.path.normpath(abs_path)


def test_src_prefix_resolves_against_declared_repo(tmp_path):
    """``src:`` 相对模型声明的源码树，而不是仓库根或文件夹。"""
    repo_dir = os.path.join(str(tmp_path), "src_tree")
    os.makedirs(repo_dir)
    folder = os.path.join(str(tmp_path), "f")
    os.makedirs(folder)
    got = mr.resolve_path("src:configs/gzda/x.py", folder, REPO, str(tmp_path),
                          src_root=repo_dir)
    assert got == os.path.normpath(os.path.join(repo_dir, "configs/gzda/x.py"))
    # 没声明 repo 却用 src: → 明确报错
    with pytest.raises(mr.ModelSpecError):
        mr.resolve_path("src:x.py", folder, REPO, str(tmp_path))


def test_empty_path_is_none(tmp_path):
    assert mr.resolve_path("", str(tmp_path), REPO, str(tmp_path)) is None
    assert mr.resolve_path(None, str(tmp_path), REPO, str(tmp_path)) is None


def test_all_resolved_paths_are_absolute(tmp_path):
    folder = make_model_folder(str(tmp_path), "abs_check", _rio_spec(),
                               {"last.pth": 16, "riov2.yml": 16},
                               spec_name="model.json")
    spec = _rio_spec(config="riov2.yml")
    with open(os.path.join(folder, "model.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f)
    preset = mr.load_spec(folder, REPO, str(tmp_path))
    for role, path in preset.files.items():
        assert os.path.isabs(path), f"{role} 不是绝对路径：{path}"


# ------------------------------------------------------------------ 扫描


def test_scan_sorts_and_reports(tmp_path):
    root = str(tmp_path)
    make_model_folder(root, "b_second", _strip_spec(key="second", order=20),
                      {"best.pth": 16})
    make_model_folder(root, "a_first", _strip_spec(key="first", order=10),
                      {"best.pth": 16})
    make_model_folder(root, "c_default", _strip_spec(key="third"),
                      {"best.pth": 16})
    # 没有 model.json 的目录 → skipped，且原因说得清
    os.makedirs(os.path.join(root, "legacy_nojson"))
    presets, skipped = mr.scan(root, REPO)
    assert [p.key for p in presets] == ["first", "second", "third"]
    assert any("model.json" in s["reason"] for s in skipped)
    assert any(s["folder"].endswith("legacy_nojson") for s in skipped)


def test_scan_skips_disabled_and_hidden(tmp_path):
    root = str(tmp_path)
    make_model_folder(root, "off", _strip_spec(key="off", enabled=False),
                      {"best.pth": 16})
    make_model_folder(root, "_private", _strip_spec(key="hidden"), {"best.pth": 16})
    presets, skipped = mr.scan(root, REPO)
    assert presets == []
    assert any("enabled=false" in s["reason"] for s in skipped)


def test_scan_deduplicates_keys(tmp_path):
    root = str(tmp_path)
    make_model_folder(root, "dup_a", _strip_spec(key="same", order=1), {"best.pth": 16})
    make_model_folder(root, "dup_b", _strip_spec(key="same", order=2), {"best.pth": 16})
    presets, skipped = mr.scan(root, REPO)
    assert len(presets) == 1
    assert any("重复" in s["reason"] for s in skipped)


def test_scan_missing_root_is_empty_not_error(tmp_path):
    presets, skipped = mr.scan(os.path.join(str(tmp_path), "nope"), REPO)
    assert presets == [] and skipped == []


def test_builtin_vector_only_present(tmp_path):
    builtin = mr.builtin_presets(str(tmp_path), REPO)
    assert [p.key for p in builtin] == ["vector_only"]
    assert builtin[0].backend == "jsonl"


def test_describe_exposes_role_labels(tmp_path):
    """面板显示"权重/配置/源码"的中文名由 describe() 提供，别让 UI 自己猜 key。"""
    folder = make_model_folder(str(tmp_path), "s", _strip_spec(),
                               {"best.pth": 16})
    preset = mr.load_spec(folder, REPO, str(tmp_path))
    info = preset.describe()
    assert set(info["files"]) == {"weights", "config", "repo"}
    assert info["files"]["weights"]["label"] == "权重"
    assert info["files"]["repo"]["label"] == "源码"
    assert os.path.isabs(info["files"]["weights"]["path"])


# ------------------------------------------------------------------ job.json 展开


def test_to_job_fields_carries_repo_and_options(tmp_path):
    folder = make_model_folder(str(tmp_path), "rio_gzda", _rio_spec(),
                               {"last.pth": 16, "riov2_hgnetv2_x_gzda.yml": 16})
    preset = mr.load_spec(folder, REPO, str(tmp_path))
    fields = preset.to_job_fields()
    assert fields["detector_profile"] == "rio_detr"
    assert os.path.normpath(fields["detector_repo"]).endswith(
        os.path.join("third_party", "RiO-DETRv2"))
    assert fields["detector_weights"].endswith("last.pth")
    assert fields["detector_config"].endswith(".yml")
    assert fields["detector_num_classes"] == 1
    assert fields["detector_options"] == {"score_thr": 0.4}


def test_to_job_fields_hf_dir_model(tmp_path):
    """HF 那种"目录即模型"：weights 写 '.' 应解析成文件夹自身。"""
    spec = {"spec_version": 1, "key": "rtdetr", "label": "rtdetr",
            "backend": "rtdetr_hf", "num_classes": 1, "weights": ".",
            "detector_options": {"score_thr": 0.3}}
    folder = make_model_folder(str(tmp_path), "detector_rt", spec,
                               {"config.json": 32, "model.safetensors": 32})
    preset = mr.load_spec(folder, REPO, str(tmp_path))
    assert preset.is_available()
    assert os.path.normpath(preset.files["weights"]) == os.path.normpath(folder)


# ------------------------------------------------------------------ 迁移


def test_migration_writes_valid_specs(tmp_path):
    """迁移脚本为已知老目录写出的 model.json 必须能被注册表读成可用条目。"""
    from deployment import migrate_model_folders as mig   # noqa: F401
    root = str(tmp_path)
    for name, weight in (("detector_strip_s_scene", "best_mAP_iter_37620.pth"),
                         ("detector_strip_s", "best_mAP_iter_90288.pth")):
        folder = os.path.join(root, name)
        os.makedirs(folder)
        with open(os.path.join(folder, weight), "wb") as f:
            f.write(b"\0" * 2048)
    hf = os.path.join(root, "detector_rt-detr_bbox")
    os.makedirs(hf)
    for n in ("config.json", "model.safetensors"):
        with open(os.path.join(hf, n), "wb") as f:
            f.write(b"\0" * 2048)

    assert mig.main(["--models-root", root]) == 0
    presets, skipped = mr.scan(root, REPO)
    keys = {p.key for p in presets}
    assert keys == {"strip_s_gzda_scene", "strip_s_gzda", "rtdetr_roof"}, keys
    assert skipped == []
    for p in presets:
        assert p.is_available(), (p.key, p.missing())


def test_migration_is_idempotent(tmp_path):
    from deployment import migrate_model_folders as mig
    root = str(tmp_path)
    folder = os.path.join(root, "detector_strip_s_scene")
    os.makedirs(folder)
    with open(os.path.join(folder, "best_mAP_iter_37620.pth"), "wb") as f:
        f.write(b"\0" * 2048)
    assert mig.main(["--models-root", root]) == 0
    path = os.path.join(folder, "model.json")
    first = open(path, encoding="utf-8").read()
    # 用户手改过 → 再跑一次不能被覆盖
    with open(path, "w", encoding="utf-8") as f:
        f.write(first.replace("大范围推理推荐", "我改过的说明"))
    edited = open(path, encoding="utf-8").read()
    assert mig.main(["--models-root", root]) == 0
    assert open(path, encoding="utf-8").read() == edited


def test_add_model_folder_tool(tmp_path, monkeypatch):
    """``add_model_folder.py`` 生成的 model.json 必须能被扫出来（路径写法别写错）。"""
    from deployment import add_model_folder as amf

    target = os.path.join(str(tmp_path), "models")
    os.makedirs(target, exist_ok=True)
    monkeypatch.setattr(amf.mr, "default_models_root", lambda *a, **k: target)
    weights = os.path.join(str(tmp_path), "best.pth")
    with open(weights, "wb") as f:
        f.write(b"\0" * 64)

    rc = amf.main([
        "--name", "strip_fusion", "--backend", "strip_s_gzda",
        "--weights", weights,
        "--config", os.path.join("third_party", "Strip-R-CNN", "configs",
                                 "gzda", "strip_s_gzda.py"),
        "--label", "融合数据集 · Strip-R-CNN-S", "--num-classes", "1",
        "--score-thr", "0.05", "--order", "15",
    ])
    assert rc == 0
    spec = json.load(open(os.path.join(target, "strip_fusion", "model.json"),
                          encoding="utf-8"))
    assert spec["weights"] == "best.pth"          # 相对文件夹，可整包搬机器
    assert spec["config"].startswith("src:")      # 相对源码树
    presets, _ = mr.scan(target, REPO)
    got = {p.key: p for p in presets}["strip_fusion"]
    assert got.order == 15
    assert got.files["weights"] == os.path.join(target, "strip_fusion", "best.pth")


def test_build_models_package_specs_are_valid(tmp_path):
    """网盘包里的 model.json 必须能被注册表读成合法条目（否则用户装完是空的）。"""
    from deployment import build_models_package as bmp

    out = os.path.join(str(tmp_path), "上传")
    assert bmp.main(["--out", out, "--zip"]) == 0
    models_dir = os.path.join(out, "models")
    assert os.path.isfile(os.path.join(out, "models.zip")), "要能直接压出 models.zip"

    presets, skipped = mr.scan(models_dir, REPO)
    assert skipped == [], skipped
    keys = {p.key for p in presets}
    assert {"strip_s_gzda_scene", "strip_s_gzda", "rtdetr_roof",
            "strip_s_fusion", "rio_gzda"} <= keys, keys
    for preset in presets:
        assert preset.spec_error == "", (preset.key, preset.spec_error)
        # 只允许"权重还没拷进来"，声明本身不能缺字段
        assert all("未配置" not in m for m in preset.missing()), preset.missing()


def test_models_package_matches_migration(tmp_path):
    """网盘包里的声明必须与迁移脚本补出来的一致（两份不能各写一套）。"""
    from deployment import build_models_package as bmp
    from deployment.migrate_model_folders import LEGACY

    out = os.path.join(str(tmp_path), "上传")
    assert bmp.main(["--out", out]) == 0
    for folder, spec in LEGACY.items():
        path = os.path.join(out, "models", folder, mr.SPEC_FILENAME)
        assert json.load(open(path, encoding="utf-8")) == spec, folder


def test_models_root_can_be_elsewhere(tmp_path, monkeypatch):
    """模型可以放在仓库外（环境变量 / manifest 声明），不必塞进 clone 目录。"""
    elsewhere = os.path.join(str(tmp_path), "D盘-models")
    os.makedirs(elsewhere)
    make_model_folder(elsewhere, "m1", _strip_spec(key="m1"), {"best.pth": 16})

    monkeypatch.setenv(mr.MODELS_ROOT_ENV, elsewhere)
    assert mr.resolve_models_root(REPO) == os.path.normpath(elsewhere)

    # manifest 显式声明优先于环境变量
    assert mr.resolve_models_root(REPO, elsewhere) == os.path.normpath(elsewhere)
    # 声明了但不存在的路径 → 回落到默认，不静默指向空目录
    monkeypatch.delenv(mr.MODELS_ROOT_ENV)
    assert mr.resolve_models_root(REPO, os.path.join(str(tmp_path), "不存在")) == \
        mr.default_models_root(REPO)


# ------------------------------------------------------------------ 两份镜像


def test_registry_mirror_is_byte_identical():
    """插件侧镜像必须与权威文件逐字节相同（否则两边行为会漂移）。"""
    a = os.path.join(REPO, "backend", "contracts", "model_registry.py")
    b = os.path.join(plugin_parent(), "vecrot_qgis", "jobs", "model_registry.py")
    assert open(a, "rb").read() == open(b, "rb").read(), (
        "两份 model_registry.py 不一致：改完请 "
        "cp backend/contracts/model_registry.py "
        "vecrot_qgis/jobs/model_registry.py")


def test_both_sides_scan_identically(tmp_path):
    """同一个 models 根，两侧扫出来的 key / backend / files 必须一致。"""
    from backend.contracts import presets as bp
    from vecrot_qgis.jobs import presets as pp

    root = str(tmp_path)
    make_model_folder(root, "a", _strip_spec(key="strip_a"), {"best.pth": 16})
    make_model_folder(root, "b", _rio_spec(key="rio_b"),
                      {"last.pth": 16, "riov2_hgnetv2_x_gzda.yml": 16})
    saved_b = (bp.repo_root(), bp.models_root())
    saved_p = (pp.repo_root(), pp.models_root())
    try:
        bp.refresh_models_root(root)
        pp.refresh_models_root(root)
        a = {p.key: (p.backend, p.num_classes, p.files, p.detector_options)
             for p in bp.PRESETS}
        b = {p.key: (p.backend, p.num_classes, p.files, p.detector_options)
             for p in pp.PRESETS}
        assert a == b
        assert set(a) == {"strip_a", "rio_b", "vector_only"}
    finally:
        bp.refresh_models_root(saved_b[1])
        pp.refresh_models_root(saved_p[1])


# ------------------------------------------------------------------ 后端与调度


def test_rio_backend_is_known_profile():
    """rio_detr 必须被工厂认识：缺源码时报 DETECTOR_REPO_MISSING 而不是 PROFILE_UNKNOWN。"""
    from backend.detector.backends import DetectorNotAvailable, build_detector

    backend = build_detector("rio_detr", {"detector_repo": "/nonexistent/rio"})
    with pytest.raises(DetectorNotAvailable) as exc:
        backend.prepare()
    assert exc.value.code == "DETECTOR_REPO_MISSING"


def test_unknown_profile_still_rejected():
    from backend.detector.backends import DetectorNotAvailable, build_detector

    with pytest.raises(DetectorNotAvailable) as exc:
        build_detector("definitely_not_a_backend", {})
    assert exc.value.code == "DETECTOR_PROFILE_UNKNOWN"


def test_rio_runs_in_its_own_environment(tmp_path):
    """rio_detr 必须走 detector_rio 环境（torch 2.2 + timm），不能借用 mmcv 那套。"""
    from backend.coordinator.runner import Coordinator
    from backend.contracts.jobspec import parse_job_spec

    spec = parse_job_spec({
        "schema_version": "1.0",
        "job_id": "env_role_test",
        "input": {"raster_path": os.path.join(REPO, "x.tif"),
                  "bands_rgb": [1, 2, 3],
                  "aoi_geojson": {"type": "Polygon",
                                  "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}},
        "inference": {"detector_profile": "rio_detr"},
        "output": {"directory": REPO},
    })
    manifest = {"environments": {"detector_rio": {"python": "/usr/bin/python3"},
                                 "detector": {"python": "/usr/bin/python3"}}}
    coord = Coordinator(spec, manifest, str(tmp_path))   # work_root 用临时目录，别污染仓库
    assert coord._env_role_for("detector", {"profile": "rio_detr"}) == "detector_rio"
    assert coord._env_role_for("detector", {"profile": "strip_s_gzda"}) == "detector"
    assert coord._env_role_for("detector", {"profile": "rtdetr_hf"}) == "detector"


def test_rio_detector_options_reach_backend(tmp_path):
    """model.json 的阈值必须一路走到后端 options 里（不能被 manifest 默认值盖掉）。"""
    from backend.coordinator.runner import Coordinator
    from backend.contracts.jobspec import parse_job_spec

    spec = parse_job_spec({
        "schema_version": "1.0",
        "job_id": "opts_test",
        "input": {"raster_path": os.path.join(REPO, "x.tif"),
                  "bands_rgb": [1, 2, 3],
                  "aoi_geojson": {"type": "Polygon",
                                  "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}},
        "inference": {
            "detector_profile": "rio_detr",
            "detector_weights": os.path.join(REPO, "w.pth"),
            "detector_config": os.path.join(REPO, "c.yml"),
            "detector_repo": os.path.join(REPO, "third_party", "RiO-DETRv2"),
            "detector_options": {"score_thr": 0.55, "max_per_img": 120},
        },
        "output": {"directory": REPO},
    })
    manifest = {"models": {"detector": {"profile": "strip_s_gzda", "score_thr": 0.05}}}
    got = Coordinator(spec, manifest, str(tmp_path))._detector_options()
    assert got["profile"] == "rio_detr"
    assert got["options"]["score_thr"] == 0.55        # job 压过 manifest
    assert got["options"]["max_per_img"] == 120
    assert got["options"]["detector_repo"].endswith("RiO-DETRv2")
    assert got["options"]["detector_config"].endswith("c.yml")
