"""Installer isolation, reversible uninstall and development-link safety."""
import importlib.util
import json
from pathlib import Path

import pytest

from deployment import plugin_manager as pm


def source_at(tmp_path):
    source = tmp_path / "源码 with spaces" / "vecrot_qgis"
    source.mkdir(parents=True)
    (source / "metadata.txt").write_text("[general]\nname=Vecrot", encoding="utf-8")
    (source / "plugin.py").write_text("VERSION = 1", encoding="utf-8")
    return source


def test_copy_update_uninstall_preserves_source_and_previous_install(tmp_path):
    source = source_at(tmp_path)
    profile = pm.profile_directory("default", str(tmp_path / "profiles"))
    first = pm.install_files(str(source), profile)
    target = Path(first["target"])
    (source / "plugin.py").write_text("VERSION = 2", encoding="utf-8")
    second = pm.install_files(str(source), profile)
    assert (Path(second["backup"]) / "plugin.py").read_text() == "VERSION = 1"
    assert (target / "plugin.py").read_text() == "VERSION = 2"
    models = tmp_path / "models"
    models.mkdir()
    removed = pm.uninstall_files(profile)
    assert removed["removed"] and not target.exists()
    assert (Path(removed["backup"]) / "plugin.py").is_file()
    assert source.is_dir() and models.is_dir()
    assert not pm.uninstall_files(profile)["removed"]


@pytest.mark.parametrize("name", ["..", "../other", "x\\y", "C:foo"])
def test_reject_profile_traversal(tmp_path, name):
    with pytest.raises(ValueError):
        pm.profile_directory(name, str(tmp_path))


def test_failed_update_restores_old_plugin(tmp_path, monkeypatch):
    source = source_at(tmp_path)
    profile = tmp_path / "profile"
    target = Path(pm.install_files(str(source), profile)["target"])
    def failed_copy(*args, **kwargs):
        raise OSError("injected copy failure")
    monkeypatch.setattr(pm.shutil, "copytree", failed_copy)
    with pytest.raises(OSError, match="injected"):
        pm.install_files(str(source), profile)
    assert (target / "metadata.txt").is_file()


def test_dev_link_uninstall_does_not_move_source(tmp_path):
    source = source_at(tmp_path)
    profile = tmp_path / "profile"
    try:
        result = pm.install_files(str(source), profile, dev=True)
    except RuntimeError:
        pytest.skip("directory symlink permission unavailable")
    target = Path(result["target"])
    assert target.is_symlink() and target.resolve() == source
    (source / "plugin.py").write_text("VERSION = 3", encoding="utf-8")
    assert (target / "plugin.py").read_text() == "VERSION = 3"
    result = pm.uninstall_files(profile)
    assert source.is_dir() and (source / "plugin.py").is_file()
    assert Path(result["backup"]).is_symlink()


def load_installer():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("vecrot_installer_test", root / "install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_keeps_explicit_existing_interpreter(tmp_path, monkeypatch):
    module = load_installer()
    template = tmp_path / "template.json"
    output = tmp_path / "manifest.json"
    python = tmp_path / "custom-env" / "python.exe"
    python.parent.mkdir()
    python.touch()
    template.write_text(json.dumps({"models": {}, "environments": {"vector": {"python": ""}}}))
    output.write_text(json.dumps({"environments": {"vector": {"python": str(python)}}}))
    monkeypatch.setattr(module, "TPL", str(template))
    monkeypatch.setattr(module, "OUT_MANIFEST", str(output))
    monkeypatch.delenv("VECROT_PY_VECTOR", raising=False)
    monkeypatch.setattr(module, "detect_environments", lambda: {})
    before = output.read_bytes()
    assert module.build_manifest(check_only=True)["environments"]["vector"]["python"] == str(python)
    assert output.read_bytes() == before


def test_uninstall_cli_does_not_need_models(tmp_path, monkeypatch):
    module = load_installer()
    monkeypatch.setattr(module, "build_manifest", lambda **kw: pytest.fail("uninstall loaded models"))
    args = ["--profiles-path", str(tmp_path), "--qgis-profile", "test", "--uninstall", "--no-qgis-settings"]
    assert module.main(args) == 0


def test_precheck_failure_does_not_install_or_write_manifest(tmp_path, monkeypatch):
    module = load_installer()
    calls = []
    def manifest(check_only=False):
        assert check_only
        calls.append("check")
        return {"models": {}}
    monkeypatch.setattr(module, "build_manifest", manifest)
    monkeypatch.setattr(module, "selfcheck", lambda _: 1)
    monkeypatch.setattr(module, "install_files", lambda *a, **kw: pytest.fail("installed before precheck"))
    assert module.main(["--profiles-path", str(tmp_path), "--no-qgis-settings"]) == 1
    assert calls == ["check"]


def test_repo_named_vecrot_installs_from_another_working_directory(tmp_path):
    """A fresh checkout uses its own location, never the author's checkout or cwd."""
    import shutil
    import subprocess
    import sys

    original = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "任意下载位置 with spaces" / "vecrot"
    (checkout / "deployment").mkdir(parents=True)
    for name in ("install.py", "uninstall.py", "deployment/plugin_manager.py",
                 "deployment/manifest.template.json"):
        shutil.copy2(original / name, checkout / name)
    source = checkout / "vecrot_qgis"
    source.mkdir()
    (source / "metadata.txt").write_text("[general]\nname=Vecrot", encoding="utf-8")
    profiles = tmp_path / "different user" / "profiles"
    outside = tmp_path / "unrelated cwd"
    outside.mkdir()
    common = ["--profiles-path", str(profiles), "--no-qgis-settings"]
    proc = subprocess.run([sys.executable, "-X", "utf8", str(checkout / "install.py"),
                           "--ui-only", *common], cwd=outside, capture_output=True,
                          text=True, encoding="utf-8", timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    manifest = json.loads((checkout / "deployment/manifest.json").read_text(encoding="utf-8"))
    assert Path(manifest["repo_root"]) == checkout
    for block in manifest["models"].values():
        if not isinstance(block, dict):
            continue
        for key in ("repo", "weights", "config", "vecrot_dir", "ckpt", "base"):
            if block.get(key):
                assert Path(block[key]).is_relative_to(checkout), block[key]
    target = profiles / "default/python/plugins/vecrot_qgis"
    assert (target / "metadata.txt").is_file()
    assert not (outside / "deployment").exists()
    proc = subprocess.run([sys.executable, "-X", "utf8", str(checkout / "uninstall.py"),
                           *common], cwd=outside, capture_output=True,
                          text=True, encoding="utf-8", timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not target.exists()
    assert (source / "metadata.txt").is_file()
