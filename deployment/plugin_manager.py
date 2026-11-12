"""Manage only the Vecrot plugin in a selected QGIS profile (standard library only)."""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

PLUGIN_NAME = "vecrot_qgis"


def profile_directory(profile: str, profiles_path: str | None = None) -> Path:
    if not profile or profile in (".", "..") or any(c in profile for c in '/\\:'):
        raise ValueError("QGIS profile 必须是单个目录名，不能包含路径分隔符")
    if profiles_path:
        base = Path(profiles_path).expanduser().resolve()
    else:
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise ValueError("未找到 APPDATA，请用 --profiles-path 指定 QGIS profiles 根目录")
        base = Path(appdata).resolve() / "QGIS" / "QGIS3" / "profiles"
    result = base / profile
    # Never follow a linked profile into another location during file operations.
    if result.resolve().parent != base.resolve():
        raise ValueError("profile 路径解析后超出指定 profiles 根目录")
    return result


def plugin_target(profile_dir: Path) -> Path:
    profile_dir = profile_dir.resolve()
    parent = profile_dir / "python" / "plugins"
    if not parent.resolve().is_relative_to(profile_dir):
        raise ValueError("插件父目录解析后超出目标 profile")
    return parent / PLUGIN_NAME


def _exists(path: Path) -> bool:
    return os.path.lexists(path)


def _backup(profile_dir: Path, target: Path) -> Path | None:
    if not _exists(target):
        return None
    # Rename the directory entry itself: a dev symlink is not traversed.
    backup_root = profile_dir / "vecrot-plugin-backups"
    if not backup_root.resolve().is_relative_to(profile_dir.resolve()):
        raise ValueError("备份目录解析后超出目标 profile")
    backup_root.mkdir(parents=True, exist_ok=True)
    backup = backup_root / (datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8])
    target.rename(backup)
    return backup


def install_files(source: str, profile_dir: Path, dev: bool = False) -> dict:
    source_path = Path(source).resolve()
    if not (source_path / "metadata.txt").is_file():
        raise ValueError(f"无效插件源码目录：{source_path}")
    target = plugin_target(profile_dir)
    if source_path == target or (target.exists() and source_path == target.resolve() and not target.is_symlink()):
        raise ValueError("插件源目录与安装目录相同")
    if source_path.is_relative_to(profile_dir.resolve()):
        raise ValueError("源码必须位于目标 profile 外，避免卸载影响源码")
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = _backup(profile_dir, target)
    try:
        if dev:
            try:
                target.symlink_to(source_path, target_is_directory=True)
            except OSError as exc:
                raise RuntimeError("无法创建调试符号链接；请启用 Windows 开发者模式或使用管理员终端，或改用普通复制安装") from exc
        else:
            shutil.copytree(source_path, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    except Exception:
        # Preserve partial files, restore previous working installation when possible.
        if _exists(target):
            _backup(profile_dir, target)
        if backup is not None:
            backup.rename(target)
        raise
    return {"target": str(target), "backup": str(backup) if backup else None,
            "mode": "dev-link" if dev else "copy"}


def uninstall_files(profile_dir: Path) -> dict:
    target = plugin_target(profile_dir)
    backup = _backup(profile_dir, target)
    return {"target": str(target), "backup": str(backup) if backup else None,
            "removed": backup is not None}
