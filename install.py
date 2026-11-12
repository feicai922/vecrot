#!/usr/bin/env python
"""从 GitHub 仓库安装 Vecrot QGIS 插件。

    python install.py                      # 常规安装
    python install.py --check              # 只检查，不改任何文件
    python install.py --qgis-profile default

前置条件（请先做完，否则本脚本只能报错）：

1. 模型已按 docs/模型下载.md 放到仓库根目录下
2. 三个 conda 环境已按 docs/安装教程-自建环境.md 建好

常用操作：
    python install.py --check
    python install.py --uninstall
    python install.py --dev --ui-only --qgis-profile vecrot-dev

普通安装先自检，再生成本机 manifest、安装薄插件并写入目标 profile 设置。
旧插件以备份方式保留；卸载不要求模型或环境，不删除源码与模型。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from deployment.plugin_manager import profile_directory, install_files, uninstall_files

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
PLUGIN_SRC = os.path.join(REPO, "vecrot_qgis")
TPL = os.path.join(REPO, "deployment", "manifest.template.json")
OUT_MANIFEST = os.path.join(REPO, "deployment", "manifest.json")

#: 环境名 → 候选 conda 环境目录名
CONDA_NAMES = {
    "coordinator": ("vecrot-geo", "vecrot_coord"),
    "detector": ("vecrot-det", "vecrot_det"),
    "vector": ("vecrot-vllm", "vecrot_vllm", "vecrot-vector"),
    "detector_hf": ("vecrot-vllm", "vecrot_vllm"),
}

#: 环境角色 → 环境变量覆盖名
ENV_VARS = {
    "coordinator": "VECROT_PY_COORDINATOR",
    "detector": "VECROT_PY_DETECTOR",
    "vector": "VECROT_PY_VECTOR",
    "detector_hf": "VECROT_PY_VECTOR",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _conda_roots() -> list:
    """可能装了 conda 的根目录，按可靠性排序。"""
    roots = []
    conda_exe = os.environ.get("CONDA_EXE", "")
    if conda_exe:
        roots.append(os.path.dirname(os.path.dirname(conda_exe)))
    roots += [
        os.path.expanduser("~/miniconda3"),
        os.path.expanduser("~/anaconda3"),
        os.path.expanduser("~/miniconda"),
        os.path.expanduser("~/anaconda"),
        r"C:\ProgramData\miniconda3",
        r"C:\ProgramData\anaconda3",
        r"D:\Program Files\miniconda",
        r"D:\Program Files\anaconda3",
        r"C:\Program Files\miniconda3",
        r"D:\miniconda3",
        r"C:\miniconda3",
        r"D:\anaconda3",
        r"C:\anaconda3",
    ]
    # 去重且保留顺序
    seen = set()
    out = []
    for r in roots:
        if not r:
            continue
        k = os.path.normcase(os.path.abspath(r))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _candidate_pythons(role: str) -> list:
    out = []
    envvar = ENV_VARS.get(role)
    if envvar and os.environ.get(envvar):
        out.append(os.environ[envvar])
    for root in _conda_roots():
        for name in CONDA_NAMES.get(role, ()):
            out.append(os.path.join(root, "envs", name, "python.exe"))
    return out


def detect_environments() -> dict:
    mapping = {}
    for role in ("coordinator", "detector", "vector", "detector_hf"):
        found = ""
        for p in _candidate_pythons(role):
            if p and os.path.isfile(p):
                found = p
                break
        mapping[role] = found
    # detector_hf 复用 vector 环境（transformers + torch 同版本）
    if not mapping["detector_hf"] and mapping["vector"]:
        mapping["detector_hf"] = mapping["vector"]
    return mapping


def build_manifest(check_only: bool = False) -> dict:
    """按模板生成 manifest，填入本机的具体路径。"""
    if not os.path.isfile(TPL):
        log(f"       ✗ 找不到 manifest 模板：{TPL}")
        return {}
    with open(TPL, encoding="utf-8") as f:
        m = json.load(f)

    def resolve(p):
        if not p:
            return p
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(REPO, p))

    models = m.get("models") or {}
    for key in ("detector", "vectorizer"):
        block = models.get(key) or {}
        for fld in ("repo", "weights", "config", "vecrot_dir", "ckpt", "base",
                    "detections_jsonl"):
            if block.get(fld):
                block[fld] = resolve(block[fld])
        models[key] = block
    if models.get("detections_jsonl"):
        models["detections_jsonl"] = resolve(models["detections_jsonl"])
    m["models"] = models

    envs = detect_environments()
    previous = {}
    if os.path.isfile(OUT_MANIFEST):
        with open(OUT_MANIFEST, encoding="utf-8") as f:
            previous = json.load(f).get("environments") or {}
    for role, cfg in (m.get("environments") or {}).items():
        override = os.environ.get(ENV_VARS.get(role, ""), "")
        old = (previous.get(role) or {}).get("python", "")
        candidates = (override, old, cfg.get("python", ""), envs.get(role, ""))
        cfg["python"] = override or next((v for v in candidates if v and os.path.isfile(v)), "")
        cfg["_note"] = "本机配置" if cfg["python"] else "未找到，需手工指定 python.exe 绝对路径"

    # manifest 就在 <仓库根>/deployment/ 下，仓库根是它的上一级
    m["repo_root"] = REPO

    if not check_only:
        os.makedirs(os.path.dirname(OUT_MANIFEST), exist_ok=True)
        with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
        log(f"       → {OUT_MANIFEST}")
    return m


def update_qgis_settings(profile_dir: Path, enabled: bool) -> None:
    """Edit only the selected profile; never initialize a GIS/model runtime."""
    runner = os.path.join(REPO, "deployment", "run_in_qgis.py")
    ini = profile_dir / "QGIS" / "QGIS3.ini"
    ini.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the original settings before touching them.
    if ini.is_file():
        import uuid
        shutil.copy2(ini, str(ini) + ".vecrot-backup-" + uuid.uuid4().hex[:8])
    with tempfile.TemporaryDirectory(prefix="vecrot-settings-") as tmp:
        snippet = Path(tmp) / "settings.py"
        snippet.write_text(SETTING_SNIPPET, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", runner, str(snippet), str(ini),
             OUT_MANIFEST, "1" if enabled else "0"], cwd=REPO)
    if proc.returncode:
        raise RuntimeError("QGIS 设置更新失败；插件文件操作已完成。请在 QGIS 插件管理器中手动启用/停用，并在面板选择 deployment/manifest.json")


SETTING_SNIPPET = '''import sys
from qgis.PyQt.QtCore import QSettings
ini, manifest, enabled = sys.argv[1:]
s = QSettings(ini, QSettings.IniFormat)
s.setValue("PythonPlugins/vecrot_qgis", enabled == "1")
if enabled == "1":
    s.setValue("vecrot/manifest_path", manifest)
else:
    s.remove("vecrot/manifest_path")
s.sync()
if s.status() != QSettings.NoError:
    raise RuntimeError("无法写入目标 QGIS profile 设置")
print("[OK] 已更新目标 profile 设置", flush=True)
'''


def selfcheck(m: dict) -> int:
    problems = 0

    log("[3/4] 自检：Python 环境")
    for role in ("coordinator", "detector", "vector", "detector_hf"):
        p = ((m.get("environments") or {}).get(role) or {}).get("python", "")
        if p and os.path.isfile(p):
            log(f"       ✓ {role:12} {p}")
            continue
        if role == "detector_hf":
            vec = ((m.get("environments") or {}).get("vector") or {}).get("python", "")
            if vec and os.path.isfile(vec):
                log(f"       ✗ {role:12} 应指向 vector 环境，但 manifest 里为空")
                problems += 1
            else:
                log(f"       - {role:12} 未配置（只影响 RT-DETR）")
            continue
        log(f"       ✗ {role:12} 未找到")
        problems += 1

    if problems:
        log("")
        log("       建环境：docs/安装教程-自建环境.md")
        log("       已建好但没检测到？用环境变量指定，然后重跑本脚本：")
        log("         setx VECROT_PY_COORDINATOR <vecrot-geo 环境的 python.exe 绝对路径>")
        log("         setx VECROT_PY_DETECTOR    <vecrot-det 环境的 python.exe 绝对路径>")
        log("         setx VECROT_PY_VECTOR      <vecrot-vllm 环境的 python.exe 绝对路径>")
        log("       （setx 后要新开一个命令行窗口才生效）")

    log("\n[4/4] 自检：模型文件")
    rc = subprocess.run([sys.executable, "-X", "utf8", os.path.join(REPO, "deployment", "verify_models.py")],
                        cwd=REPO)
    if rc.returncode != 0:
        problems += 1

    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="安装、检查或卸载 Vecrot QGIS 薄插件；操作前请关闭 QGIS")
    ap.add_argument("--qgis-profile", default="default")
    ap.add_argument("--profiles-path", help="QGIS profiles 根目录，支持便携版/独立调试 profile")
    actions = ap.add_mutually_exclusive_group()
    actions.add_argument("--check", "--check-only", action="store_true", help="只检查，不修改文件")
    actions.add_argument("--uninstall", action="store_true", help="卸载插件（备份保留，不删除模型/环境）")
    ap.add_argument("--dev", action="store_true", help="目录符号链接安装；需 Windows 开发者模式或管理员权限")
    ap.add_argument("--ui-only", action="store_true", help="允许尚未下载模型/建立环境时只安装界面用于调试")
    ap.add_argument("--no-qgis-settings", action="store_true", help="跳过自动启用/停用；请在 QGIS 插件管理器手动操作")
    ap.add_argument("--skip-envs", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.dev and (args.check or args.uninstall):
        ap.error("--dev 仅用于安装")
    try:
        profile = profile_directory(args.qgis_profile, args.profiles_path)
        log(f"仓库根：{REPO}\n目标 profile：{profile}")
        log("请确保该 profile 的 QGIS 已关闭；操作完成后重新打开。")
        if args.uninstall:
            result = uninstall_files(profile)
            log("[OK] 插件已移出加载目录" if result["removed"] else "[OK] 插件未安装，无需重复卸载")
            if result["backup"]:
                log(f"插件备份：{result['backup']}")
            if not args.no_qgis_settings:
                update_qgis_settings(profile, False)
            log("模型、环境、源码和作业成果均保留。")
            return 0
        manifest = build_manifest(check_only=True)
        if not manifest:
            return 1
        if args.check or not args.ui_only:
            problems = selfcheck(manifest)
            if problems:
                log(f"{problems} 项未就绪。修好后重跑；只调试界面可使用 --ui-only。")
                return 1
        if args.check:
            log("[OK] 检查通过（模型未执行推理）")
            return 0
        build_manifest(check_only=False)
        result = install_files(PLUGIN_SRC, profile, dev=args.dev)
        log(f"[OK] 安装完成（{result['mode']}）：{result['target']}")
        if result["backup"]:
            log(f"旧插件备份：{result['backup']}")
        if not args.no_qgis_settings:
            update_qgis_settings(profile, True)
        else:
            log(f"请手动启用 Vecrot 并选择部署 manifest：{OUT_MANIFEST}")
        if args.ui_only:
            log("仅安装界面；环境和模型未验证，暂不保证提取功能可用。")
        log("重启 QGIS 后生效；普通安装更新源码后需重装，--dev 修改后可重启或重新加载插件。")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        log(f"[FAIL] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
