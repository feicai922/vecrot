"""用 QGIS 自带运行时执行任意脚本（不依赖 bat 引号问题）。

用法：
    python deployment/run_in_qgis.py deployment/qgis_runtime_check.py

它会：
* 找 QGIS 安装目录（可用环境变量 QGIS_PREFIX 覆盖）
* 用 o4w_env 的语义在 Python 里重建环境变量
* 用 QGIS 的 python.exe 启动目标脚本
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys


def find_qgis_root() -> str:
    env = os.environ.get("QGIS_PREFIX") or os.environ.get("OSGEO4W_ROOT")
    if env and os.path.isdir(env):
        return env
    patterns = [
        r"D:\Program Files\QGIS *",
        r"C:\Program Files\QGIS *",
        r"C:\OSGeo4W",
        r"D:\OSGeo4W",
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat), reverse=True)
        for h in hits:
            if os.path.isfile(os.path.join(h, "bin", "o4w_env.bat")):
                return h
    raise SystemExit("找不到 QGIS 安装目录；请设置 QGIS_PREFIX")


def build_env(root: str) -> dict:
    """复刻 o4w_env.bat 的关键效果：干净的 PATH + 各组件 ini 里的路径。"""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PYTHON", "QT_", "QGIS_"))}
    path_parts = [os.path.join(root, "bin")]
    for ini in sorted(glob.glob(os.path.join(root, "etc", "ini", "*.bat"))):
        text = open(ini, encoding="utf-8", errors="replace").read()
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("set ") and "=" in line:
                key, val = line[4:].split("=", 1)
                key, val = key.strip(), val.strip()
                if not key or "%" in val:
                    continue
                if key.lower() == "path":
                    path_parts.extend(p.strip() for p in val.split(";") if p.strip())
                else:
                    env[key] = val
    win = env.get("SystemRoot", r"C:\Windows")
    path_parts += [os.path.join(win, "system32"), win,
                   os.path.join(win, "system32", "Wbem")]
    env["PATH"] = os.pathsep.join(dict.fromkeys(path_parts))
    env["PYTHONPATH"] = os.path.join(root, "apps", "qgis-ltr", "python")
    env["PYTHONHOME"] = ""
    env["PYTHONUTF8"] = "1"
    env["QT_QPA_PLATFORM"] = env.get("QT_QPA_PLATFORM", "offscreen")
    return env


def find_python(root: str) -> str:
    cands = sorted(glob.glob(os.path.join(root, "apps", "Python3*", "python.exe")))
    if not cands:
        raise SystemExit(f"在 {root}/apps 下找不到 Python3*/python.exe")
    return cands[-1]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = find_qgis_root()
    if sys.argv[1] == "--probe-only":
        print(f"QGIS={root}")
        print(f"python={find_python(root)}")
        return 0
    target = os.path.abspath(sys.argv[1])
    script_args = sys.argv[2:]
    python = find_python(root)
    env = build_env(root)
    print(f"[run_in_qgis] QGIS={root}")
    print(f"[run_in_qgis] python={python}")
    print(f"[run_in_qgis] script={target}\n", flush=True)
    # -u：无缓冲。否则 QGIS 崩溃时子进程输出会丢，很难诊断
    proc = subprocess.run([python, "-u", target, *script_args], env=env, cwd=os.getcwd())
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
