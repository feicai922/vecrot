"""把插件安装到 QGIS profile，并让它在 profile 下也能找到部署 manifest。

    python deployment/install_plugin.py            # 安装 + 写注册表
    python deployment/install_plugin.py --uninstall

为什么需要这个脚本
------------------
插件装在 QGIS profile 的 ``python/plugins/vecrot_qgis`` 下，那句「从插件目录向上找
deployment/manifest.json」在 profile 里永远找不到仓库（实测确认）。三条可行通道：

1. 环境变量 ``VECROT_MANIFEST`` —— 对 GUI 启动的 QGIS 不太方便
2. QGIS 设置 ``vecrot/manifest_path`` —— 面板里手动选一次就记住（本脚本直接写）
3. 把 manifest 复制到 ``%USERPROFILE%\\vecrot\\deployment\\manifest.json``

本脚本做 2 + 3：写 QgsSettings 需要 QGIS 运行时，所以这一步通过
``run_in_qgis.py`` 调用 ``_write_qgssetting.py`` 完成；纯文件操作（复制插件、
复制 manifest）在当前 Python 里直接做。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

PLUGIN_SRC = os.path.join(REPO, "qgis_plugin", "vecrot_qgis")
MANIFEST_SRC = os.path.join(HERE, "manifest.json")
PLUGIN_KEY = "vecrot_qgis"

WRITE_SETTING_SCRIPT = '''"""在 QGIS 运行时里写入设置并启用插件（由 install_plugin.py 调用）。"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from qgis.core import QgsApplication, QgsSettings

path = sys.argv[1]
plugin_key = sys.argv[2]

app = QgsApplication([], False)
app.initQgis()
try:
    # 同时写两个作用域：默认作用域与 QGIS/QGIS3。
    # 实测：headless + o4w_env 下默认作用域落到
    # ".../Roaming/python/profiles/default/Unknown Organization/python.ini"，
    # 而正常启动的 QGIS 读的是 ".../QGIS/QGIS3/.../QGIS3.ini"。只写一个会读不到。
    results = []
    for args in ((), ("QGIS", "QGIS3")):
        try:
            s = QgsSettings(*args)
            s.setValue("vecrot/manifest_path", path)
            s.sync()
            v = s.value("vecrot/manifest_path", "", type=str)
            results.append((args, v, v == path))
        except Exception as exc:            # noqa: BLE001
            results.append((args, str(exc), False))
    for args, v, ok in results:
        tag = "default" if not args else "/".join(args)
        print(("OK  " if ok else "FAIL") + f" [{tag}] vecrot/manifest_path = {v}")

    # 启用插件：QGIS 用 mSettings\enabledPlugins 记录已勾选的插件。
    # 不在这个列表里 → 插件管理器里看不到勾选状态，菜单/工具栏也不会出现。
    enabled = s.value("mSettings/enabledPlugins", []) or []
    if isinstance(enabled, str):
        enabled = [enabled]
    if plugin_key not in enabled:
        enabled.append(plugin_key)
        s.setValue("mSettings/enabledPlugins", enabled)
    s.sync()
    after = s.value("mSettings/enabledPlugins", []) or []
    if isinstance(after, str):
        after = [after]
    print(("OK  " if plugin_key in after else "FAIL")
          + f" enabledPlugins = {after}")
finally:
    app.exitQgis()
'''


def profile_plugin_dir() -> str:
    appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(appdata, "QGIS", "QGIS3", "profiles", "default",
                        "python", "plugins")


def home_manifest_dir() -> str:
    return os.path.join(os.path.expanduser("~"), "vecrot", "deployment")


def install() -> int:
    dest = os.path.join(profile_plugin_dir(), PLUGIN_KEY)
    print(f"== 安装插件 ==\n  从 {PLUGIN_SRC}\n  到 {dest}")
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(PLUGIN_SRC, dest,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    n = sum(len(files) for _r, _d, files in os.walk(dest))
    print(f"  已复制 {n} 个文件")

    print("\n== 部署 manifest 副本 ==")
    target_dir = home_manifest_dir()
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, "manifest.json")
    shutil.copy2(MANIFEST_SRC, target)
    # manifest 里的相对路径以「仓库根」为基准。副本放到仓库外后必须改成绝对路径，
    # 否则协调器解析出 <home>/vecrot/vecrot/models/... 这种错误路径。
    with open(target, encoding="utf-8") as f:
        m = json.load(f)
    def abspath(p):
        if not p:
            return p
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(REPO, p))
    models = m.get("models") or {}
    for key in ("detector", "vectorizer"):
        block = models.get(key) or {}
        for fld in ("repo", "weights", "config", "vecrot_dir", "ckpt", "base"):
            if fld in block:
                block[fld] = abspath(block[fld])
        models[key] = block
    if models.get("detections_jsonl"):
        models["detections_jsonl"] = abspath(models["detections_jsonl"])
    m["models"] = models
    m["_installed_copy"] = {
        "source": MANIFEST_SRC,
        "note": "relative paths converted to absolute against the repo root",
    }
    # 关键：manifest 被复制到仓库外后，「上一层」不再是代码位置。
    # 显式写入 repo_root，插件据此设置 QProcess 的 cwd（否则 -m backend.cli 找不到模块）。
    m["repo_root"] = REPO
    with open(target, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"  已写 {target}（相对路径转为绝对路径）")

    print("\n== 写入 QGIS 设置 ==")
    tmp = os.path.join(REPO, "work", "_write_qgssetting.py")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(WRITE_SETTING_SCRIPT)
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "run_in_qgis.py"), tmp, target, PLUGIN_KEY],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (proc.stdout or "").strip().splitlines()
    tail = [l for l in out if "vecrot/manifest_path" in l or l.startswith(("OK", "FAIL"))]
    for line in tail:
        print("  " + line)
    if proc.returncode != 0:
        print("  ⚠ 写入失败；可改为设置环境变量 VECROT_MANIFEST 指向：")
        print(f"     {target}")

    print("\n== 下一步 ==")
    print("  1. 重启 QGIS（或插件管理器里重新勾选）")
    print("  2. 菜单 Vecrot → Vecrot 建筑提取")
    print("  3. 选影像 → 框选 → 输出目录 → 开始提取")
    print(f"\n  manifest: {target}")
    print(f"  插件目录: {dest}")
    return 0


def uninstall() -> int:
    dest = os.path.join(profile_plugin_dir(), PLUGIN_KEY)
    if os.path.isdir(dest):
        shutil.rmtree(dest)
        print(f"已删除 {dest}")
    else:
        print(f"未安装：{dest}")
    for p in (os.path.join(home_manifest_dir(), "manifest.json"),):
        if os.path.isfile(p):
            os.remove(p)
            print(f"已删除 {p}")
    print("QGIS 设置项 vecrot/manifest_path 未清除（无害，可在下次安装时覆盖）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uninstall", action="store_true")
    args = ap.parse_args()
    return uninstall() if args.uninstall else install()


if __name__ == "__main__":
    sys.exit(main())
