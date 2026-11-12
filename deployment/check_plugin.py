#!/usr/bin/env python
"""插件静态检查（在**没有 QGIS** 的机器上也能跑）。

本机未安装 QGIS，因此 QGIS 运行时验证属于 ``not_run``。这个脚本能覆盖的部分：
* 所有 .py 语法正确
* metadata.txt 必填字段齐全、qgisMinimumVersion 合法
* classFactory 可解析且返回类名正确
* .ui 文件可被 XML 解析，且 dock_widget.py 里引用的每个控件名都在 .ui 中出现
* 插件里没有 import torch / rasterio / transformers 等禁止依赖（§15 第 2 条）

用法：``python deployment/check_plugin.py``
"""
from __future__ import annotations

import ast
import os
import re
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PLUGIN_DIR = os.path.join(REPO, "qgis_plugin", "vecrot_qgis")

#: §15 第 2 条：绝不向 QGIS Python 安装/导入的推理与 GIS 依赖
FORBIDDEN_IMPORTS = {
    "torch", "torchvision", "transformers", "mmcv", "mmdet", "mmrotate",
    "rasterio", "pyogrio", "geopandas", "shapely", "pyproj", "cv2", "timm",
    "safetensors", "numpy",
}
METADATA_REQUIRED = ["name", "description", "about", "version", "qgisMinimumVersion",
                     "author", "email", "category"]


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def check_metadata() -> int:
    path = os.path.join(PLUGIN_DIR, "metadata.txt")
    if not os.path.isfile(path):
        fail("缺少 metadata.txt")
        return 1
    text = open(path, encoding="utf-8").read()
    fields = dict(re.findall(r"^(\w+)=(.*)$", text, re.MULTILINE))
    problems = 0
    for key in METADATA_REQUIRED:
        if key not in fields:
            fail(f"metadata.txt 缺字段 {key}")
            problems += 1
    mv = fields.get("qgisMinimumVersion", "")
    if not re.fullmatch(r"\d+\.\d+", mv):
        fail(f"qgisMinimumVersion 格式不合法：{mv!r}")
        problems += 1
    if problems == 0:
        ok(f"metadata.txt 完整（min QGIS {mv}）")
    return problems


def check_syntax() -> int:
    problems = 0
    files = []
    for root, _dirs, names in os.walk(PLUGIN_DIR):
        for n in names:
            if n.endswith(".py"):
                files.append(os.path.join(root, n))
    for path in files:
        try:
            ast.parse(open(path, encoding="utf-8").read())
        except SyntaxError as exc:
            fail(f"语法错误 {path}:{exc.lineno}: {exc.msg}")
            problems += 1
    if problems == 0:
        ok(f"{len(files)} 个 .py 语法正确")
    return problems


def check_forbidden_imports() -> int:
    problems = 0
    for root, _dirs, names in os.walk(PLUGIN_DIR):
        for n in names:
            if not n.endswith(".py"):
                continue
            path = os.path.join(root, n)
            tree = ast.parse(open(path, encoding="utf-8").read())
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    mods = [node.module.split(".")[0]]
                for m in mods:
                    if m in FORBIDDEN_IMPORTS:
                        fail(f"{os.path.relpath(path, REPO)} 导入了禁止依赖 {m}")
                        problems += 1
    if problems == 0:
        ok("QGIS 侧无推理/GIS 重依赖导入")
    return problems


def check_entrypoint() -> int:
    path = os.path.join(PLUGIN_DIR, "__init__.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if "classFactory" not in funcs:
        fail("__init__.py 缺少 classFactory")
        return 1
    src = open(path, encoding="utf-8").read()
    if "VecrotPlugin" not in src:
        fail("classFactory 未返回 VecrotPlugin")
        return 1
    ok("classFactory 入口正确")
    return 0


def check_ui_widgets() -> int:
    ui = os.path.join(PLUGIN_DIR, "ui", "dock_widget_base.ui")
    py = os.path.join(PLUGIN_DIR, "ui", "dock_widget.py")
    if not os.path.isfile(ui):
        fail("缺少 dock_widget_base.ui")
        return 1
    try:
        tree = ET.parse(ui)
    except ET.ParseError as exc:
        fail(f".ui 不是合法 XML：{exc}")
        return 1
    widget_names = {w.get("name") for w in tree.iter("widget") if w.get("name")}
    src = open(py, encoding="utf-8").read()
    # 找出 self.<Name> 形式的控件引用
    used = set(re.findall(r"\bself\.([A-Za-z_]\w*)\b", src))
    # 排除 Python 侧自己定义的属性、方法与信号
    own = set(re.findall(r"\bself\.([A-Za-z_]\w*)\s*=", src))
    own |= set(re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", src))
    own |= set(re.findall(r"^\s*([A-Za-z_]\w*)\s*=\s*pyqtSignal", src, re.MULTILINE))
    # 类级常量（全大写）也属于自有成员，不是 .ui 控件
    own |= set(re.findall(r"^\s{4}([A-Z][A-Z0-9_]+)\s*=", src, re.MULTILINE))
    problems = 0
    for name in sorted(used - own):
        if name in {"setupUi", "iface", "bridge", "aoi_geometry", "aoi_crs",
                    "last_out_dir", "manifest_path", "_aoi_tool"}:
            continue
        if name.startswith("_"):
            continue
        if name not in widget_names:
            fail(f"dock_widget.py 引用了 .ui 中不存在的控件 self.{name}")
            problems += 1
    if problems == 0:
        ok(f".ui 控件名与代码引用一致（{len(widget_names)} 个控件）")
    return problems


def check_metadata_parity() -> int:
    """插件 metadata 版本与后端包版本保持一致。"""
    md = os.path.join(PLUGIN_DIR, "metadata.txt")
    text = open(md, encoding="utf-8").read()
    m = re.search(r"^version=(.*)$", text, re.MULTILINE)
    plugin_ver = (m.group(1) if m else "").strip()
    init = open(os.path.join(REPO, "backend", "__init__.py"), encoding="utf-8").read()
    backend_ver = re.search(r'__version__\s*=\s*"([^"]+)"', init).group(1)
    if plugin_ver != backend_ver:
        fail(f"版本不一致：插件 {plugin_ver} ≠ 后端 {backend_ver}")
        return 1
    ok(f"插件与后端版本一致（{plugin_ver}）")
    return 0


def main() -> int:
    print("=== 插件静态检查（QGIS 运行时验证为 not_run）===")
    problems = 0
    for fn in (check_metadata, check_syntax, check_forbidden_imports,
               check_entrypoint, check_ui_widgets, check_metadata_parity):
        problems += fn()
    print(f"\n{'全部通过' if problems == 0 else f'{problems} 项未通过'}")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
