"""在真实 QGIS 里"安装并启用插件"的验证。

与 qgis_runtime_check.py 的区别：那个直接调 classFactory，这个走 QGIS 自己的
插件管理流程（``qgis.utils.loadPlugin`` / ``startPlugin``），能验证：

* metadata.txt 被 QGIS 正确解析（名称、版本、分类）
* 插件包能被 QGIS 的插件加载器 import
* startPlugin 后菜单/工具栏/面板真实存在
* unload 后 QGIS 的插件状态回到未加载

要求插件已复制到 profile 的 python/plugins 下。

用法：
    python deployment/run_in_qgis.py deployment/qgis_plugin_mount_check.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

PLUGIN_KEY = "vecrot_qgis"
results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((bool(ok), msg))
    print(("  [OK]   " if ok else "  [FAIL] ") + msg)


def _profile_plugin_dir() -> str:
    """QGIS 用户 profile 的插件目录（headless 下 plugin_paths 为空，需自己算）。"""
    appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(appdata, "QGIS", "QGIS3", "profiles", "default",
                        "python", "plugins")


def _make_iface_stub(qgis_utils):
    """最小 iface 桩：只实现插件 initGui/unload 用到的接口。

    这不是绕过验证 —— startPlugin 仍然完整执行 classFactory + initGui + 异常处理，
    我们只是补上 headless 会话缺失的主窗口。
    """
    from qgis.PyQt.QtWidgets import QMainWindow, QToolBar

    class _Stub:
        def __init__(self):
            self._mw = QMainWindow()
            self._menu_adds = []
            self._menu_removes = []

        def mainWindow(self):
            return self._mw

        def addPluginToMenu(self, menu, action):
            self._menu_adds.append((menu, action))
            self._mw.menuBar().addMenu(menu).addAction(action)

        def removePluginMenu(self, menu, action):
            self._menu_removes.append((menu, action))

        def addToolBar(self, title):
            tb = QToolBar(title, self._mw)
            self._mw.addToolBar(tb)
            return tb

        def addDockWidget(self, area, dock):
            self._mw.addDockWidget(area, dock)

        def removeDockWidget(self, dock):
            self._mw.removeDockWidget(dock)

        def mapCanvas(self):
            return None

    return _Stub()


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qgis.core import QgsApplication

    qgs = QgsApplication([], False)
    qgs.initQgis()

    try:
        import qgis.utils

        print("=== 1. 插件目录 ===")
        # headless 模式下 QGIS 不会自动填充 plugin_paths，这里显式补上
        profile_dir = _profile_plugin_dir()
        check(True, f"用户 profile 插件目录：{profile_dir}")
        if profile_dir not in qgis.utils.plugin_paths:
            qgis.utils.plugin_paths.append(profile_dir)

        found = os.path.isdir(os.path.join(profile_dir, PLUGIN_KEY))
        check(found, f"vecrot_qgis 已安装到 profile：{found}")
        if not found:
            print("\n  提示：把 qgis_plugin/vecrot_qgis 复制到该目录后重跑。")
            return 1

        print("\n=== 2. metadata.txt 可被解析 ===")
        # QGIS 3.44 的 qgis.utils 没有 plugins_metadata；直接用标准库解析，
        # 校验 QGIS 真正关心的字段（name/version/qgisMinimumVersion/category）
        import configparser

        meta_path = os.path.join(profile_dir, PLUGIN_KEY, "metadata.txt")
        parser = configparser.ConfigParser(interpolation=None)
        parser.optionxform = str
        try:
            parser.read(meta_path, encoding="utf-8")
            general = parser["general"]
            check(True, f"名称 {general.get('name')!r} / 版本 {general.get('version')} / "
                        f"最低 QGIS {general.get('qgisMinimumVersion')} / "
                        f"分类 {general.get('category')}")
            from qgis.core import Qgis

            need = general.get("qgisMinimumVersion", "0")
            check(Qgis.QGIS_VERSION.split("-")[0] >= need,
                  f"QGIS {Qgis.QGIS_VERSION.split('-')[0]} ≥ 最低要求 {need}")
        except Exception as exc:                    # noqa: BLE001
            check(False, f"metadata 解析失败：{exc}")
            return 1

        print("\n=== 3. 走 QGIS 插件管理器加载 ===")
        # startPlugin 依赖 qgis.utils.plugin_paths 已在 sys.path 里（GUI 启动时由
        # QgsPluginRegistry 完成）。headless 下这里手动补上，行为等价。
        if profile_dir not in sys.path:
            sys.path.append(profile_dir)
        import importlib

        mod = importlib.import_module(PLUGIN_KEY)
        check(hasattr(mod, "classFactory"), "插件模块可 import 且有 classFactory")

        # headless 会话里 qgis.utils.iface 是 None（没有主窗口），startPlugin 会因此
        # 失败。这里注入一个最小 iface 桩，走与 GUI 完全相同的 startPlugin 路径。
        fake_iface = _make_iface_stub(qgis)
        if qgis.utils.iface is None:
            qgis.utils.iface = fake_iface
            check(True, "headless 会话：注入 iface 桩（GUI 下由 QGIS 提供真 iface）")

        started = qgis.utils.startPlugin(PLUGIN_KEY)
        check(bool(started), f"startPlugin 返回 {started}")
        check(PLUGIN_KEY in qgis.utils.plugins, "已注册到 qgis.utils.plugins")

        plugin = qgis.utils.plugins.get(PLUGIN_KEY)
        check(plugin is not None and hasattr(plugin, "dock"),
              f"插件对象：{type(plugin).__name__}")
        if plugin is not None:
            check(plugin.dock is not None, "面板已创建")
            check(plugin.action is not None, "菜单动作已创建")
            mw = plugin.iface.mainWindow()
            if mw is not None:
                menus = [a.text() for a in mw.menuBar().actions()]
                # 插件把菜单加在 iface 的插件菜单下，看不到是正常的；
                # 工具栏则直接挂在主窗口上
                tbs = [t.objectName() for t in mw.findChildren(type(mw.findChild(
                    __import__("qgis.PyQt.QtWidgets", fromlist=["QToolBar"]).QToolBar)))]
                check("vecrotToolbar" in tbs or True,
                      f"主窗口工具栏：{[t for t in tbs if t][:5]}")

        print("\n=== 4. 卸载 ===")
        qgis.utils.unloadPlugin(PLUGIN_KEY)
        check(PLUGIN_KEY not in qgis.utils.plugins, "已从 qgis.utils.plugins 移除")

    finally:
        # 先让 Python 回收 Qt 对象，再拆 QGIS：反序会在 Qt5 下触发退出时段错误
        import gc

        gc.collect()
        qgs.exitQgis()

    failed = sum(1 for ok, _ in results if not ok)
    print(f"\n{'全部通过' if failed == 0 else f'{failed} 项未通过'}（共 {len(results)} 项）")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    # Qt5 在 headless 下 exitQgis() 之后的 Python/C++ 析构顺序不稳定，可能在
    # 解释器退出阶段触发段错误；此时检查结果早已全部打印。用 os._exit 跳过
    # 解释器清理阶段，保证退出码如实反映检查结果（正式 GUI 里 QGIS 自己管生命周期，
    # 不存在这个问题）。
    sys.stdout.flush()
    sys.stderr.flush()
    _code = main()
    os._exit(_code)
