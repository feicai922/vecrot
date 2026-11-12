"""在真实 QGIS 运行时里验证插件能被加载（M3 实机验证）。

用 QGIS 自带的 python 运行：

    "D:\\Program Files\\QGIS 3.44.14\\bin\\python-qgis-ltr.bat" ^
        deployment\\qgis_runtime_check.py

检查项：
1. QgsApplication 初始化（QGIS 宿主环境）
2. classFactory 返回插件对象
3. initGui 建面板/动作（不显示窗口，用 offscreen）
4. .ui 能被 uic 编译、控件齐全
5. AOI 工具的坐标转换函数在真实 QgsGeometry 上工作
6. unload 干净收尾（无异常、动作移除）
7. 成果加载器能把演示 GPKG 加成图层组

不做的事：不弹窗、不跑模型、不写用户 profile。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PLUGIN_PARENT = os.path.join(REPO, "qgis_plugin")

results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((bool(ok), msg))
    print(("  [OK]   " if ok else "  [FAIL] ") + msg)


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, PLUGIN_PARENT)
    sys.path.insert(0, REPO)

    from qgis.core import QgsApplication, QgsGeometry, QgsPointXY, QgsRectangle, QgsProject
    from qgis.core import QgsCoordinateReferenceSystem

    print("=== 1. QGIS 运行时 ===")
    qgs = QgsApplication([], False)
    qgs.initQgis()
    try:
        from qgis.core import Qgis
        from qgis.PyQt.QtCore import QT_VERSION_STR

        check(True, f"QGIS {Qgis.QGIS_VERSION} / Python {sys.version.split()[0]} / Qt {QT_VERSION_STR}")

        print("\n=== 2. 插件入口 ===")
        import vecrot_qgis
        check(hasattr(vecrot_qgis, "classFactory"), "包内有 classFactory")

        print("\n=== 3. .ui 编译 ===")
        from qgis.PyQt import uic

        ui_path = os.path.join(PLUGIN_PARENT, "vecrot_qgis", "ui", "dock_widget_base.ui")
        form_class, base_class = uic.loadUiType(ui_path)
        check(form_class is not None, f"uic.loadUiType 成功（{os.path.basename(ui_path)}）")

        print("\n=== 4. GisGeometry 上的 AOI 转换 ===")
        from vecrot_qgis.maptools.aoi_tool import ensure_polygon_in_crs, geometry_area_m2

        geom = QgsGeometry.fromRect(QgsRectangle(0, 0, 100, 50))
        crs_4326 = QgsCoordinateReferenceSystem("EPSG:4326")
        # 投影 CRS 之间转换（避免经纬度面积歧义）
        crs_utm = QgsCoordinateReferenceSystem("EPSG:32648")
        geom_utm = QgsGeometry.fromRect(QgsRectangle(400000, 3000000, 400100, 3000050))
        geojson, crs_str = ensure_polygon_in_crs(geom_utm, crs_utm, crs_utm)
        check(geojson["type"] == "Polygon" and len(geojson["coordinates"][0]) == 5,
              f"AOI → GeoJSON（{len(geojson['coordinates'][0])} 个顶点，CRS {crs_str}）")
        area = geometry_area_m2(geom_utm, crs_utm)
        check(area is not None and abs(area - 5000.0) < 1e-6, f"投影坐标系面积 {area} m²")
        area_geo = geometry_area_m2(geom, crs_4326)
        check(area_geo is None, "地理坐标系面积返回 None（不猜）")

        print("\n=== 5. 插件生命周期（offscreen）===")
        from qgis.utils import iface as real_iface

        class FakeIface:
            """不启动完整 GUI 时替代 iface：只提供插件用到的方法。"""

            def __init__(self):
                self.canvas = None
                self._actions = []

            def mainWindow(self):
                from qgis.PyQt.QtWidgets import QMainWindow

                if not hasattr(self, "_mw"):
                    self._mw = QMainWindow()
                return self._mw

            def addPluginToMenu(self, menu, action):
                self._actions.append(("menu", menu, action))

            def removePluginMenu(self, menu, action):
                self._actions.append(("unmenu", menu, action))

            def addToolBar(self, title):
                from qgis.PyQt.QtWidgets import QToolBar

                tb = QToolBar(title)
                self._mw.addToolBar(tb)
                return tb

            def addDockWidget(self, area, dock):
                self._mw.addDockWidget(area, dock)

            def removeDockWidget(self, dock):
                self._mw.removeDockWidget(dock)

            def mapCanvas(self):
                return None

        fake = FakeIface()
        plugin = vecrot_qgis.classFactory(fake)
        check(plugin is not None, f"classFactory 返回 {type(plugin).__name__}")

        plugin.initGui()
        check(plugin.dock is not None, "initGui 建好 Dock 面板")
        check(plugin.action is not None, "initGui 建好菜单动作")
        # 面板默认配置
        check(plugin.dock.manifestEdit.text().endswith("manifest.json")
              or plugin.dock.manifestEdit.text() == "",
              f"默认 manifest 探测：{plugin.dock.manifestEdit.text() or '（未找到，需手动选）'}")
        check(plugin.dock.scoreSpin.value() == 0.3, "默认分数阈值 0.3")
        check(plugin.dock.batchSpin.value() == 4, "默认批大小 4")
        check(plugin.dock.crsPolicyEdit.text() == "source", "默认输出 CRS = source")

        print("\n=== 6. 成果加载器（用演示 GPKG）===")
        demo_dir = os.path.join(REPO, "work", "demo", "output")
        if os.path.isfile(os.path.join(demo_dir, "manifest.json")):
            from vecrot_qgis.layers.result_loader import load_results

            project = QgsProject.instance()
            group = load_results(project, demo_dir)
            names = [c.name() for c in group.children()]
            check(len(names) == 2, f"加载 2 个图层：{names}")
            for child in group.children():
                layer = child.layer() if hasattr(child, "layer") else None
                if layer is not None:
                    check(layer.isValid() and layer.featureCount() > 0,
                          f"图层 {layer.name()} 有效，{layer.featureCount()} 个要素")
            project.removeAllMapLayers()
        else:
            check(False, "缺少演示成果，先运行 deployment/demo.py")

        print("\n=== 7. 卸载 ===")
        plugin.unload()
        check(plugin.dock is None, "unload 释放面板")
        check(plugin.action is None, "unload 移除动作")
        removed = [a for a in fake._actions if a[0] == "unmenu"]
        check(len(removed) == 1, "unload 从菜单移除")

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
