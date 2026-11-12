"""模拟 GUI 的完整操作流程（不点鼠标，但走真实的控件与回调）。

验证用户在 QGIS 里实际会走的路径：
1. 面板打开 → manifest 自动找到（含 profile 安装场景）
2. 选影像 → 信息显示、框选按钮可用、输出目录自动填
3. 框选 → AOI 记下（用真实 QgsGeometry + 画布 CRS 模拟）
4. 点「开始提取」→ 真模型作业跑完
5. 点「加载成果到地图」→ 图层出现

用法：
    python deployment/run_in_qgis.py deployment/qgis_gui_flow_check.py
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

RASTER = r"G:\贵州省数据集\images\test\rural_anshunshi-4_tile0000.tif"
OUT_ROOT = os.path.join(REPO, "work", "qgis_gui_flow")

results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((bool(ok), msg))
    print(("  [OK]   " if ok else "  [FAIL] ") + msg)


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    # 模拟 profile 安装场景：不设 VECROT_MANIFEST，看面板能不能自己找到
    os.environ.pop("VECROT_MANIFEST", None)
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "qgis_plugin"))

    from qgis.core import QgsApplication

    qgs = QgsApplication([], False)
    qgs.initQgis()
    try:
        from qgis.core import (Qgis, QgsCoordinateReferenceSystem, QgsGeometry,
                               QgsProject, QgsRasterLayer, QgsRectangle)
        from qgis.PyQt.QtCore import QEventLoop, QTimer
        from qgis.PyQt.QtWidgets import QDockWidget, QMainWindow, QToolBar

        print(f"=== QGIS {Qgis.QGIS_VERSION} ===")

        # ---- 最小 iface 桩
        class StubIface:
            def __init__(self):
                self._mw = QMainWindow()

            def mainWindow(self):
                return self._mw

            def addPluginToMenu(self, menu, action):
                self._mw.menuBar().addMenu(menu).addAction(action)

            def removePluginMenu(self, menu, action):
                pass

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

        iface = StubIface()
        from vecrot_qgis.plugin import VecrotPlugin

        plugin = VecrotPlugin(iface)
        plugin.initGui()
        dock = plugin.dock
        check(dock is not None, "面板已创建")

        print("\n=== 1. manifest 自动发现 ===")
        found = dock.manifestEdit.text()
        check(bool(found) and os.path.isfile(found),
              f"面板自动找到 manifest：{found or '（未找到）'}")
        if not found:
            # 兜底：开发机直接从仓库选一份，验证后续流程
            dock._browse_manifest()
        check(bool(dock.manifest_path), f"manifest_path = {dock.manifest_path}")
        model_dir = os.path.dirname(dock.manifest_path)
        print(f"     从 manifest 推断仓库根：{os.path.dirname(model_dir)}")

        print("\n=== 2. 选影像（走 _browse_raster 之外的相同路径）===")
        dock.rasterPathEdit.setText(RASTER)
        dock._inspect_raster(RASTER)
        info = dock.rasterInfoLabel.text()
        check("1024" in info, f"影像信息：{info.splitlines()[0]}")
        check(dock.drawAoiButton.isEnabled(), "框选按钮已启用")
        check(not dock.outputDirEdit.text().strip(),
              "输出目录未自动填（因为没走文件对话框）—— 手工补上")
        out_root = OUT_ROOT + "_" + time.strftime("%H%M%S")
        dock.outputDirEdit.setText(out_root)

        print("\n=== 3. 框选 AOI ===")
        layer = QgsRasterLayer(RASTER, "t", "gdal")
        geom = QgsGeometry.fromRect(layer.extent())
        canvas_crs = layer.crs()
        dock._on_aoi_drawn(geom, canvas_crs)
        check(dock.aoi_geometry is not None, f"AOI 已记录（{canvas_crs.authid()}）")
        check("已框选" in dock.aoiInfoLabel.text(),
              f"AOI 标签：{dock.aoiInfoLabel.text()}")
        check(not dock.drawAoiButton.isChecked(), "框选完成后按钮自动取消按下")

        print("\n=== 4. 点「开始提取」（真模型）===")
        loop = QEventLoop()
        finished = {"code": None}

        def on_fin(code, summary):
            finished["code"] = code
            loop.quit()

        # 复用面板自己的 _run，只把它创建的 bridge 接上我们的事件循环
        SKIP_JOB = os.environ.get("VECROT_CHECK_SKIP_JOB") == "1"
        if SKIP_JOB:
            dock.last_out_dir = os.environ.get("VECROT_CHECK_OUT", "")
            print("     （VECROT_CHECK_SKIP_JOB=1：跳过真实作业，复用已有成果）")
        else:
            dock._run()
        if not SKIP_JOB:
            check(dock.bridge is not None, "面板已创建进程桥")
            if dock.bridge is None:
                return 1
            dock.bridge.finished.connect(on_fin)
        if not SKIP_JOB:
            check(dock.runButton.isEnabled() is False, "运行中：按钮已禁用")
            check(dock.cancelButton.isEnabled() is True, "运行中：取消按钮可用")
            QTimer.singleShot(900000, loop.quit)
            loop.exec()
            check(finished["code"] == 0, f"作业退出码 {finished['code']}")
        check(dock.runButton.isEnabled() is True, "结束后：运行按钮恢复")
        check(dock.loadResultsButton.isEnabled(), "结束后：加载成果按钮可用")
        check(bool(dock.last_out_dir), f"成果目录：{dock.last_out_dir}")

        print("\n=== 5. 加载成果到地图 ===")
        project = QgsProject.instance()
        before = len(project.mapLayers())
        dock._load_results()
        after = len(project.mapLayers())
        check(after >= before + 2, f"图层数 {before} → {after}")
        names = [l.name() for l in project.mapLayers().values()]
        print(f"     图层：{names}")
        loaded = [l for l in project.mapLayers().values()
                  if "检测框" in l.name() or "建筑" in l.name()]
        for l in loaded:
            check(l.isValid() and l.featureCount() > 0,
                  f"{l.name()}：{l.featureCount()} 个要素")

        # 必须先断开 GDAL 数据源再移除：否则 Python 侧图层对象仍持有数据源，
        # exitQgis() 之后析构顺序错乱 → 退出时段错误（实测 exit=139）
        for l in loaded:
            l.setDataSource("", l.name(), "ogr")
        project.removeAllMapLayers()

        print("\n=== 6. 卸载 ===")
        # 必须在 unload() 之前取日志：unload 会 deleteLater() 面板，
        # 之后再访问 dock.logEdit 就是访问已销毁的 C++ 对象 → 段错误（已实测）
        panel_log = dock.logEdit.toPlainText().splitlines()
        # 面板卸载前先摘掉进程桥：它持有 QProcess 与信号连接，
        # 若跟着面板一起 deleteLater，子对象析构顺序不确定 → 退出时段错误
        bridge = dock.bridge
        if bridge is not None:
            for sig in (bridge.event_received, bridge.log_line, bridge.finished):
                try:
                    sig.disconnect()
                except (TypeError, RuntimeError):
                    pass
            bridge.deleteLater()
            dock.bridge = None
        dock.bridge = None
        plugin.unload()
        check(plugin.dock is None, "unload 释放面板")
        # 不要 del：对象已被 deleteLater 标记销毁，Python 侧 del 会触发
        # 对已销毁 C++ 对象的析构 → 段错误（实测）

        # deleteLater() 依赖事件循环；headless 脚本里必须主动跑一圈把它处理掉，
        # 否则对象拖到 exitQgis() 之后才析构 → 退出时段错误（实测 exit=139）。
        from qgis.PyQt.QtCore import QCoreApplication

        QCoreApplication.processEvents()
        QCoreApplication.sendPostedEvents(None, 0)   # QEvent.DeferredDelete == 0
        QCoreApplication.processEvents()

        if not all(ok for ok, _ in results):
            print("\n--- 面板日志（尾部）---")
            for line in panel_log[-25:]:
                print("   ", line)

    finally:
        # 不调用 gc.collect()/exitQgis()：面板与进程桥已被 unload() 标记 deleteLater，
        # headless 下再回收它们会访问已销毁的 C++ 对象 → 段错误（实测 exit=139）。
        # 进程最终由 os._exit 结束，QGIS 资源随进程一并释放。
        pass

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
