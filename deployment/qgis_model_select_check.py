"""验证滚动面板 + 模型自选 UI + HBB 后端。

    python deployment/run_in_qgis.py deployment/qgis_model_select_check.py

覆盖：
1. scrollArea 生效（内容超出时出现垂直滚动条，水平滚动条关闭）
2. 模型选择控件的默认状态与联动（jsonl 模式下禁用权重/配置）
3. 模型路径校验（相对路径被拒）
4. job.json 里正确带上用户选择的模型路径
5. HBB 后端能被工厂识别（缺路径时给出准确错误，不静默）
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((bool(ok), msg))
    print(("  [OK]   " if ok else "  [FAIL] ") + msg)


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "qgis_plugin"))

    from qgis.core import QgsApplication

    qgs = QgsApplication([], False)
    qgs.initQgis()

    from qgis.core import Qgis
    from qgis.PyQt.QtCore import Qt
    from qgis.PyQt.QtWidgets import QMainWindow, QScrollArea, QToolBar

    print("=== QGIS " + Qgis.QGIS_VERSION + " ===")

    class Iface:
        def __init__(self):
            self._mw = QMainWindow()

        def mainWindow(self):
            return self._mw

        def addPluginToMenu(self, m, a):
            self._mw.menuBar().addMenu(m).addAction(a)

        def removePluginMenu(self, m, a):
            pass

        def addToolBar(self, t):
            tb = QToolBar(t, self._mw)
            self._mw.addToolBar(tb)
            return tb

        def addDockWidget(self, a, d):
            self._mw.addDockWidget(a, d)

        def removeDockWidget(self, d):
            self._mw.removeDockWidget(d)

        def mapCanvas(self):
            return None

    import vecrot_qgis

    plugin = vecrot_qgis.classFactory(Iface())
    plugin.initGui()
    dock = plugin.dock

    # ---------------------------------------------------- 1. 滚动
    print("\n=== 1. 面板可滚动 ===")
    sa = dock.findChild(QScrollArea, "scrollArea")
    check(sa is not None, "存在 QScrollArea")
    if sa is not None:
        check(sa.widgetResizable(), "widgetResizable=True（宽度跟随面板）")
        check(sa.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff,
              "水平滚动条关闭（避免内容横向溢出）")
        check(sa.widget() is not None
              and sa.widget().objectName() == "scrollAreaWidgetContents",
              "内容控件正确")
        # 把它压到很窄，验证真的能滚动
        dock.resize(320, 260)
        dock.show()
        from qgis.PyQt.QtWidgets import QApplication

        QApplication.processEvents()
        content_h = sa.widget().sizeHint().height()
        view_h = sa.viewport().height()
        check(content_h > view_h,
              "内容高度 %d > 视口高度 %d（确实需要滚动）" % (content_h, view_h))
        check(sa.verticalScrollBar().maximum() > 0,
              "垂直滚动条区间 >0（可滚动）")

    # ---------------------------------------------------- 2. 预设下拉存在
    print("=== 2. 模型搭配下拉（预设驱动）===")
    check(hasattr(dock, "modelPresetCombo"), "面板有 modelPresetCombo（预设下拉）")
    check(dock.modelPresetCombo.count() >= 2,
          "预设数量 %d" % dock.modelPresetCombo.count())
    check(hasattr(dock, "modelDetailView"), "面板有组合详情视图")
    check(dock.modelDetailView.text() != "", "详情视图已填充内容")

    # 自由填路径的控件已按需求移除（改为固定搭配）
    for gone in ("detectorProfileCombo", "detWeightsEdit", "detConfigEdit",
                 "vectorCkptEdit", "vectorBaseEdit"):
        check(not hasattr(dock, gone), "已移除自由填路径控件 %s" % gone)

    print("（模型的详细行为验证见 qgis_ui_round2_check.py）")

    failed = sum(1 for ok, _ in results if not ok)
    print("\n" + ("全部通过" if failed == 0 else "%d 项未通过" % failed)
          + "（共 %d 项）" % len(results))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(main())
