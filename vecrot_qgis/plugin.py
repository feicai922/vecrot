"""插件生命周期（开发文档 §10）。

initGui 建菜单/工具栏/面板；unload 断开信号、恢复地图工具、回收进程。
"""
from __future__ import annotations

import os

from qgis.core import QgsApplication, QgsProject
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction

from .ui.dock_widget import VecrotDockWidget


class VecrotPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dock = None
        self._toolbar = None

    # ---- 生命周期
    def initGui(self):                        # noqa: N802
        icon = self._load_icon()
        self.action = QAction(icon, "Vecrot 建筑提取", self.iface.mainWindow())
        self.action.setCheckable(True)
        self.action.setObjectName("vecrotExtractAction")
        self.action.toggled.connect(self._toggle_dock)
        self.iface.addPluginToMenu("&Vecrot", self.action)
        self._toolbar = self.iface.addToolBar("Vecrot")
        self._toolbar.setObjectName("vecrotToolbar")
        self._toolbar.addAction(self.action)

        self.dock = VecrotDockWidget(self.iface, self.iface.mainWindow())
        self.dock.setObjectName("vecrotDock")
        self.dock.closed.connect(lambda: self.action.setChecked(False))
        self.iface.addDockWidget(0x2, self.dock)      # Qt.RightDockWidgetArea
        self.dock.hide()

    def unload(self):
        try:
            if self.dock is not None:
                self.dock.cleanup()
                self.iface.removeDockWidget(self.dock)
                self.dock.deleteLater()
                self.dock = None
            if self.action is not None:
                self.iface.removePluginMenu("&Vecrot", self.action)
                self.action.toggled.disconnect()
                self.action = None
            if self._toolbar is not None:
                self._toolbar.deleteLater()
                self._toolbar = None
            # 恢复地图工具，避免卸载后画布仍停在框选工具
            canvas = self.iface.mapCanvas()
            if canvas is not None:
                canvas.unsetMapTool(canvas.mapTool())
        except Exception:                       # noqa: BLE001 - 卸载不应抛错打断 QGIS
            pass

    # ---- 面板开关
    def _load_icon(self) -> QIcon:
        """加载插件图标。

        优先 PNG：部分 QGIS 构建在缺少 Qt SVG 图像格式插件时加载 SVG 会得到空图标，
        表现为工具栏上一个看不见的按钮。SVG 仅作回退。
        """
        here = os.path.dirname(__file__)
        for name in ("icon.png", "icon.svg"):
            path = os.path.join(here, name)
            if os.path.isfile(path):
                icon = QIcon(path)
                if not icon.isNull() and not icon.availableSizes() == []:
                    return icon
                if not icon.isNull():
                    return icon
        return QIcon()

    def _toggle_dock(self, checked: bool) -> None:
        if self.dock is None:
            return
        self.dock.setVisible(bool(checked))
        if checked:
            self.dock.raise_()
