"""验证两项新交互：项目图层选择 + 增量提取追加。

    python deployment/run_in_qgis.py deployment/qgis_incremental_check.py

流程：
1. 往项目里加两个栅格图层，确认下拉框自动列出
2. 选中其中一个，确认影像信息与路径同步
3. 首次完整作业
4. 加载成果到项目
5. 框选第二块区域 -> 增量作业 -> 确认要素数增长、图层被刷新
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

TILES = [
    r"G:\贵州省数据集\images\test\rural_anshunshi-4_tile0000.tif",
    r"G:\贵州省数据集\images\test\rural_anshunshi-4_tile0002.tif",
]
OUT_ROOT = os.path.join(REPO, "work", "qgis_incremental")

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

    from qgis.core import (Qgis, QgsGeometry, QgsProject, QgsRasterLayer,
                           QgsRectangle, QgsVectorLayer)
    from qgis.PyQt.QtCore import QEventLoop, QTimer
    from qgis.PyQt.QtWidgets import QMainWindow, QToolBar

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
    project = QgsProject.instance()

    # ---------------------------------------------------- 1. 下拉框
    print("\n=== 1. 项目图层出现在下拉框 ===")
    layers = []
    for p in TILES:
        if not os.path.isfile(p):
            check(False, "测试瓦片不存在：" + p)
            return 1
        ly = QgsRasterLayer(p, os.path.basename(p), "gdal")
        check(ly.isValid(), "加载 " + os.path.basename(p))
        project.addMapLayer(ly)
        layers.append(ly)
    dock._refresh_layers()
    n = dock.rasterCombo.count()
    check(n == len(TILES), "下拉框列出 %d 个栅格图层" % n)
    print("     列表：" + str([dock.rasterCombo.itemText(i) for i in range(n)]))
    check(dock.rasterCombo.isEnabled(), "下拉框可用")

    # ---------------------------------------------------- 2. 选中同步
    print("\n=== 2. 选中第二项，信息同步 ===")
    dock.rasterCombo.setCurrentIndex(1)
    check(dock.rasterPathEdit.text() == TILES[1],
          "路径已同步：" + os.path.basename(dock.rasterPathEdit.text()))
    check("1024" in dock.rasterInfoLabel.text(), "影像信息已显示")
    check(dock.drawAoiButton.isEnabled(), "框选按钮已启用")

    # ---------------------------------------------------- 3. 首次作业
    print("\n=== 3. 首次完整作业 ===")
    out_root = OUT_ROOT + "_" + time.strftime("%H%M%S")
    dock.outputDirEdit.setText(out_root)
    target = layers[1]
    dock._on_aoi_drawn(QgsGeometry.fromRect(target.extent()), target.crs())

    loop = QEventLoop()
    done = {"code": None}
    dock._run()
    check(dock.bridge is not None, "进程桥已创建")
    if dock.bridge is not None:
        dock.bridge.finished.connect(
            lambda c, s: (done.__setitem__("code", c), loop.quit()))
        QTimer.singleShot(900000, loop.quit)
        loop.exec()
    check(done["code"] == 0, "首次作业退出码 %s" % done["code"])
    check(bool(dock.last_out_dir), "成果目录：" + str(dock.last_out_dir))
    base_dir = dock.last_out_dir
    if not base_dir:
        return 1

    def count_layer(layer: str) -> int:
        """从磁盘重新打开统计：已缓存的 QgsVectorLayer 看不到后来追加的要素。"""
        path = os.path.join(base_dir, layer + ".gpkg")
        ly = QgsVectorLayer(path + "|layername=" + layer, layer, "ogr")
        cnt = ly.featureCount() if ly.isValid() else -1
        ly.setDataSource("", layer, "ogr")
        return cnt

    n_boxes_1 = count_layer("boxes")
    n_build_1 = count_layer("buildings")
    print("     首次成果：boxes=%d buildings=%d" % (n_boxes_1, n_build_1))
    check(n_boxes_1 > 0, "首次检出 %d 个框" % n_boxes_1)

    # ---------------------------------------------------- 4. 自动加载
    print("\n=== 4. 成果已自动加载到项目 ===")
    # 作业完成后面板自动加载成果（autoLoadCheck 默认勾选），这里验证自动加载
    check(dock.autoLoadCheck.isChecked(), "自动加载默认开启")
    loaded = [l for l in project.mapLayers().values()
              if l.name().endswith("检测框") or l.name().endswith("建筑轮廓")]
    check(len(loaded) == 2, "项目里有 %d 个成果图层（自动加载）" % len(loaded))
    check(dock.continueButton.isEnabled(), "加载后「继续提取」可用")

    # ---------------------------------------------------- 5. 增量作业
    print("\n=== 5. 框选第二块区域 -> 增量作业 ===")
    ext = target.extent()
    w, h = ext.width(), ext.height()
    sub = QgsRectangle(ext.xMinimum(), ext.yMinimum(),
                       ext.xMinimum() + w * 0.55, ext.yMinimum() + h * 0.55)
    dock._on_aoi_drawn(QgsGeometry.fromRect(sub), target.crs())
    check(dock.aoi_geometry is not None, "第二块 AOI 已记录")

    done2 = {"code": None}
    loop2 = QEventLoop()
    dock._continue_extraction()
    check(dock.bridge is not None, "增量作业进程桥已创建")
    if dock.bridge is not None:
        dock.bridge.finished.connect(
            lambda c, s: (done2.__setitem__("code", c), loop2.quit()))
        QTimer.singleShot(900000, loop2.quit)
        loop2.exec()
    check(done2["code"] == 0, "增量作业退出码 %s" % done2["code"])

    # ---------------------------------------------------- 6. 追加结果
    print("\n=== 6. 追加结果检查 ===")
    n_boxes_2 = count_layer("boxes")
    n_build_2 = count_layer("buildings")
    print("     追加后：boxes=%d buildings=%d（原 %d / %d）"
          % (n_boxes_2, n_build_2, n_boxes_1, n_build_1))
    check(n_boxes_2 > n_boxes_1, "框数增加：%d -> %d" % (n_boxes_1, n_boxes_2))
    check(n_build_2 > n_build_1, "建筑数增加：%d -> %d" % (n_build_1, n_build_2))

    ids = set()
    ly = QgsVectorLayer(os.path.join(base_dir, "boxes.gpkg") + "|layername=boxes",
                        "b", "ogr")
    for f in ly.getFeatures():
        ids.add(f["det_id"])
    total = ly.featureCount()
    ly.setDataSource("", "b", "ogr")
    check(len(ids) == total, "det_id 唯一：%d/%d" % (len(ids), total))

    # ---------------------------------------------------- 7. 图层刷新
    print("\n=== 7. 项目图层已刷新 ===")
    refreshed = [l for l in project.mapLayers().values()
                 if l.name().endswith("检测框")]
    if refreshed:
        n_feat = refreshed[0].featureCount()
        check(n_feat == n_boxes_2, "项目检测框图层已刷新：%d" % n_feat)

    if not all(ok for ok, _ in results):
        print("\n--- 面板日志（尾部）---")
        for line in dock.logEdit.toPlainText().splitlines()[-30:]:
            print("   ", line)

    failed = sum(1 for ok, _ in results if not ok)
    print("\n" + ("全部通过" if failed == 0 else "%d 项未通过" % failed)
          + "（共 %d 项）" % len(results))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(main())
