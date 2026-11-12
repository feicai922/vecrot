"""验证这一轮的四项改动。

    python deployment/run_in_qgis.py deployment/qgis_ui_round2_check.py

1. AOI 框选后**持久显示**在画布上（切换工具也不消失，点清除才消失）
2. 检测框 = 无填充线框；建筑轮廓 = 实心多边形（红色）
3. 模型搭配下拉来自固定预设，缺文件时禁止运行
4. 作业完成后自动加载成果（autoLoadCheck 默认开启）
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

RASTER = r"G:\贵州省数据集\images\test\rural_anshunshi-4_tile0000.tif"
OUT_DIR = os.path.join(REPO, "work", "qgis_incremental_094235",
                       "vecrot_20261111_094235")

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

    from qgis.core import (Qgis, QgsCoordinateReferenceSystem, QgsGeometry,
                           QgsPointXY, QgsProject, QgsRectangle, QgsWkbTypes)
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt.QtCore import QPoint, Qt
    from qgis.PyQt.QtGui import QMouseEvent
    from qgis.PyQt.QtWidgets import QApplication, QMainWindow, QToolBar

    print("=== QGIS " + Qgis.QGIS_VERSION + " ===")

    class Iface:
        def __init__(self):
            self._mw = QMainWindow()
            self._canvas = QgsMapCanvas()
            self._canvas.resize(400, 300)

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
            return self._canvas

    iface = Iface()
    canvas = iface._canvas
    crs = QgsCoordinateReferenceSystem("EPSG:32648")
    canvas.setDestinationCrs(crs)
    canvas.setExtent(QgsRectangle(400000, 3000000, 401000, 3000750))
    canvas.show()
    QApplication.processEvents()

    import vecrot_qgis

    plugin = vecrot_qgis.classFactory(iface)
    plugin.initGui()
    dock = plugin.dock
    project = QgsProject.instance()

    # ------------------------------------------------ 1. AOI 持久显示
    print("\n=== 1. AOI 框选后持久显示 ===")
    dock.drawAoiButton.setChecked(True)
    tool = dock._aoi_tool
    check(tool is not None, "框选工具已创建")

    def mouse(kind, x, y):
        etype = {"press": QMouseEvent.MouseButtonPress,
                 "move": QMouseEvent.MouseMove,
                 "release": QMouseEvent.MouseButtonRelease}[kind]
        ev = QMouseEvent(etype, QPoint(x, y), Qt.LeftButton, Qt.LeftButton,
                         Qt.NoModifier)
        getattr(tool, "canvas" + kind.capitalize() + "Event")(ev)

    mouse("press", 80, 60)
    mouse("move", 300, 240)
    mouse("release", 300, 240)
    check(dock.aoi_geometry is not None, "AOI 几何已记录")
    def band_has_geom(b):
        if b is None:
            return False
        g = b.asGeometry()
        return g is not None and not g.isEmpty()

    band = getattr(tool, "_band", None)
    check(band_has_geom(band), "画布上的预览矩形非空（框选后可见）")
    check(not dock.drawAoiButton.isChecked(), "框选后自动退出框选模式")

    # 切走工具：预览应保留
    canvas.setMapTool(None)
    QApplication.processEvents()
    check(band_has_geom(band), "切换地图工具后预览仍保留（不会被清掉）")

    # 点清除：预览消失
    dock._clear_aoi()
    check(not band_has_geom(band), "点「清除」后预览消失")
    check(dock.aoi_geometry is None, "AOI 记录已清空")

    # ------------------------------------------------ 2. 样式
    print("\n=== 2. 框=线框 / 轮廓=实心 ===")
    if os.path.isdir(OUT_DIR):
        from vecrot_qgis.layers.result_loader import load_results

        group = load_results(project, OUT_DIR)
        layers = {c.layer().name(): c.layer() for c in group.children()}
        box_ly = next((l for n, l in layers.items() if n.endswith("检测框")), None)
        build_ly = next((l for n, l in layers.items() if n.endswith("建筑轮廓")), None)
        check(box_ly is not None and build_ly is not None, "两个图层都已加载")

        # 用离屏渲染采样判断样式，比符号内省可靠：
        # QgsFillSymbol 没有 outlineColor()，而逐层访问 symbolLayer 在 headless 下
        # 会让进程直接崩（实测 exit=127）。
        from qgis.core import QgsMapRendererParallelJob, QgsMapSettings
        from qgis.PyQt.QtCore import QSize
        from qgis.PyQt.QtGui import QColor

        def render_layers(layers, size=500):
            ext = layers[0].extent()
            ext = ext.buffered(max(ext.width(), ext.height()) * 0.15)
            ms = QgsMapSettings()
            ms.setLayers(layers)
            ms.setDestinationCrs(layers[0].crs())
            ms.setExtent(ext)
            ms.setOutputSize(QSize(size, size))
            ms.setBackgroundColor(QColor(255, 255, 255))
            job = QgsMapRendererParallelJob(ms)
            job.start()
            job.waitForFinished()
            return job.renderedImage()

        def color_hist(img):
            from collections import Counter

            cnt = Counter()
            for y in range(0, img.height(), 2):
                for x in range(0, img.width(), 2):
                    c = img.pixelColor(x, y)
                    if c.red() > 250 and c.green() > 250 and c.blue() > 250:
                        continue           # 跳过白底
                    cnt[(c.red() // 24 * 24, c.green() // 24 * 24,
                         c.blue() // 24 * 24)] += 1
            return cnt

        # 只渲染检测框：应几乎只有彩色线，没有大面积填充
        if box_ly is not None:
            img = render_layers([box_ly])
            hist = color_hist(img)
            total = sum(hist.values())
            # 填充 alpha=0 时，非白像素只来自线，占比应很小
            check(total < img.width() * img.height() * 0.10,
                  "检测框只画线：非白像素占 %.1f%%（<10%%）"
                  % (100.0 * total / (img.width() * img.height())))
            # 线色应偏暖（红橙 #d6604d 一侧）到偏冷（蓝）
            warm = sum(v for (r, g, b), v in hist.items() if r > b + 20)
            cool = sum(v for (r, g, b), v in hist.items() if b > r + 20)
            check(warm > 0 and cool > 0,
                  "检测框分级色覆盖冷暖两端（暖 %d / 冷 %d）" % (warm, cool))

        # 只渲染轮廓：应有大面积红色填充
        if build_ly is not None:
            img = render_layers([build_ly])
            hist = color_hist(img)
            total = sum(hist.values())
            red = sum(v for (r, g, b), v in hist.items() if r > 150 and g < 120 and b < 120)
            check(total > 0, "轮廓有可见内容（非白像素 %d）" % total)
            check(red > total * 0.5,
                  "轮廓以红色实心面为主：红色占 %.0f%%" % (100.0 * red / max(total, 1)))

        # 图层顺序：轮廓在下、框在上
        names = [c.layer().name() for c in group.children()]
        check(names[0].endswith("建筑轮廓"),
              "轮廓在框之下（%s 在上）" % names[-1])
        project.removeAllMapLayers()
    else:
        check(False, "缺少测试成果目录：" + OUT_DIR)

    # ------------------------------------------------ 3. 预设
    print("\n=== 3. 固定模型搭配 ===")
    combo = dock.modelPresetCombo
    check(combo.count() >= 2, "预设下拉有 %d 项" % combo.count())
    labels = [combo.itemText(i) for i in range(combo.count())]
    for lb in labels:
        print("     · " + lb)
    # 默认应是第一个预设（当前是带场景分类的版本）。不写死 key，
    # 只验证「默认项 == PRESETS[0] 且可用」，避免每次调整顺序就红一次。
    from vecrot_qgis.jobs.presets import PRESETS

    check(combo.currentData() == PRESETS[0].key,
          "默认选中第一个预设：%s" % combo.currentData())
    check(PRESETS[0].is_available(),
          "默认预设可用（文件齐备）")
    detail = dock.modelDetailView.text()
    print("     详情（前 3 行）：")
    for line in detail.splitlines()[:3]:
        print("       " + line)
    check("✓" in detail, "详情里显示了文件就绪标记")

    # RT-DETR 预设：应可用且显示正确的后端
    idx = next((i for i in range(combo.count())
                if combo.itemData(i) == "rtdetr_roof"), -1)
    check(idx >= 0, "存在 RT-DETR 预设")
    if idx >= 0:
        combo.setCurrentIndex(idx)
        detail = dock.modelDetailView.text()
        check("rtdetr_hf" in detail, "详情显示后端 rtdetr_hf")
        check("✓" in detail, "RT-DETR 权重文件就绪")
        preset = dock._current_preset()
        flds = preset.to_job_fields()
        check(flds.get("detector_profile") == "rtdetr_hf",
              "展开 profile=%s" % flds.get("detector_profile"))
        check(flds.get("detector_num_classes") == 1,
              "展开类别数=%s" % flds.get("detector_num_classes"))
        combo.setCurrentIndex(0)

    # 缺文件的预设应当被标出并禁止运行（用一个临时构造的不可用预设验证逻辑）
    from vecrot_qgis.jobs.presets import ModelPreset

    bad = ModelPreset(key="_test_missing", label="测试缺失",
                      backend="strip_s_gzda",
                      files={"weights": os.path.join(REPO, "no_such_file.pth")})
    check(not bad.is_available(), "缺文件的预设 is_available=False")
    check(len(bad.missing()) == 1, "missing() 报出 1 个缺失项")

    # ------------------------------------------------ 4. 自动加载
    print("\n=== 4. 自动加载开关 ===")
    check(dock.autoLoadCheck.isChecked(), "autoLoadCheck 默认勾选")
    check(dock.autoLoadCheck.text().find("自动加载") >= 0,
          "开关文案：%s" % dock.autoLoadCheck.text())

    print("\n" + ("全部通过" if all(ok for ok, _ in results)
                  else "%d 项未通过" % sum(1 for ok, _ in results if not ok))
          + "（共 %d 项）" % len(results))
    return 0 if all(ok for ok, _ in results) else 1


if __name__ == "__main__":
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(main())
