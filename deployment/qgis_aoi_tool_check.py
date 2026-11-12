"""真实实例化 AOI 框选工具并模拟鼠标拖拽。

之前的检查脚本都是直接调 ``_on_aoi_drawn()`` 模拟框选，从未真正构造过
``AoiRectangleTool``，因此漏掉了 ``QgsRubberBand(canvas, False)`` 在 QGIS 3.44
上抛 TypeError 的问题（实战报错：
``QgsRubberBand(): argument 2 has unexpected type 'bool'``）。

本脚本覆盖：
1. 构造工具（会创建 QgsRubberBand）
2. activate / deactivate 生命周期
3. 用合成的 QMouseEvent 模拟 press → move → release
4. 信号是否发出、几何是否非空
5. 重复框选多次（reset 路径）
6. 极小框（宽度为 0）应被忽略

    python deployment/run_in_qgis.py deployment/qgis_aoi_tool_check.py
"""
from __future__ import annotations

import os
import sys

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

    from qgis.core import (Qgis, QgsCoordinateReferenceSystem, QgsGeometry,
                           QgsPointXY, QgsProject, QgsRectangle)

    print("=== QGIS " + Qgis.QGIS_VERSION + " ===")

    # ---- 需要一个真实 canvas 才能构造 QgsRubberBand，用最小可用的 QgsMapCanvas
    from qgis.gui import QgsMapCanvas
    from qgis.PyQt.QtCore import QPoint, Qt
    from qgis.PyQt.QtGui import QMouseEvent
    from qgis.PyQt.QtWidgets import QApplication

    canvas = QgsMapCanvas()
    canvas.resize(400, 300)
    crs_utm = QgsCoordinateReferenceSystem("EPSG:32648")
    canvas.setDestinationCrs(crs_utm)
    canvas.setExtent(QgsRectangle(400000, 3000000, 401000, 3000750))
    canvas.show()
    QApplication.processEvents()

    print("\n=== 1. 构造工具（关键：QgsRubberBand 签名）===")
    from vecrot_qgis.maptools.aoi_tool import AoiRectangleTool, POLYGON_TYPE

    print("     POLYGON_TYPE = %r" % (POLYGON_TYPE,))
    try:
        tool = AoiRectangleTool(canvas)
        check(True, "AoiRectangleTool 构造成功（rubber band 已创建）")
    except TypeError as exc:
        check(False, "构造失败：%s" % exc)
        return 1
    except Exception as exc:                        # noqa: BLE001
        check(False, "构造异常：%s: %s" % (type(exc).__name__, exc))
        return 1

    print("\n=== 2. activate / deactivate ===")
    try:
        tool.activate()
        check(True, "activate 成功")
        tool.deactivate()
        check(True, "deactivate 成功（reset 路径已覆盖）")
    except Exception as exc:                        # noqa: BLE001
        check(False, "生命周期失败：%s: %s" % (type(exc).__name__, exc))

    print("\n=== 3. 模拟鼠标拖拽 ===")
    drawn: list = []

    def on_drawn(geom, crs):
        drawn.append((geom, crs))

    tool.aoi_drawn.connect(on_drawn)
    canvas.setMapTool(tool)

    def mouse(kind, x, y, button=Qt.LeftButton):
        etype = {"press": QMouseEvent.MouseButtonPress,
                 "move": QMouseEvent.MouseMove,
                 "release": QMouseEvent.MouseButtonRelease}[kind]
        ev = QMouseEvent(etype, QPoint(x, y), button, button, Qt.NoModifier)
        if kind == "press":
            tool.canvasPressEvent(ev)
        elif kind == "move":
            tool.canvasMoveEvent(ev)
        else:
            tool.canvasReleaseEvent(ev)

    try:
        mouse("press", 100, 80)
        check(tool._start is not None, "按下后记录起点")
        mouse("move", 250, 200)
        check(True, "移动事件未抛异常（rubber band 已更新）")
        mouse("release", 250, 200)
        check(len(drawn) == 1, "松开后发出 aoi_drawn 信号（%d 次）" % len(drawn))
    except Exception as exc:                        # noqa: BLE001
        import traceback

        traceback.print_exc()
        check(False, "拖拽失败：%s: %s" % (type(exc).__name__, exc))

    if drawn:
        geom, crs = drawn[0]
        check(not geom.isEmpty(), "AOI 几何非空")
        check(crs.isValid() and crs.authid() == "EPSG:32648",
              "AOI CRS = %s" % (crs.authid() or crs.description()))
        check(geom.isGeosValid(), "AOI 几何在 GEOS 下有效")
        rect_area = geom.area()
        check(rect_area > 0, "AOI 面积 %.1f m²" % rect_area)
        # 画布范围 1000×750 m 映射到 400×300 px，所以 150×120 px
        # ≈ (150/400*1000) × (120/300*750) = 375 × 300 = 112500 m²
        expected = (150 / 400 * 1000) * (120 / 300 * 750)
        check(abs(rect_area - expected) / expected < 0.05,
              "面积与画布比例一致：%.0f m²（期望约 %.0f）" % (rect_area, expected))

    print("\n=== 4. 连续框选第二次（覆盖 reset 路径）===")
    try:
        mouse("press", 50, 50)
        mouse("move", 300, 250)
        mouse("release", 300, 250)
        check(len(drawn) == 2, "第二次框选也发出信号（累计 %d 次）" % len(drawn))
    except Exception as exc:                        # noqa: BLE001
        check(False, "第二次框选失败：%s" % exc)

    print("\n=== 5. 零面积框应被忽略 ===")
    before = len(drawn)
    try:
        mouse("press", 120, 120)
        mouse("release", 120, 120)                  # 同一个点
        check(len(drawn) == before, "零面积框未发出信号")
    except Exception as exc:                        # noqa: BLE001
        check(False, "零面积框处理失败：%s" % exc)

    print("\n=== 6. GeoJSON 转换 ===")
    from vecrot_qgis.maptools.aoi_tool import ensure_polygon_in_crs, geometry_area_m2

    if drawn:
        geom, crs = drawn[0]
        gj, crs_str = ensure_polygon_in_crs(geom, crs, crs)
        ring = gj["coordinates"][0]
        check(gj["type"] == "Polygon", "GeoJSON 类型正确")
        check(len(ring) == 5, "闭合环有 5 个点（4 角 + 闭合）")
        check(ring[0] == ring[-1], "首尾点相同（已闭合）")
        check(crs_str == "EPSG:32648", "CRS 字符串 = %s" % crs_str)
        area = geometry_area_m2(geom, crs)
        check(area is not None and area > 0,
              "投影坐标系面积可算：%.1f m²" % (area or 0))
        # 地理坐标系应返回 None
        crs_geo = QgsCoordinateReferenceSystem("EPSG:4326")
        check(geometry_area_m2(geom, crs_geo) is None,
              "地理坐标系面积返回 None（不猜）")

    print("\n=== 7. 跨 CRS 转换 ===")
    if drawn:
        geom, crs = drawn[0]
        gj2, crs2 = ensure_polygon_in_crs(
            geom, crs, QgsCoordinateReferenceSystem("EPSG:4326"))
        r2 = gj2["coordinates"][0]
        lon_ok = all(100 < p[0] < 110 for p in r2)
        lat_ok = all(20 < p[1] < 35 for p in r2)
        check(lon_ok and lat_ok,
              "换到 WGS84 后落在合理经纬度范围（%.4f,%.4f）"
              % (r2[0][0], r2[0][1]))

    # 清理（headless 退出用 os._exit，避免 Qt 析构段错误）
    canvas.setMapTool(None)
    tool.deactivate()
    canvas.hide()
    del tool, canvas

    failed = sum(1 for ok, _ in results if not ok)
    print("\n" + ("全部通过" if failed == 0 else "%d 项未通过" % failed)
          + "（共 %d 项）" % len(results))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(main())
