"""AOI 框选地图工具（开发文档 §6.5 / §10）。

只做「在地图画矩形 → 得到 AOI 几何」这一件事。跨 CRS 的加密变换在外部协调器里做，
插件只把画布 CRS 与 AOI 的 GeoJSON 一起传下去。

## 关于 QgsRubberBand 的构造参数

QGIS 3.x 的签名是 ``QgsRubberBand(mapCanvas, geometryType)``，第二个参数必须是
``QgsWkbTypes.GeometryType`` 枚举（如 ``PolygonGeometry``）。

早期版本接受 bool（True=POLYGON / False=LINE），但 3.44 起 bool 会直接抛
``TypeError: argument 2 has unexpected type 'bool'``（实战报错）。这里统一用枚举，
并在极老版本上回退到 bool。
"""
from __future__ import annotations

from qgis.core import (QgsCoordinateReferenceSystem, QgsCoordinateTransform,
                       QgsGeometry, QgsPointXY, QgsProject, QgsRectangle,
                       QgsWkbTypes)
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor

#: 画布上的 AOI 预览样式
AOI_RGB = (255, 140, 0)
AOI_LINE_ALPHA = 210
AOI_FILL_ALPHA = 45


def _polygon_geometry_type():
    """取 ``QgsWkbTypes.PolygonGeometry``；老版本返回 True（旧 API 的 POLYGON）。"""
    try:
        return QgsWkbTypes.GeometryType.PolygonGeometry
    except AttributeError:
        return getattr(QgsWkbTypes, "PolygonGeometry", True)


POLYGON_TYPE = _polygon_geometry_type()


class AoiRectangleTool(QgsMapTool):
    """在地图上拖出一个矩形作为 AOI。"""

    aoi_drawn = pyqtSignal(object, object)      # (QgsGeometry, QgsCoordinateReferenceSystem)

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self._start: QgsPointXY | None = None
        self._band = self._new_band()

    def _new_band(self) -> QgsRubberBand:
        """按当前 QGIS 版本的签名构造 rubber band。"""
        try:
            band = QgsRubberBand(self.canvas, POLYGON_TYPE)
        except TypeError:
            # 更老的版本只认 bool
            band = QgsRubberBand(self.canvas, True)
        band.setColor(QColor(*AOI_RGB, AOI_LINE_ALPHA))
        band.setWidth(2)
        band.setFillColor(QColor(*AOI_RGB, AOI_FILL_ALPHA))
        return band

    def _clear_band(self) -> None:
        """清空预览。``reset`` 在部分版本要求几何类型参数，这里都试一遍。"""
        try:
            self._band.reset(POLYGON_TYPE)
        except TypeError:
            try:
                self._band.reset(True)
            except TypeError:
                self._band.reset(QgsWkbTypes.PolygonGeometry)

    # ---- 事件
    def activate(self) -> None:
        super().activate()
        try:
            self.canvas.setCursor(Qt.CrossCursor)
        except Exception:                       # noqa: BLE001 - 纯装饰性，失败不影响框选
            pass

    def deactivate(self) -> None:
        # 注意：**不清空** rubber band。用户框选完通常会切回平移工具查看结果，
        # 此时若清空预览就看不到自己框的边界了（实战反馈）。清空交给 clear()。
        self._start = None
        try:
            if hasattr(self, "_band") and self._band is not None:
                self._band.show()
        except Exception:                       # noqa: BLE001 - 纯装饰性
            pass
        try:
            super().deactivate()
        except TypeError:                       # 某些版本签名不同
            pass

    def canvasPressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        self._start = self.toMapCoordinates(event.pos())
        self._clear_band()

    def canvasMoveEvent(self, event) -> None:
        if self._start is None:
            return
        cur = self.toMapCoordinates(event.pos())
        rect = QgsRectangle(self._start, cur)
        self._band.setToGeometry(QgsGeometry.fromRect(rect), None)

    def canvasReleaseEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self._start is None:
            return
        cur = self.toMapCoordinates(event.pos())
        # QgsRectangle(a, b) 已经按坐标排好序；normalize() 在 PyQGIS 里返回 None
        # （原地修改），所以**不能**写成 rect = QgsRectangle(...).normalize() ——
        # 那会把 rect 变成 None（实测 AttributeError: 'NoneType' has no attribute 'width'）。
        rect = QgsRectangle(self._start, cur)
        self._start = None
        if rect.width() <= 0 or rect.height() <= 0:
            return
        geom = QgsGeometry.fromRect(rect)
        # 框选完成后**保留**预览，让用户看到自己框了哪里。
        # 之前松手时不清空，但下一次 press 会清空，所以第一次框完是可见的；
        # 真正的问题是切走地图工具时 deactivate() 会 reset，导致框不见了。
        self._band.setToGeometry(geom, None)
        self.aoi_drawn.emit(geom, self.canvas.mapSettings().destinationCrs())

    def clear(self) -> None:
        """外部主动清除 AOI 预览（用户点「清除」时调用）。"""
        self._clear_band()


def ensure_polygon_in_crs(geometry: QgsGeometry, src_crs,
                          dst_crs: QgsCoordinateReferenceSystem) -> tuple[dict, str]:
    """把画布上的几何换到目标 CRS，返回 (GeoJSON, CRS 字符串)。

    这里只做顶点变换；真正需要加密的是**源 CRS → 影像 CRS** 这一步，
    由外部协调器按 AOI 边长自动加密（见 backend.geo.transform.aoi_transform_densified）。
    """
    if geometry is None or geometry.isEmpty():
        raise ValueError("AOI 为空")
    geom = QgsGeometry(geometry)
    if src_crs is not None and src_crs.isValid() and not src_crs == dst_crs:
        xform = QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance())
        geom.transform(xform)
    if geom.isMultipart():
        parts = geom.asMultiPolygon()
        if not parts:
            raise ValueError("AOI 不是面几何")
        geom = QgsGeometry.fromPolygonXY(parts[0])
    poly = geom.asPolygon()
    if not poly or len(poly[0]) < 3:
        raise ValueError("AOI 顶点不足")
    ring = [[float(p.x()), float(p.y())] for p in poly[0]]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}, dst_crs.authid()


def geometry_area_m2(geometry: QgsGeometry, crs) -> float | None:
    """AOI 面积（m²）；地理坐标系下返回 None（不猜）。"""
    if crs is None or not crs.isValid():
        return None
    if crs.isGeographic():
        return None
    return float(geometry.area())
