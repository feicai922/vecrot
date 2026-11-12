"""成果加载与预览样式（开发文档 §10）。

只在主线程操作图层。加载的是最终 GPKG（manifest 已验证），预览不写最终文件。
"""
from __future__ import annotations

import json
import os

from qgis.core import (QgsApplication, QgsClassificationQuantile, QgsFillSymbol,
                       QgsGradientColorRamp, QgsLayerTreeGroup, QgsLineSymbol,
                       QgsMarkerSymbol, QgsProject, QgsSingleSymbolRenderer,
                       QgsSymbolLayerUtils, QgsVectorLayer, QgsCategorizedSymbolRenderer,
                       QgsRendererCategory, QgsGraduatedSymbolRenderer)
from qgis.PyQt.QtGui import QColor


def load_results(project: QgsProject, out_dir: str) -> QgsLayerTreeGroup:
    """把两个 GPKG 加载为以 job_id 命名的图层组，并配好样式。"""
    manifest_path = os.path.join(out_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"成果未完成（缺 manifest.json）：{out_dir}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    job_id = manifest.get("job_id", os.path.basename(out_dir))

    root = project.layerTreeRoot()
    group = root.insertGroup(0, f"Vecrot {job_id}")
    group.setExpanded(True)

    boxes_path = os.path.join(out_dir, "boxes.gpkg")
    buildings_path = os.path.join(out_dir, "buildings.gpkg")

    # 顺序很重要：先加的建筑轮廓在下层，检测框在上层。
    # 轮廓是「填充 + 细边框」，框是「全透明填充 + 分级彩色边框」——
    # 轮廓放上层时，两者几乎重合的边界会互相压盖，看起来像"轮廓没显示"。
    # 实测（600×600 离屏渲染）确认：轮廓图层确实有内容，只是被框线挡住。
    if os.path.isfile(buildings_path):
        buildings = QgsVectorLayer(f"{buildings_path}|layername=buildings",
                                   f"{job_id} 建筑轮廓", "ogr")
        if buildings.isValid():
            buildings.setRenderer(_buildings_renderer())
            project.addMapLayer(buildings, False)
            group.addLayer(buildings)
    if os.path.isfile(boxes_path):
        boxes = QgsVectorLayer(f"{boxes_path}|layername=boxes", f"{job_id} 检测框", "ogr")
        if boxes.isValid():
            boxes.setRenderer(_boxes_renderer(boxes))
            project.addMapLayer(boxes, False)
            group.addLayer(boxes)
    return group


def _boxes_renderer(layer: QgsVectorLayer):
    """检测框：透明填充 + 按 det_score 分级线色（§10）。

    历史坑（都已实测）：
    * 分级类在 ``ranges()`` 不是 ``categories()``（那是 QgsCategorizedSymbolRenderer）；
    * ``renderer.ranges()`` 返回的是**副本列表**，对元素 ``setSymbol`` 不会写回
      renderer —— 必须用 ``updateRangeSymbol(i, symbol)``；
    * Qt5 下 ``QgsGradientColorRamp.color()`` 返回 int（ARGB），Qt6 返回 QColor。
    """
    if layer.fields().indexOf("det_score") < 0:
        return _fallback_line_renderer(layer)
    values = {f["det_score"] for f in layer.getFeatures() if f["det_score"] is not None}
    if len(values) < 2:
        return _fallback_line_renderer(layer)

    ramp = QgsGradientColorRamp(QColor(214, 96, 77), QColor(50, 130, 190))
    renderer = QgsGraduatedSymbolRenderer("", [])
    renderer.setClassAttribute("det_score")
    renderer.setSourceColorRamp(ramp)
    renderer.setClassificationMethod(QgsClassificationQuantile())

    # 顺序必须是这样：先把 renderer 挂到图层，再 updateClasses。
    # 反过来（先 updateClasses 再 setRenderer）会让 ranges 变空 —— 实测
    # 「挂载后 ranges: 0 / updateClasses 后 ranges: 5」。
    layer.setRenderer(renderer)
    live = layer.renderer()
    live.updateClasses(layer, min(5, len(values)))

    n = max(len(live.ranges()), 1)
    for i in range(n):
        raw = ramp.color(i / max(n - 1, 1))
        color = raw if isinstance(raw, QColor) else QColor.fromRgb(raw)
        if not color.isValid():
            continue
        # ranges() 返回副本，必须用 updateRangeSymbol 写回
        live.updateRangeSymbol(i, _outline_symbol(color))

    return live


def _outline_symbol(color: QColor):
    sym = QgsFillSymbol.createSimple({
        "color": "0,0,0,0",
        "outline_color": f"{color.red()},{color.green()},{color.blue()},{color.alpha()}",
        "outline_width": "0.5",
    })
    return sym


def _fallback_line_renderer(layer: QgsVectorLayer):
    symbol = QgsFillSymbol.createSimple({
        "color": "255,140,0,0",
        "outline_color": "230,120,20,255",
        "outline_width": "0.4",
    })
    renderer = QgsSingleSymbolRenderer(symbol)
    layer.setRenderer(renderer)
    return renderer


#: 建筑轮廓默认样式：红色实线边 + 淡红半透明填充。
#: 红色与检测框的冷色渐变（红→蓝按分值）区分明显，且红边在遥感影像上更醒目。
BUILDING_OUTLINE_RGB = (220, 40, 40)
#: 轮廓填充不透明度。用户要求"轮廓用掩码式（多边形）展示"，
#: 所以这里给较高的实心度，让它读起来是"建筑面"而不是"框"。
BUILDING_FILL_ALPHA = 165


def _buildings_renderer():
    """建筑轮廓：实心红色多边形（掩码式展示）。

    与检测框的分工（按用户要求）：
    * **检测框**：只画线，不填充（`_outline_symbol`，fill alpha=0）
    * **建筑轮廓**：实心填充 + 同色边，读作"建筑覆盖范围"

    两者叠在一起时，红色实心面 = 模型给出的建筑，外面的彩线 = 检测框。
    """
    r, g, b = BUILDING_OUTLINE_RGB
    symbol = QgsFillSymbol.createSimple({
        "color": "{},{},{},{}".format(r, g, b, BUILDING_FILL_ALPHA),
        "outline_color": "{},{},{},255".format(r, g, b),
        "outline_width": "0.6",
        "outline_style": "solid",
    })
    return QgsSingleSymbolRenderer(symbol)


def open_report(out_dir: str) -> str | None:
    """用系统默认程序打开 report.json。"""
    from qgis.PyQt.QtCore import QUrl
    from qgis.PyQt.QtGui import QDesktopServices

    path = os.path.join(out_dir, "report.json")
    if not os.path.isfile(path):
        return None
    QDesktopServices.openUrl(QUrl.fromLocalFile(path))
    return path


# --------------------------------------------------------------- 增量追加


#: 追加时不该写入的字段（由 QGIS/OGR 自增或属于源作业）
_SKIP_FIELDS = {"fid", "ogc_fid"}


def append_gpkg(target_gpkg: str, source_gpkg: str, layer: str,
                id_prefix: str = "") -> int:
    """把 source 图层追加到 target 的同名图层，返回新增要素数。

    为什么不用 pyogrio/geopandas：QGIS 环境里没有它们（也不该装）。
    这里用 QGIS 的数据提供者逐要素写入。

    实测过的两个坑：
    1. ``QgsVectorFileWriter + AppendToLayerAddFields`` 传 ``target.fields()``
       会尝试 ``ADD COLUMN fid`` → ``duplicate column name: fid``。
    2. 用 provider 的 ``addFeatures`` 时 ``fid`` 必须显式置 NULL 让 GPKG
       主键自增；否则 ``UNIQUE constraint failed: boxes.fid``。
    """
    import os

    from qgis.core import QgsFeature, QgsVectorLayer

    if not os.path.isfile(target_gpkg):
        raise FileNotFoundError(f"目标成果不存在：{target_gpkg}")
    if not os.path.isfile(source_gpkg):
        return 0

    target = QgsVectorLayer(f"{target_gpkg}|layername={layer}", "target", "ogr")
    source = QgsVectorLayer(f"{source_gpkg}|layername={layer}", "source", "ogr")
    if not target.isValid():
        raise RuntimeError(f"无法打开目标图层：{target_gpkg}|{layer}")
    if not source.isValid():
        raise RuntimeError(f"无法打开源图层：{source_gpkg}|{layer}")

    tgt_names = [f.name() for f in target.fields()]
    src_names = source.fields().names()
    keep = [(i, n) for i, n in enumerate(tgt_names)
            if n in src_names and n.lower() not in _SKIP_FIELDS]
    if not keep:
        raise RuntimeError(f"{layer}：源与目标的字段没有交集，无法追加")

    provider = target.dataProvider()
    rename = ("det_id", "job_id")
    feats = []
    for feat in source.getFeatures():
        if not feat.hasGeometry():
            continue
        nf = QgsFeature(target.fields())
        nf.setGeometry(feat.geometry())
        attrs = [None] * len(tgt_names)
        for i, name in keep:
            if name.lower() in ("fid", "ogc_fid"):
                attrs[i] = None                 # 主键交给 GPKG 自增
                continue
            v = feat[name]
            if name in rename and v is not None:
                v = f"{id_prefix}{v}"
            attrs[i] = v
        nf.setAttributes(attrs)
        feats.append(nf)

    if not feats:
        return 0
    ok, added = provider.addFeatures(feats)
    if not ok:
        raise RuntimeError(f"追加写入被驱动拒绝（{len(feats)} 个要素）")
    count = len(added) if isinstance(added, list) else int(added or 0)
    target.updateExtents()
    # 断开数据源，确保关闭文件句柄（后续 refresh 需要重新打开）
    target.setDataSource("", target.name(), "ogr")
    source.setDataSource("", source.name(), "ogr")
    return count


def refresh_layers_from(project, out_dir: str) -> int:
    """重新读取成果，刷新（或新建）项目里的对应图层，返回总要素数。

    增量追加写的是磁盘上的 GPKG，已加载的图层不会自动看到新要素，
    必须重建数据源。这里保留原图层名与渲染样式。
    """
    import os

    from qgis.core import QgsVectorLayer

    manifest_path = os.path.join(out_dir, "manifest.json")
    job_id = os.path.basename(os.path.normpath(out_dir))
    if os.path.isfile(manifest_path):
        try:
            import json

            with open(manifest_path, encoding="utf-8") as f:
                job_id = json.load(f).get("job_id", job_id)
        except (OSError, ValueError):
            pass

    specs = [
        (os.path.join(out_dir, "boxes.gpkg"), "boxes", "检测框"),
        (os.path.join(out_dir, "buildings.gpkg"), "buildings", "建筑轮廓"),
    ]
    total = 0
    for gpkg, layer, label in specs:
        if not os.path.isfile(gpkg):
            continue
        name = f"{job_id} {label}"
        existing = [l for l in project.mapLayers().values()
                    if l.name() == name and isinstance(l, QgsVectorLayer)]
        if existing:
            only = existing[0]
            style = only.renderer().clone() if only.renderer() else None
            only.setDataSource(f"{gpkg}|layername={layer}", name, "ogr")
            only.triggerRepaint()
            if style is not None:
                only.setRenderer(style)
            total += only.featureCount()
        else:
            layer_obj = QgsVectorLayer(f"{gpkg}|layername={layer}", name, "ogr")
            if not layer_obj.isValid():
                continue
            if layer == "boxes":
                layer_obj.setRenderer(_boxes_renderer(layer_obj))
            else:
                layer_obj.setRenderer(_buildings_renderer())
            project.addMapLayer(layer_obj, False)
            # 找到同名图层组，没有就建
            root = project.layerTreeRoot()
            group = root.findGroup(f"Vecrot {job_id}")
            if group is None:
                group = root.insertGroup(0, f"Vecrot {job_id}")
                group.setExpanded(True)
            group.addLayer(layer_obj)
            total += layer_obj.featureCount()
    return total
