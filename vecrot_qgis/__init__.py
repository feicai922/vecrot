"""QGIS 插件入口。只依赖 PyQGIS / qgis.PyQt / 标准库。"""


def classFactory(iface):                     # noqa: N802 - QGIS 约定名
    from .plugin import VecrotPlugin

    return VecrotPlugin(iface)
