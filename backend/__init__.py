"""Vecrot × QGIS 后端（QGIS 之外）。

包内分工（对应开发文档 §5）：
    contracts/    JSON Schema 与事件协议
    geo/          栅格窗口、AOI、坐标变换、去重、GeoPackage 导出
    detector/     检测 worker（可插拔后端）
    vectorizer/   轮廓生成 worker（可插拔后端）
    coordinator/  状态机、作业台账、进程调度

入口：``python -m backend.cli``
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
SCHEMA_VERSION = "1.0"
