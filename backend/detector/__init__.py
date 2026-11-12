"""检测器适配层（开发文档 §7.1）。

核心约束：检测实现依赖 Strip-R-CNN（MMRotate 2.x），本机未安装也无法在 QGIS 环境里安装。
因此这里定义**后端接口**并随仓库提供一个显式的 ``null`` 后端（无检测结果，用于闭环
验证空结果路径）与一个 ``jsonl`` 后端（读取已有检测结果，用于无 GPU 的开发验证）。

真正的 Strip-R-CNN 后端需要单独环境，见 ``detector/strip_rcnn_backend.py``。
"""
