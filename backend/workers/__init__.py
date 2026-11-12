"""worker 入口：每类 worker 一个独立进程、独立环境、独立 GPU 生命周期。

设计（开发文档 §4）：
* 检测 worker 与轮廓 worker **不共享进程**：单卡默认检测进程退出释放显存后再启动轮廓进程。
* worker 之间只通过本地文件通信，不传 Python 对象或张量。
* 协议走 stdout；日志走 stderr。

启动方式一律 ``<python.exe> -m backend.workers.<name> --request <json>``，
or 由协调器按 manifest 决定解释器。
"""
