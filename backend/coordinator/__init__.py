"""作业协调器：状态机、台账、进程调度（开发文档 §4 / §8）。

协调器是**唯一**写台账的进程；worker 只写自己的结果分片。
进程模型：
    QGIS ──QProcess──> coordinator ──Popen──> detector worker
                                   └────────> vector worker

单卡默认「检测进程退出后再启动轮廓进程」，确保显存真的释放（§4.2）。
"""
