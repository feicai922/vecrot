"""作业相关：QProcess 桥、job.json 构造。

注意：``bridge`` 依赖 PyQGIS，只在 QGIS 宿主内导入；``jobfile`` 是纯逻辑，
可在普通 Python 下测试。因此这里不做顶层 re-export，避免 `import vecrot_qgis.jobs`
就要求 QGIS 存在。
"""

__all__ = ["build_job_payload", "write_job_file", "jobfile"]
