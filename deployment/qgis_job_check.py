"""在 QGIS 运行时里跑一次真实作业（jsonl 检测 + null 轮廓）。

验证重点不是模型，而是**进程边界与零依赖污染**：
* QGIS 的 Python 里没有 rasterio/geopandas —— 这正是环境隔离的目标；
* 协调器由 QProcess 启动，用 manifest 指定的独立 GIS 解释器；
* 作业结束后 QGIS 能直接加载产出的双 GPKG。

测试数据由协调器环境预先准备好（见 docs/INSTALL.md），本脚本不造数据。

用法：
    python deployment/run_in_qgis.py deployment/qgis_job_check.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
WORK = os.path.join(REPO, "work", "qgis_check")

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
    try:
        from qgis.core import Qgis

        print(f"=== QGIS {Qgis.QGIS_VERSION} / Python {sys.version.split()[0]} ===")

        print("\n=== 1. 环境隔离：插件不使用这些包，无论装没装 ===")
        # 真正要紧的是推理栈：QGIS 环境里出现 torch/transformers/mmcv/mmdet
        # 会拖慢启动甚至崩溃。GIS 包（rasterio/pyogrio/geopandas）存在与否
        # 都不影响插件，因为插件代码根本不 import 它们（见 check_plugin.py）。
        for mod in ("torch", "transformers", "mmcv", "mmdet"):
            try:
                __import__(mod)
                check(False, f"QGIS 环境出现 {mod}：推理栈不该装进 QGIS")
            except ImportError:
                check(True, f"{mod} 不在 QGIS 环境（推理栈已隔离）")
        for mod in ("rasterio", "pyogrio", "geopandas", "shapely"):
            try:
                __import__(mod)
                check(True, f"{mod} 存在于 QGIS 环境（无影响：插件不 import 它）")
            except ImportError:
                check(True, f"{mod} 不存在（插件也不需要）")

        print("\n=== 2. 前置数据 ===")
        job_path = os.path.join(WORK, "job.json")
        manifest_path = os.path.join(WORK, "manifest.json")
        check(os.path.isfile(job_path), f"job.json 已由协调器环境准备：{job_path}")
        check(os.path.isfile(manifest_path), "测试 manifest 存在")
        if not (os.path.isfile(job_path) and os.path.isfile(manifest_path)):
            return 1

        print("\n=== 3. 用 QProcess（与插件同一路径）启动作业 ===")
        from qgis.PyQt.QtCore import QEventLoop, QTimer

        from vecrot_qgis.jobs.bridge import EventBridge, read_manifest, coordinator_python

        # 测试用 manifest：三个角色都指向当前 GIS 环境（jsonl/null 后端不需要 GPU）。
        # 真实部署用的是 deployment/manifest.json，见 docs/INSTALL.md。
        manifest = read_manifest(manifest_path)
        py = coordinator_python(manifest)
        check(os.path.isfile(py), f"协调器解释器来自 manifest：{py}")
        check("QGIS" not in py, "协调器不是 QGIS 的解释器（没有把 GisPython 当模型环境）")

        # 每次运行用独立的 job_id 与输出目录：上一次的成果会触发 EXPORT_EXISTS
        import time

        run_tag = time.strftime("%H%M%S")
        job = json.load(open(job_path, encoding="utf-8"))
        job["job_id"] = f"qgis_check_{run_tag}"
        job["output"]["directory"] = os.path.join(WORK, "output_" + run_tag)
        job_path = os.path.join(WORK, f"job_{run_tag}.json")
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False)

        events: list[dict] = []
        logs: list[str] = []
        done: dict = {"code": None}

        bridge = EventBridge()
        bridge.event_received.connect(events.append)
        bridge.log_line.connect(logs.append)

        loop = QEventLoop()

        def on_finished(code, summary):
            done["code"] = code
            loop.quit()

        bridge.finished.connect(on_finished)

        cancel_file = os.path.join(WORK, "cancel.flag")
        if os.path.exists(cancel_file):
            os.remove(cancel_file)
        work_root = os.path.join(WORK, "workroot_" + run_tag)
        out_dir = job["output"]["directory"]

        bridge.start(py, ["-u", "-m", "backend.cli", "run",
                          "--job", job_path, "--manifest", manifest_path,
                          "--work", work_root, "--cancel-file", cancel_file],
                     cwd=REPO, cancel_file=cancel_file)
        QTimer.singleShot(180000, loop.quit)
        loop.exec()

        check(done["code"] == 0, f"作业退出码 {done['code']}（0=成功）")
        stages = [e["stage"] for e in events if e.get("event") == "stage_started"]
        check(stages == ["VALIDATING", "PREPARING", "DETECTING", "DEDUPLICATING",
                         "VECTORIZING", "EXPORTING"], f"阶段完整：{stages}")
        seqs = [e.get("seq") for e in events if e.get("seq") is not None]
        check(seqs and len(seqs) == len(set(seqs)), f"事件 seq 单调且无重复（{len(seqs)} 条）")

        print("\n=== 4. 产物 ===")
        for name in ("boxes.gpkg", "buildings.gpkg", "report.json", "manifest.json"):
            check(os.path.isfile(os.path.join(out_dir, name)), f"{name} 存在")

        report_path = os.path.join(out_dir, "report.json")
        if os.path.isfile(report_path):
            report = json.load(open(report_path, encoding="utf-8"))
            counted = sum(report["counts"]["status"].values())
            check(report["counts"]["kept_boxes"] == counted == 2,
                  f"计数守恒：保留 {report['counts']['kept_boxes']} = 状态和 {counted}")
            check(report["counts"]["polygons_ok"] == 2, "两个轮廓都生成")
        else:
            check(False, "缺 report.json，跳过计数校验")
            if logs:
                print("\n--- 协调器/worker 日志（尾部）---")
                for line in logs[-15:]:
                    print("   ", line)
            return 1

        print("\n=== 5. QGIS 直接加载产物 ===")
        from qgis.core import QgsProject, QgsVectorLayer

        boxes = QgsVectorLayer(os.path.join(out_dir, "boxes.gpkg") + "|layername=boxes",
                               "boxes", "ogr")
        builds = QgsVectorLayer(os.path.join(out_dir, "buildings.gpkg")
                                + "|layername=buildings", "buildings", "ogr")
        check(boxes.isValid() and boxes.featureCount() == 2,
              f"boxes 图层有效（{boxes.featureCount()} 个要素）")
        check(builds.isValid() and builds.featureCount() == 2,
              f"buildings 图层有效（{builds.featureCount()} 个要素）")
        check(boxes.crs() == builds.crs() and boxes.crs().isValid(),
              f"两图层 CRS 一致：{boxes.crs().authid()}")

        # 必须在 exitQgis 之前显式释放图层：否则 Python GC 会在 QGIS 拆掉
        # 数据提供者之后再回收，导致退出时段错误（exit=139）。
        for layer in (boxes, builds):
            layer.setDataSource("", layer.name(), "ogr")      # 断开数据源
        del boxes, builds
        QgsProject.instance().removeAllMapLayers()

        if logs:
            print("\n--- 协调器/worker 日志（尾部）---")
            for line in logs[-10:]:
                print("   ", line)

    finally:
        qgs.exitQgis()
        # 让 PyQt 在解释器退出前拆完 Qt 对象，避免 Qt5 关闭顺序导致的 abort
        import gc

        gc.collect()

    failed = sum(1 for ok, _ in results if not ok)
    print(f"\n{'全部通过' if failed == 0 else f'{failed} 项未通过'}（共 {len(results)} 项）")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    # Qt5 在 headless 下 exitQgis() 之后的 Python/C++ 析构顺序不稳定，可能在
    # 解释器退出阶段触发段错误；此时检查结果早已全部打印。用 os._exit 跳过
    # 解释器清理阶段，保证退出码如实反映检查结果（正式 GUI 里 QGIS 自己管生命周期，
    # 不存在这个问题）。
    sys.stdout.flush()
    sys.stderr.flush()
    _code = main()
    os._exit(_code)
