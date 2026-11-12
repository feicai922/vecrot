"""在 QGIS 运行时里跑一次**真模型**作业（检测 + 轮廓都走真后端）。

这与 qgis_job_check.py 的区别：那个用 jsonl/null 后端验证管线，这个用
deployment/manifest.json 的真实 detector/vector 环境，跑真实瓦片。

用法：
    python deployment/run_in_qgis.py deployment/qgis_real_job_check.py
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

RASTER = r"G:\贵州省数据集\images\test\rural_anshunshi-4_tile0000.tif"
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

        print("\n=== 1. QGIS 侧零推理依赖（隔离检查）===")
        for mod in ("torch", "transformers", "mmcv", "mmdet", "rasterio"):
            try:
                __import__(mod)
                check(False, f"{mod} 出现在 QGIS 环境（不应发生）")
            except ImportError:
                check(True, f"{mod} 不在 QGIS 环境")

        print("\n=== 2. 真实 manifest 就绪 ===")
        manifest_path = os.path.join(REPO, "deployment", "manifest.json")
        from vecrot_qgis.jobs.bridge import coordinator_python, read_manifest, resolve

        manifest = read_manifest(manifest_path)
        py = coordinator_python(manifest)
        check(os.path.isfile(py), f"协调器解释器：{py}")
        for role in ("detector", "vector"):
            exe = (manifest.get("environments") or {}).get(role, {}).get("python")
            check(bool(exe) and os.path.isfile(exe), f"{role} 环境：{exe}")
        base = resolve(manifest, (manifest["models"]["vectorizer"] or {}).get("base"))
        check(bool(base) and os.path.isdir(base), f"轮廓基座：{base}")

        print("\n=== 3. 构造真实作业（真模型，无 profile 覆盖）===")
        check(os.path.isfile(RASTER), f"测试瓦片：{RASTER}")
        if not os.path.isfile(RASTER):
            return 1

        # 用 QGIS 自己的栅格读取算 AOI（QGIS 里没有 rasterio，这正是隔离的体现）
        from qgis.core import QgsCoordinateReferenceSystem, QgsCoordinateTransform, \
            QgsProject, QgsRasterLayer, QgsRectangle

        layer = QgsRasterLayer(RASTER, "test_tile", "gdal")
        check(layer.isValid(), f"QGIS 能打开瓦片：{layer.width()}×{layer.height()} "
                               f"{layer.crs().authid()}")
        if not layer.isValid():
            return 1
        ext = layer.extent()
        dst = QgsCoordinateReferenceSystem("EPSG:4326")
        xform = QgsCoordinateTransform(layer.crs(), dst, QgsProject.instance())
        wgs = xform.transformBoundingBox(ext)
        aoi = {"type": "Polygon", "coordinates": [[
            [wgs.xMinimum(), wgs.yMinimum()], [wgs.xMaximum(), wgs.yMinimum()],
            [wgs.xMaximum(), wgs.yMaximum()], [wgs.xMinimum(), wgs.yMaximum()],
            [wgs.xMinimum(), wgs.yMinimum()]]]}
        print(f"     AOI（经纬度）：{wgs.xMinimum():.6f},{wgs.yMinimum():.6f} → "
              f"{wgs.xMaximum():.6f},{wgs.yMaximum():.6f}")
        # 注意：不写 detector_profile / vector_profile，交给 manifest 决定
        job = {
            "schema_version": "1.0",
            "job_id": "qgis_real_" + time.strftime("%H%M%S"),
            "input": {"raster_path": RASTER, "bands_rgb": [1, 2, 3],
                      "aoi_geojson": aoi, "aoi_crs": "EPSG:4326"},
            "inference": {"mode": "sequential", "score_threshold": 0.3,
                          "tile_size": 1024, "tile_overlap": 256,
                          "crop_expand": 1.45, "vector_batch": 4, "gpu_index": 0},
            "output": {"directory": os.path.join(REPO, "work", "qgis_real",
                                                 "out_" + time.strftime("%H%M%S")),
                       "crs_policy": "source",
                       "aoi_policy": "intersect_keep_complete"},
        }
        work_dir = os.path.join(REPO, "work", "qgis_real", job["job_id"])
        os.makedirs(work_dir, exist_ok=True)
        job_path = os.path.join(work_dir, "job.json")
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False, indent=2)
        out_dir = job["output"]["directory"]
        print(f"     job_id={job['job_id']}")
        print(f"     输出={out_dir}")

        print("\n=== 4. 用插件的 QProcess 桥启动（与 GUI 同一路径）===")
        from qgis.PyQt.QtCore import QEventLoop, QTimer

        from vecrot_qgis.jobs.bridge import EventBridge

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
        cancel_file = os.path.join(work_dir, "cancel.flag")
        if os.path.exists(cancel_file):
            os.remove(cancel_file)

        t0 = time.time()
        bridge.start(py, ["-u", "-m", "backend.cli", "run",
                          "--job", job_path, "--manifest", manifest_path,
                          "--work", os.path.join(REPO, "work", "qgis_real", "workroot"),
                          "--cancel-file", cancel_file],
                     cwd=REPO, cancel_file=cancel_file)
        QTimer.singleShot(900000, loop.quit)      # 15 分钟兜底（模型首次加载慢）
        loop.exec()
        elapsed = time.time() - t0

        check(done["code"] == 0, f"作业退出码 {done['code']}（0=成功），耗时 {elapsed:.0f}s")
        stages = [e["stage"] for e in events if e.get("event") == "stage_started"]
        check(stages == ["VALIDATING", "PREPARING", "DETECTING", "DEDUPLICATING",
                         "VECTORIZING", "EXPORTING"], f"阶段完整：{stages}")

        print("\n=== 5. 产物与真模型凭据 ===")
        for name in ("boxes.gpkg", "buildings.gpkg", "report.json", "manifest.json"):
            check(os.path.isfile(os.path.join(out_dir, name)), f"{name} 存在")

        rp = os.path.join(out_dir, "report.json")
        if os.path.isfile(rp):
            report = json.load(open(rp, encoding="utf-8"))
            mf = report.get("manifest") or {}
            det = mf.get("detector") or {}
            vec = mf.get("vectorizer") or {}
            check(mf.get("detector_profile") == "strip_s_gzda",
                  f"检测后端 = {mf.get('detector_profile')}（真模型，非 jsonl/null）")
            check(mf.get("vector_profile") == "vecrot",
                  f"轮廓后端 = {mf.get('vector_profile')}（真模型）")
            check((det.get("weight_loading") or {}).get("matched_all_structural") is True,
                  "检测权重结构全部匹配")
            check(vec.get("need_scale") is True, "轮廓 ScaleCond 已启用")
            counts = report["counts"]
            print(f"     计数：{counts}")
            check(counts["kept_boxes"] > 0, f"检出 {counts['kept_boxes']} 个框（>0）")
            check(counts["buildings_written"] > 0,
                  f"写出 {counts['buildings_written']} 个建筑轮廓")
            check(sum(counts["status"].values()) == counts["kept_boxes"], "计数守恒")

        if not all(ok for ok, _ in results):
            print("\n--- 协调器/worker 日志（尾部）---")
            for line in logs[-25:]:
                print("   ", line)

    finally:
        # 同 gui_flow：headless 下跳过 gc/exitQgis，避免对已 deleteLater 对象二次析构
        pass

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
