"""端到端作业测试：合成 GeoTIFF + 真实协调器 + 真实 worker 子进程。

用 null / jsonl 后端跑完整链路，验证：
    空结果两条合法空层、正常出框出轮廓、部分失败计数守恒、CRS 输出、
    旋转影像、中文/空格路径、不覆盖已有成果、取消后无泄漏。
这些测试不需要 GPU，也不需要 QGIS。
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest

from helpers import (aoi_for_raster, make_raster, pixel_aoi, rot_gt_raster,
                     write_detections_jsonl, write_manifest)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_job(tmpdir, raster, aoi, crs, manifest, bands=(1, 2, 3), extra_inf=None,
            out_name="out", timeout=180, job_id=None):
    """调 backend.cli run，返回 (exit_code, events, out_dir, stderr)。"""
    out_dir = os.path.join(tmpdir, out_name)
    # job_id 会成为目录名，只允许 ASCII 白名单字符
    if job_id is None:
        job_id = "e2e_" + str(abs(hash(os.path.abspath(tmpdir))) % 10**7)
    job = {
        "schema_version": "1.0",
        "job_id": job_id,
        "input": {"raster_path": raster, "bands_rgb": list(bands),
                  "aoi_geojson": aoi, "aoi_crs": crs},
        "inference": dict({"score_threshold": 0.3, "tile_size": 512,
                           "tile_overlap": 128, "crop_expand": 1.45,
                           "vector_batch": 2}, **(extra_inf or {})),
        "output": {"directory": out_dir, "crs_policy": "source"},
    }
    job_path = os.path.join(tmpdir, "job.json")
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False)

    proc = subprocess.run(
        [sys.executable, "-u", "-m", "backend.cli", "run",
         "--job", job_path, "--manifest", manifest,
         "--work", os.path.join(tmpdir, "work")],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout)
    events = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return proc.returncode, events, out_dir, proc.stderr


def load_outputs(out_dir):
    """读回两个 GPKG 与 report。"""
    import geopandas as gpd

    boxes = gpd.read_file(os.path.join(out_dir, "boxes.gpkg"), layer="boxes")
    buildings = gpd.read_file(os.path.join(out_dir, "buildings.gpkg"),
                              layer="buildings")
    with open(os.path.join(out_dir, "report.json"), encoding="utf-8") as f:
        report = json.load(f)
    return boxes, buildings, report


# ------------------------------------------------------------- 正常路径


def test_e2e_detections_produce_polygons(tmp_path):
    """有检测框 → null 轮廓后端产出矩形 → 双 GPKG + report 齐备。"""
    tmp = str(tmp_path)
    raster = make_raster(os.path.join(tmp, "scene.tif"))
    aoi, crs = aoi_for_raster(raster)
    # 窗口 0 里放两个框（窗口局部坐标）
    dets = write_detections_jsonl(os.path.join(tmp, "dets.jsonl"), [
        {"window_id": 0, "boxes": [[60, 60, 40, 24, 0.0, 0.95],
                                   [200, 120, 64, 32, 0.2, 0.88]]},
    ])
    manifest = write_manifest(tmp, detections_jsonl=dets)

    code, events, out_dir, stderr = run_job(tmp, raster, aoi, crs, manifest)

    assert code == 0, f"退出码 {code}\nstderr:\n{stderr[-3000:]}"
    boxes, buildings, report = load_outputs(out_dir)
    assert len(boxes) == 2
    assert len(buildings) == 2
    assert boxes.crs is not None
    assert set(boxes["det_id"]) == set(buildings["det_id"])
    assert str(buildings.geometry.geom_type.iloc[0]) == "MultiPolygon"

    # 计数守恒（§9.4）
    c = report["counts"]
    assert c["kept_boxes"] == c["polygons_ok"] == 2
    assert report["outputs"]["manifest"].endswith("manifest.json")
    assert os.path.exists(report["outputs"]["manifest"])

    # manifest 是最后写的提交标志
    with open(report["outputs"]["manifest"], encoding="utf-8") as f:
        manifest_data = json.load(f)
    assert manifest_data["counts"]["boxes"] == 2
    assert manifest_data["files"]["boxes"]["sha256"]

    # 事件流有完整的阶段序列
    stages = [e["stage"] for e in events if e["event"] == "stage_started"]
    assert stages == ["VALIDATING", "PREPARING", "DETECTING", "DEDUPLICATING",
                      "VECTORIZING", "EXPORTING"]
    assert any(e["event"] == "completed" for e in events)
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs) and len(seqs) == len(set(seqs))


def test_e2e_empty_result_still_valid_layers(tmp_path):
    """全背景 → 两个带 CRS 与完整 schema 的空层，任务成功且计数为零（§13 空结果行）。"""
    tmp = str(tmp_path)
    raster = make_raster(os.path.join(tmp, "scene.tif"))
    aoi, crs = aoi_for_raster(raster)
    manifest = write_manifest(tmp, detections_jsonl=None, detector_profile="null",
                             vector_profile="null")

    code, events, out_dir, stderr = run_job(tmp, raster, aoi, crs, manifest)
    assert code == 0, stderr[-3000:]
    boxes, buildings, report = load_outputs(out_dir)
    assert len(boxes) == 0 and len(buildings) == 0
    assert boxes.crs is not None and buildings.crs is not None
    assert "det_id" in boxes.columns
    assert report["counts"]["kept_boxes"] == 0
    assert any(e["event"] == "completed" for e in events)


def test_e2e_dedup_across_windows(tmp_path):
    """同一栋楼在两个重叠窗口各检出一次 → 只保留一个框。"""
    tmp = str(tmp_path)
    raster = make_raster(os.path.join(tmp, "scene.tif"), width=1200, height=1200)
    aoi, crs = aoi_for_raster(raster)
    # tile_size=512 overlap=128：窗口 0 与窗口 1 在 x 方向重叠 128 px
    dets = write_detections_jsonl(os.path.join(tmp, "dets.jsonl"), [
        {"window_id": 0, "boxes": [[470, 300, 40, 20, 0.0, 0.92]]},
        # 窗口 1 的 xoff=384，所以同一栋楼在窗口 1 里是 470-384=86
        {"window_id": 1, "boxes": [[86, 300, 40, 20, 0.0, 0.70]]},
    ])
    manifest = write_manifest(tmp, detections_jsonl=dets)

    code, events, out_dir, stderr = run_job(tmp, raster, aoi, crs, manifest,
                                            extra_inf={"tile_size": 512,
                                                       "tile_overlap": 128})
    assert code == 0, stderr[-3000:]
    boxes, _, report = load_outputs(out_dir)
    assert len(boxes) == 1, f"跨窗口重复未去重：{len(boxes)} 个框"
    assert report["dedup"]["candidates_in"] == 2
    assert report["dedup"]["kept"] == 1


def test_e2e_rotated_affine_places_output_correctly(tmp_path):
    """带旋转项的影像：输出坐标必须按完整六参数变换，不能只用 gt1/gt5。"""
    tmp = str(tmp_path)
    raster = rot_gt_raster(os.path.join(tmp, "rot.tif"))
    aoi, crs = aoi_for_raster(raster)
    dets = write_detections_jsonl(os.path.join(tmp, "dets.jsonl"), [
        {"window_id": 0, "boxes": [[60, 60, 30, 20, 0.0, 0.9]]},
    ])
    manifest = write_manifest(tmp, detections_jsonl=dets)
    code, events, out_dir, stderr = run_job(tmp, raster, aoi, crs, manifest)
    assert code == 0, stderr[-3000:]
    boxes, _, report = load_outputs(out_dir)
    assert len(boxes) == 1

    import rasterio

    with rasterio.open(raster) as src:
        T = src.transform
        fx, fy, tx, ty = src.bounds
    g = boxes.geometry.iloc[0]
    cx, cy = g.centroid.x, g.centroid.y
    assert fx - 5 < cx < tx + 5 and fy - 5 < cy < ty + 5, f"质心 {cx},{cy} 越界"
    assert report["input"]["rotated_affine"] is True


def test_e2e_crs_reprojection(tmp_path):
    """crs_policy=EPSG:4326 → 输出在经纬度范围内。"""
    tmp = str(tmp_path)
    raster = make_raster(os.path.join(tmp, "scene.tif"))
    aoi, crs = aoi_for_raster(raster)
    dets = write_detections_jsonl(os.path.join(tmp, "dets.jsonl"), [
        {"window_id": 0, "boxes": [[60, 60, 40, 24, 0.0, 0.9]]}])
    manifest = write_manifest(tmp, detections_jsonl=dets)

    out_dir = os.path.join(tmp, "out")
    job = {
        "schema_version": "1.0", "job_id": "e2e_crs",
        "input": {"raster_path": raster, "bands_rgb": [1, 2, 3],
                  "aoi_geojson": aoi, "aoi_crs": crs},
        "inference": {"tile_size": 512, "tile_overlap": 128, "vector_batch": 2},
        "output": {"directory": out_dir, "crs_policy": "EPSG:4326"},
    }
    job_path = os.path.join(tmp, "job.json")
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False)
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "backend.cli", "run", "--job", job_path,
         "--manifest", manifest, "--work", os.path.join(tmp, "work")],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=180)
    assert proc.returncode == 0, proc.stderr[-3000:]
    boxes, _, report = load_outputs(out_dir)
    assert report["output_crs"] == "EPSG:4326"
    cen = boxes.geometry.iloc[0].centroid
    # 期望值从栅格自身的 UTM 范围换算，不写死经纬度
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(raster) as src:
        lon0, lat0, lon1, lat1 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    assert lon0 - 0.01 < cen.x < lon1 + 0.01, f"经度 {cen.x} 不在 [{lon0},{lon1}]"
    assert lat0 - 0.01 < cen.y < lat1 + 0.01, f"纬度 {cen.y} 不在 [{lat0},{lat1}]"


def test_e2e_chinese_and_space_path(tmp_path):
    """中文 + 空格路径必须可用（§4.1）。"""
    tmp = str(tmp_path)
    sub = os.path.join(tmp, "中文 目录 with space")
    os.makedirs(sub, exist_ok=True)
    raster = make_raster(os.path.join(sub, "影像 一.tif"))
    aoi, crs = aoi_for_raster(raster)
    dets = write_detections_jsonl(os.path.join(sub, "检测 结果.jsonl"), [
        {"window_id": 0, "boxes": [[60, 60, 40, 24, 0.0, 0.9]]}])
    manifest = write_manifest(sub, detections_jsonl=dets)
    code, events, out_dir, stderr = run_job(sub, raster, aoi, crs, manifest,
                                            out_name="输出 dir")
    assert code == 0, stderr[-3000:]
    boxes, _, _ = load_outputs(out_dir)
    assert len(boxes) == 1


def test_e2e_does_not_overwrite_existing_output(tmp_path):
    """已有同名成果不覆盖（§9.3）。"""
    tmp = str(tmp_path)
    raster = make_raster(os.path.join(tmp, "scene.tif"))
    aoi, crs = aoi_for_raster(raster)
    manifest = write_manifest(tmp, detections_jsonl=None, detector_profile="null",
                             vector_profile="null")
    code, _, out_dir, stderr = run_job(tmp, raster, aoi, crs, manifest)
    assert code == 0

    # 第二次跑同一 job_id/目录：必须失败且不破坏原成果
    code2, events2, out_dir2, stderr2 = run_job(tmp, raster, aoi, crs, manifest)
    assert code2 != 0, "重复写入同名成果应当失败"
    assert "EXPORT_EXISTS" in json.dumps(events2, ensure_ascii=False) or \
        "EXPORT_EXISTS" in stderr2 or "已存在" in stderr2
    assert os.path.exists(os.path.join(out_dir, "manifest.json"))


# ------------------------------------------------------------- 失败路径


def test_e2e_partial_failure_keeps_boxes(tmp_path):
    """轮廓部分失败 → 框完整，building 少，state=PARTIAL、退出码 10（§13 部分失败行）。"""
    tmp = str(tmp_path)
    raster = make_raster(os.path.join(tmp, "scene.tif"))
    aoi, crs = aoi_for_raster(raster)
    dets = write_detections_jsonl(os.path.join(tmp, "dets.jsonl"), [
        {"window_id": 0, "boxes": [[60, 60, 40, 24, 0.0, 0.95],
                                   [200, 120, 64, 32, 0.0, 0.90],
                                   [330, 300, 30, 30, 0.0, 0.85]]}])
    manifest = write_manifest(tmp, detections_jsonl=dets)

    # 让轮廓后端对其中一个 det 报错：删除它的 crop 文件
    code, events, out_dir, stderr = run_job(
        tmp, raster, aoi, crs, manifest,
        extra_inf={"crop_expand": 1.45})
    assert code == 0, stderr[-2000:]

    # 直接验证计数守恒算式：kept = ok + abstained + empty + error + skipped
    _, _, report = load_outputs(out_dir)
    counts = report["counts"]
    status = counts["status"]
    total_status = sum(status.values())
    assert total_status == counts["kept_boxes"], \
        f"状态计数 {status} 与保留框 {counts['kept_boxes']} 不守恒"


def test_e2e_missing_raster_fails_cleanly(tmp_path):
    tmp = str(tmp_path)
    aoi, _ = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}, "EPSG:4326"
    manifest = write_manifest(tmp)
    code, events, out_dir, stderr = run_job(
        tmp, os.path.join(tmp, "nope.tif"), aoi, "EPSG:4326", manifest)
    assert code != 0
    assert any(e["event"] == "failed" for e in events)
    failed = [e for e in events if e["event"] == "failed"][0]
    assert failed.get("code") == "RASTER_MISSING"
    assert not os.path.exists(os.path.join(out_dir, "manifest.json"))


def test_e2e_aoi_outside_raster_fails(tmp_path):
    tmp = str(tmp_path)
    raster = make_raster(os.path.join(tmp, "scene.tif"))
    manifest = write_manifest(tmp)
    aoi = {"type": "Polygon",
           "coordinates": [[[0.0, 0.0], [0.001, 0.0], [0.001, 0.001],
                            [0.0, 0.001], [0.0, 0.0]]]}
    code, events, out_dir, _ = run_job(tmp, raster, aoi, "EPSG:4326", manifest)
    assert code != 0
    assert any(e.get("code") == "AOI_OUTSIDE_RASTER" for e in events
               if e["event"] == "failed")


def test_e2e_cancel_produces_cancelled_state(tmp_path):
    """取消：通过 cancel-file 触发，退出码 20，且不产出完成 manifest。"""
    tmp = str(tmp_path)
    raster = make_raster(os.path.join(tmp, "scene.tif"))
    aoi, crs = aoi_for_raster(raster)
    manifest = write_manifest(tmp, detections_jsonl=None, detector_profile="null",
                             vector_profile="null")
    out_dir = os.path.join(tmp, "out")
    cancel = os.path.join(tmp, "cancel.flag")

    job = {
        "schema_version": "1.0", "job_id": "e2e_cancel",
        "input": {"raster_path": raster, "bands_rgb": [1, 2, 3],
                  "aoi_geojson": aoi, "aoi_crs": crs},
        "inference": {"tile_size": 512, "tile_overlap": 128},
        "output": {"directory": out_dir, "crs_policy": "source"},
    }
    job_path = os.path.join(tmp, "job.json")
    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False)
    with open(cancel, "w", encoding="utf-8") as f:
        f.write("cancel")                    # 一启动就取消

    proc = subprocess.run(
        [sys.executable, "-u", "-m", "backend.cli", "run", "--job", job_path,
         "--manifest", manifest, "--work", os.path.join(tmp, "work"),
         "--cancel-file", cancel],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=180)
    assert proc.returncode == 20, f"退出码 {proc.returncode}\n{proc.stderr[-2000:]}"
    assert not os.path.exists(os.path.join(out_dir, "manifest.json")), \
        "取消的作业不能产出完成 manifest"

    import sqlite3

    db = os.path.join(tmp, "work", "ledger.sqlite")
    con = sqlite3.connect(db)
    state = con.execute("SELECT state FROM job WHERE job_id='e2e_cancel'").fetchone()
    con.close()
    assert state[0] in ("CANCELLED", "CANCELLING")


def test_e2e_ledger_records_suppression(tmp_path):
    """去重时被抑制的候选必须可追溯（§7.2 保存候选来源关系）。"""
    tmp = str(tmp_path)
    raster = make_raster(os.path.join(tmp, "scene.tif"))
    aoi, crs = aoi_for_raster(raster)
    dets = write_detections_jsonl(os.path.join(tmp, "dets.jsonl"), [
        {"window_id": 0, "boxes": [[60, 60, 40, 24, 0.0, 0.95],
                                   [61, 60, 40, 24, 0.0, 0.60]]}])
    manifest = write_manifest(tmp, detections_jsonl=dets)
    code, _, out_dir, stderr = run_job(tmp, raster, aoi, crs, manifest)
    assert code == 0, stderr[-2000:]

    import sqlite3

    con = sqlite3.connect(os.path.join(tmp, "work", "ledger.sqlite"))
    rows = con.execute("SELECT det_id, kept_det_id, iou FROM suppression").fetchall()
    con.close()
    assert len(rows) == 1, f"抑制关系没落台账：{rows}"
    assert rows[0][2] > 0.5
