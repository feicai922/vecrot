"""插件侧纯逻辑测试：job.json 构造 + 事件编解码一致性。

不 import QGIS —— 这些模块刻意不依赖 PyQGIS，可以在普通 Python 下验证。
重点防的是「插件与后端各写一份协议解析，逐渐漂移」这个风险。
"""
from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import plugin_parent, plugin_dir   # noqa: E402

PLUGIN = plugin_parent()
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if PLUGIN not in sys.path:
    sys.path.insert(0, PLUGIN)


def test_plugin_jobfile_builds_valid_spec(tmp_path):
    """插件构造的 job.json 必须能被后端解析（同一份契约）。"""
    from backend.contracts.jobspec import parse_job_spec
    from vecrot_qgis.jobs.jobfile import build_job_payload

    raster = os.path.join(str(tmp_path), "scene.tif")
    with open(raster, "wb") as f:
        f.write(b"stub")                     # 只验证路径存在性检查

    payload = build_job_payload(
        raster_path=raster,
        aoi_geojson={"type": "Polygon",
                     "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        aoi_crs="EPSG:4326",
        output_dir=os.path.join(str(tmp_path), "out"),
        score_threshold=0.35, vector_batch=8,
    )
    spec = parse_job_spec(payload)           # 不抛异常即契约一致
    assert spec.job_id.startswith("vecrot_")
    assert spec.inference.score_threshold == 0.35
    assert spec.inference.detector_profile is None   # 交给 manifest 决定


def test_plugin_jobfile_rejects_missing_aoi(tmp_path):
    from vecrot_qgis.jobs.jobfile import JobSpecProblem, build_job_payload

    raster = os.path.join(str(tmp_path), "scene.tif")
    open(raster, "wb").close()
    with pytest.raises(JobSpecProblem) as ei:
        build_job_payload(raster_path=raster, aoi_geojson=None, aoi_crs="EPSG:4326",
                          output_dir=os.path.join(str(tmp_path), "out"))
    assert ei.value.code == "AOI_MISSING"


def test_plugin_jobfile_rejects_relative_raster(tmp_path):
    from vecrot_qgis.jobs.jobfile import JobSpecProblem, build_job_payload

    with pytest.raises(JobSpecProblem):
        build_job_payload(raster_path="scene.tif",
                          aoi_geojson={"type": "Polygon",
                                       "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                          aoi_crs="EPSG:4326",
                          output_dir=os.path.join(str(tmp_path), "out"))


def test_plugin_jobfile_rejects_bad_crs_policy(tmp_path):
    from vecrot_qgis.jobs.jobfile import JobSpecProblem, build_job_payload

    raster = os.path.join(str(tmp_path), "scene.tif")
    open(raster, "wb").close()
    with pytest.raises(JobSpecProblem):
        build_job_payload(raster_path=raster,
                          aoi_geojson={"type": "Polygon",
                                       "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                          aoi_crs="EPSG:4326",
                          output_dir=os.path.join(str(tmp_path), "out"),
                          crs_policy="WGS84")


def test_plugin_decoder_matches_backend_decoder():
    """插件里的解析副本必须与后端行为一致（分段 UTF-8、多行、末行无换行）。"""
    from backend.contracts.events import JsonlStreamDecoder as BackendDecoder
    from vecrot_qgis.jobs.protocol import JsonlStreamDecoder as PluginDecoder

    payload = (json.dumps({"event": "progress", "message": "轮廓 中文"},
                          ensure_ascii=False) + "\n" +
               json.dumps({"event": "completed", "message": "done"})).encode("utf-8")

    for chunk in (3, 7, 64, len(payload)):
        b = BackendDecoder()
        p = PluginDecoder()
        out_b, out_p = [], []
        for i in range(0, len(payload), chunk):
            out_b += b.feed(payload[i:i + chunk])
            out_p += p.feed(payload[i:i + chunk])
        out_b += b.flush()
        out_p += p.flush()
        assert out_b == out_p, f"chunk={chunk} 两侧解析不一致"


def test_plugin_decoder_drops_corrupt():
    from vecrot_qgis.jobs.protocol import JsonlStreamDecoder

    dec = JsonlStreamDecoder()
    objs = dec.feed(b'{"a":1}\nnot json\n\xff\xfe\n{"a":2}\n')
    assert [o["a"] for o in objs] == [1, 2]
