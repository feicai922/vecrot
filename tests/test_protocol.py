"""协议层测试：事件、UTF-8 分段解析、job.json 校验、状态机。"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from backend.contracts.events import EventEmitter, JsonlStreamDecoder
from backend.contracts.jobspec import JobSpecError, parse_job_spec
from backend.contracts.version import JobState, can_transition, check_transition


# ---------------------------------------------------------------- 事件流


def test_event_emitter_monotonic_seq():
    buf = io.StringIO()
    em = EventEmitter("job1", stream=buf)
    em.emit("ready")
    em.emit("progress", stage="DETECTING", done=1, total=10, message="x")
    lines = [json.loads(l) for l in buf.getvalue().strip().splitlines()]
    assert [l["seq"] for l in lines] == [1, 2]
    assert all(l["schema_version"] == "1.0" for l in lines)
    assert all(l["job_id"] == "job1" for l in lines)
    assert lines[1]["total"] == 10


def test_event_emitter_rejects_unknown_event():
    with pytest.raises(ValueError):
        EventEmitter("j", stream=io.StringIO()).emit("nonsense")


def test_event_emitter_unknown_total_is_null():
    buf = io.StringIO()
    EventEmitter("j", stream=buf).emit("progress", stage="DETECTING")
    rec = json.loads(buf.getvalue().strip())
    assert rec["total"] is None and rec["done"] is None


def test_decoder_split_utf8_chinese():
    """分段 UTF-8：一个中文字符被切成两半也必须正确拼回。"""
    payload = json.dumps({"message": "轮廓生成失败：中文路径"}, ensure_ascii=False) + "\n"
    raw = payload.encode("utf-8")
    dec = JsonlStreamDecoder()
    objs = []
    for i in range(0, len(raw), 3):          # 每次只喂 3 字节，必然切断多字节字符
        objs.extend(dec.feed(raw[i:i + 3]))
    objs.extend(dec.flush())
    assert len(objs) == 1
    assert objs[0]["message"] == "轮廓生成失败：中文路径"


def test_decoder_multiple_lines_one_chunk():
    dec = JsonlStreamDecoder()
    data = b'{"a":1}\n{"a":2}\n{"a":3}\n'
    objs = dec.feed(data)
    assert [o["a"] for o in objs] == [1, 2, 3]


def test_decoder_last_line_without_newline():
    dec = JsonlStreamDecoder()
    assert dec.feed(b'{"a":1}\n{"a":2}') == [{"a": 1}]
    assert dec.flush() == [{"a": 2}]


def test_decoder_corrupt_line_does_not_crash():
    dec = JsonlStreamDecoder()
    objs = dec.feed(b'{"a":1}\nNOT JSON\n{"a":2}\n\xff\xfe\n[1,2]\n')
    assert [o.get("a") for o in objs] == [1, 2]


def test_decoder_oversized_line_dropped():
    dec = JsonlStreamDecoder(max_line_bytes=64)
    out = dec.feed(b"x" * 200)
    assert out == []
    assert dec.skipped_bytes > 0
    # 之后仍能正常解析
    assert dec.feed(b'{"a":1}\n') == [{"a": 1}]


# --------------------------------------------------------------- job.json


def _base_job(**over):
    job = {
        "schema_version": "1.0",
        "job_id": "20261110_test",
        "input": {
            "raster_path": "C:/data/scene.tif",
            "bands_rgb": [1, 2, 3],
            "aoi_geojson": {"type": "Polygon",
                            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
            "aoi_crs": "EPSG:4326",
        },
        "inference": {"score_threshold": 0.3, "tile_size": 1024, "tile_overlap": 256},
        "output": {"directory": "C:/out/job1", "crs_policy": "source"},
    }
    job.update(over)
    return job


def test_jobspec_ok():
    spec = parse_job_spec(_base_job())
    assert spec.job_id == "20261110_test"
    assert spec.inference.tile_size == 1024
    assert spec.bands_rgb == (1, 2, 3)


def test_jobspec_rejects_bad_schema_version():
    with pytest.raises(JobSpecError) as ei:
        parse_job_spec(_base_job(schema_version="2.0"))
    assert ei.value.code == "SCHEMA_UNSUPPORTED"


def test_jobspec_rejects_relative_paths():
    job = _base_job()
    job["input"]["raster_path"] = "scene.tif"
    with pytest.raises(JobSpecError) as ei:
        parse_job_spec(job)
    assert "绝对路径" in ei.value.message


def test_jobspec_rejects_unsafe_job_id():
    with pytest.raises(JobSpecError):
        parse_job_spec(_base_job(job_id="../escape"))


def test_jobspec_rejects_overlap_ge_tile():
    job = _base_job()
    job["inference"]["tile_overlap"] = 1024
    with pytest.raises(JobSpecError) as ei:
        parse_job_spec(job)
    assert "tile_overlap" in ei.value.message


def test_jobspec_rejects_pipelined_mode_explicitly():
    """流水线是 M5 能力：必须明确拒绝，不能悄悄退回串行。"""
    job = _base_job()
    job["inference"]["mode"] = "pipelined"
    with pytest.raises(JobSpecError) as ei:
        parse_job_spec(job)
    assert ei.value.code == "MODE_NOT_IMPLEMENTED"


def test_jobspec_rejects_guessed_crs():
    job = _base_job()
    job["output"]["crs_policy"] = "WGS84"
    with pytest.raises(JobSpecError) as ei:
        parse_job_spec(job)
    assert "不猜 EPSG" in ei.value.message


def test_jobspec_rejects_bad_aoi():
    job = _base_job()
    job["input"]["aoi_geojson"] = {"type": "Point", "coordinates": [0, 0]}
    with pytest.raises(JobSpecError):
        parse_job_spec(job)


def test_jobspec_rejects_bad_bands():
    job = _base_job()
    job["input"]["bands_rgb"] = [1, 2]
    with pytest.raises(JobSpecError):
        parse_job_spec(job)


def test_jobspec_rejects_score_out_of_range():
    job = _base_job()
    job["inference"]["score_threshold"] = 1.5
    with pytest.raises(JobSpecError):
        parse_job_spec(job)


# --------------------------------------------------------------- 状态机


def test_state_machine_happy_path():
    check_transition(JobState.CREATED, JobState.VALIDATING)
    check_transition(JobState.VALIDATING, JobState.PREPARING)
    check_transition(JobState.EXPORTING, JobState.SUCCEEDED)
    check_transition(JobState.EXPORTING, JobState.PARTIAL)


def test_state_machine_rejects_backwards():
    assert not can_transition(JobState.DETECTING, JobState.PREPARING)
    with pytest.raises(ValueError):
        check_transition(JobState.DETECTING, JobState.PREPARING)


def test_state_machine_cancel_from_any_active():
    for s in (JobState.VALIDATING, JobState.DETECTING, JobState.VECTORIZING,
              JobState.EXPORTING):
        assert can_transition(s, JobState.CANCELLING)
    assert can_transition(JobState.CANCELLING, JobState.CANCELLED)


def test_state_machine_terminal_cannot_move():
    assert not can_transition(JobState.SUCCEEDED, JobState.DETECTING)
    assert not can_transition(JobState.CANCELLED, JobState.VALIDATING)
