"""SQLite 作业台账（开发文档 §8.3）。

只由协调器写入。记录：作业状态、阶段检查点、窗口状态、det 状态、结果分片。
取消后保留缓存，不自动删除。
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

from ..contracts.version import JobState

SCHEMA = """
CREATE TABLE IF NOT EXISTS job (
    job_id       TEXT PRIMARY KEY,
    state        TEXT NOT NULL,
    stage        TEXT,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    spec_json    TEXT NOT NULL,
    error_code   TEXT,
    error_msg    TEXT,
    exit_code    INTEGER
);
CREATE TABLE IF NOT EXISTS stage (
    job_id       TEXT NOT NULL,
    name         TEXT NOT NULL,
    state        TEXT NOT NULL,
    started_at   REAL,
    finished_at  REAL,
    metrics_json TEXT,
    PRIMARY KEY (job_id, name)
);
CREATE TABLE IF NOT EXISTS window (
    job_id       TEXT NOT NULL,
    window_id    INTEGER NOT NULL,
    xoff         INTEGER NOT NULL,
    yoff         INTEGER NOT NULL,
    width        INTEGER NOT NULL,
    height       INTEGER NOT NULL,
    pad          TEXT,
    state        TEXT NOT NULL,
    n_candidates INTEGER DEFAULT 0,
    error        TEXT,
    PRIMARY KEY (job_id, window_id)
);
CREATE TABLE IF NOT EXISTS detection (
    job_id       TEXT NOT NULL,
    det_id       TEXT NOT NULL,
    rbox_json    TEXT NOT NULL,
    score        REAL NOT NULL,
    source_window INTEGER NOT NULL,
    edge_truncated INTEGER NOT NULL DEFAULT 0,
    vector_status TEXT NOT NULL DEFAULT 'pending',
    geom_reasons TEXT,
    elapsed_s    REAL,
    PRIMARY KEY (job_id, det_id)
);
CREATE TABLE IF NOT EXISTS checkpoint (
    job_id       TEXT NOT NULL,
    key          TEXT NOT NULL,
    value        TEXT NOT NULL,
    updated_at   REAL NOT NULL,
    PRIMARY KEY (job_id, key)
);
CREATE TABLE IF NOT EXISTS suppression (
    job_id       TEXT NOT NULL,
    det_id       TEXT NOT NULL,
    kept_det_id  TEXT NOT NULL,
    iou          REAL NOT NULL,
    PRIMARY KEY (job_id, det_id)
);
CREATE INDEX IF NOT EXISTS idx_det_window ON detection(job_id, source_window);
"""


class JobLedger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)

    # ---- 生命周期 ----
    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- job ----
    def create_job(self, job_id: str, spec: dict) -> None:
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO job(job_id,state,stage,created_at,updated_at,spec_json)"
            " VALUES(?,?,?,?,?,?)",
            (job_id, JobState.CREATED.value, None, now, now,
             json.dumps(spec, ensure_ascii=False)))

    def get_job(self, job_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT job_id,state,stage,created_at,updated_at,spec_json,error_code,error_msg,exit_code"
            " FROM job WHERE job_id=?", (job_id,)).fetchone()
        if not row:
            return None
        return {
            "job_id": row[0], "state": row[1], "stage": row[2],
            "created_at": row[3], "updated_at": row[4],
            "spec": json.loads(row[5]),
            "error_code": row[6], "error_msg": row[7], "exit_code": row[8],
        }

    def set_state(self, job_id: str, state: JobState | str, stage: str | None = None,
                  error_code: str | None = None, error_msg: str | None = None,
                  exit_code: int | None = None) -> None:
        state_v = JobState(state).value
        self._conn.execute(
            "UPDATE job SET state=?, stage=COALESCE(?,stage), updated_at=?,"
            " error_code=COALESCE(?,error_code), error_msg=COALESCE(?,error_msg),"
            " exit_code=COALESCE(?,exit_code) WHERE job_id=?",
            (state_v, stage, time.time(), error_code, error_msg, exit_code, job_id))

    # ---- stage ----
    def stage_start(self, job_id: str, name: str) -> None:
        self._conn.execute(
            "INSERT INTO stage(job_id,name,state,started_at) VALUES(?,?,?,?)"
            " ON CONFLICT(job_id,name) DO UPDATE SET state='running',"
            " started_at=COALESCE(stage.started_at,excluded.started_at), finished_at=NULL",
            (job_id, name, "running", time.time()))

    def stage_finish(self, job_id: str, name: str, metrics: dict | None = None) -> None:
        self._conn.execute(
            "UPDATE stage SET state='done', finished_at=?, metrics_json=? WHERE job_id=? AND name=?",
            (time.time(), json.dumps(metrics or {}, ensure_ascii=False), job_id, name))

    def stage_state(self, job_id: str, name: str) -> str | None:
        row = self._conn.execute(
            "SELECT state FROM stage WHERE job_id=? AND name=?", (job_id, name)).fetchone()
        return row[0] if row else None

    def stage_metrics(self, job_id: str, name: str) -> dict:
        row = self._conn.execute(
            "SELECT metrics_json FROM stage WHERE job_id=? AND name=?", (job_id, name)).fetchone()
        return json.loads(row[0]) if row and row[0] else {}

    # ---- windows ----
    def add_windows(self, job_id: str, windows: list[dict]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO window(job_id,window_id,xoff,yoff,width,height,pad,state)"
                " VALUES(?,?,?,?,?,?,?,?)",
                [(job_id, w["window_id"], w["xoff"], w["yoff"], w["width"], w["height"],
                  json.dumps(w.get("pad")), "pending") for w in windows])

    def set_window_state(self, job_id: str, window_id: int, state: str,
                         n_candidates: int | None = None, error: str | None = None) -> None:
        self._conn.execute(
            "UPDATE window SET state=?, n_candidates=COALESCE(?,n_candidates),"
            " error=COALESCE(?,error) WHERE job_id=? AND window_id=?",
            (state, n_candidates, error, job_id, window_id))

    def windows(self, job_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT window_id,xoff,yoff,width,height,pad,state,n_candidates,error"
            " FROM window WHERE job_id=? ORDER BY window_id", (job_id,)).fetchall()
        return [{"window_id": r[0], "xoff": r[1], "yoff": r[2], "width": r[3],
                 "height": r[4], "pad": json.loads(r[5]) if r[5] else None,
                 "state": r[6], "n_candidates": r[7], "error": r[8]} for r in rows]

    def window_counts(self, job_id: str) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*), SUM(state='ok'), SUM(state='error') FROM window WHERE job_id=?",
            (job_id,)).fetchone()
        return {"total": row[0] or 0, "ok": row[1] or 0, "error": row[2] or 0}

    # ---- detections ----
    def add_detections(self, job_id: str, dets: list[dict]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO detection(job_id,det_id,rbox_json,score,source_window,"
                "edge_truncated,vector_status) VALUES(?,?,?,?,?,?,?)",
                [(job_id, d["det_id"], json.dumps(d["rbox"]), d["det_score"],
                  d["source_window"], int(bool(d["edge_truncated"])), "pending") for d in dets])

    def add_suppressions(self, job_id: str, rows: list[tuple[str, str, float]]) -> None:
        if not rows:
            return
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO suppression(job_id,det_id,kept_det_id,iou)"
                " VALUES(?,?,?,?)",
                [(job_id, d, k, iou) for d, k, iou in rows])

    def set_det_status(self, job_id: str, det_id: str, status: str,
                       reasons: str | None = None, elapsed_s: float | None = None) -> None:
        self._conn.execute(
            "UPDATE detection SET vector_status=?, geom_reasons=COALESCE(?,geom_reasons),"
            " elapsed_s=COALESCE(?,elapsed_s) WHERE job_id=? AND det_id=?",
            (status, reasons, elapsed_s, job_id, det_id))

    def detections(self, job_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT det_id,rbox_json,score,source_window,edge_truncated,vector_status,"
            "geom_reasons,elapsed_s FROM detection WHERE job_id=? ORDER BY det_id",
            (job_id,)).fetchall()
        return [{"det_id": r[0], "rbox": json.loads(r[1]), "det_score": r[2],
                 "source_window": r[3], "edge_truncated": bool(r[4]),
                 "vector_status": r[5], "geom_reasons": r[6], "elapsed_s": r[7]}
                for r in rows]

    def det_status_counts(self, job_id: str) -> dict:
        rows = self._conn.execute(
            "SELECT vector_status, COUNT(*) FROM detection WHERE job_id=? GROUP BY vector_status",
            (job_id,)).fetchall()
        return {r[0]: r[1] for r in rows}

    # ---- checkpoint ----
    def set_checkpoint(self, job_id: str, key: str, value) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO checkpoint(job_id,key,value,updated_at) VALUES(?,?,?,?)",
            (job_id, key, json.dumps(value, ensure_ascii=False), time.time()))

    def get_checkpoint(self, job_id: str, key: str, default=None):
        row = self._conn.execute(
            "SELECT value FROM checkpoint WHERE job_id=? AND key=?", (job_id, key)).fetchone()
        return json.loads(row[0]) if row else default


def open_ledger(work_dir: str) -> JobLedger:
    return JobLedger(os.path.join(work_dir, "ledger.sqlite"))
