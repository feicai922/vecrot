"""事件发射器：stdout 每行一个 JSON 对象（开发文档 §8.2）。

约定：
* 协议只走 stdout；worker 的人类日志走 stderr。
* `total` 未知时为 null —— 不编造百分比。
* `seq` 单调递增，插件按 seq 去重。
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
from datetime import datetime, timezone
from typing import Any, TextIO

from .version import EVENT_TYPES, SCHEMA_VERSION


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class EventEmitter:
    """线程安全的事件发射器。"""

    def __init__(self, job_id: str, stream: TextIO | None = None):
        self.job_id = job_id
        self._stream = stream if stream is not None else sys.stdout
        self._seq = 0
        self._lock = threading.Lock()

    def emit(self, event: str, stage: str | None = None, done: int | None = None,
             total: int | None = None, message: str = "", code: str | None = None,
             **extra: Any) -> dict:
        if event not in EVENT_TYPES:
            raise ValueError(f"unknown event type: {event}")
        with self._lock:
            self._seq += 1
            payload = {
                "schema_version": SCHEMA_VERSION,
                "job_id": self.job_id,
                "seq": self._seq,
                "timestamp_utc": utc_now(),
                "stage": stage,
                "event": event,
                "done": done,
                "total": total,
                "message": message,
            }
            if code is not None:
                # 失败/警告的错误码放在顶层，UI 与测试都能直接读到
                payload["code"] = code
            for k, v in extra.items():
                if k in payload:
                    raise ValueError(f"event field collision: {k}")
                payload[k] = v
            line = json.dumps(payload, ensure_ascii=False)
        try:
            self._stream.write(line + "\n")
            self._stream.flush()
        except (OSError, ValueError):
            # 管道被父进程关闭（例如父进程已取消）：协议流失效不应让 worker 崩在
            # 一个"报告错误"的调用里。后续写入同样静默忽略。
            self._stream = io.StringIO()
        return payload


class JsonlStreamDecoder:
    """按字节喂入、按行切出 JSON 对象。

    负责开发文档 §8.2 的三种破损情况：分段 UTF-8（多字节字符被切开）、
    一次读到多行、末行无换行。还有单条消息大小上限。
    """

    def __init__(self, max_line_bytes: int = 1 << 20):
        self._buf = bytearray()
        self._max = max_line_bytes
        self._skipped = 0

    @property
    def skipped_bytes(self) -> int:
        """因超长被丢弃的字节数（诊断用）。"""
        return self._skipped

    def feed(self, data: bytes) -> list[dict]:
        """返回本次能完整解出的所有对象；残缺行留在内部缓冲。"""
        if not data:
            return []
        self._buf.extend(data)
        out: list[dict] = []
        while True:
            nl = self._buf.find(b"\n")
            if nl < 0:
                if len(self._buf) > self._max:
                    # 超长行：丢弃直到下一个换行，避免内存无界增长
                    drop = len(self._buf)
                    self._buf.clear()
                    self._skipped += drop
                break
            raw = bytes(self._buf[:nl])
            del self._buf[: nl + 1]
            obj = self._decode_line(raw)
            if obj is not None:
                out.append(obj)
        return out

    def flush(self) -> list[dict]:
        """流结束时调用：处理末行无换行的情况。"""
        if not self._buf:
            return []
        raw = bytes(self._buf)
        self._buf.clear()
        obj = self._decode_line(raw)
        return [obj] if obj is not None else []

    @staticmethod
    def _decode_line(raw: bytes) -> dict | None:
        raw = raw.strip()
        if not raw:
            return None
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None          # 一个完整行却不是合法 UTF-8：跳过，不崩
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        return obj if isinstance(obj, dict) else None


def worker_log(message: str) -> None:
    """worker 的人类日志：写 stderr，绝不污染协议流。"""
    sys.stderr.write(f"[{os.getpid()}] {message}\n")
    sys.stderr.flush()
