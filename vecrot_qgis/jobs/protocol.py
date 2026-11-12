"""事件流的 UTF-8 行解析（纯标准库，无 QGIS 依赖）。

这是 ``backend/contracts/events.py`` 的**最小镜像**：插件不 import 后端包
（QGIS 环境里没有后端依赖），因此保留一份副本。两份行为一致性由
``tests/test_plugin_logic.py::test_plugin_decoder_matches_backend_decoder`` 约束；
改这里必须同步改后端，否则测试会失败。
"""
from __future__ import annotations

import json


class JsonlStreamDecoder:
    """按字节喂入、按行切出 JSON 对象。

    处理三种破损：分段 UTF-8（多字节字符被切开）、一次读到多行、末行无换行。
    """

    def __init__(self, max_line_bytes: int = 1 << 20):
        self._buf = bytearray()
        self._max = max_line_bytes

    def feed(self, data: bytes) -> list[dict]:
        if not data:
            return []
        self._buf.extend(data)
        out: list[dict] = []
        while True:
            nl = self._buf.find(b"\n")
            if nl < 0:
                if len(self._buf) > self._max:
                    self._buf.clear()
                break
            raw = bytes(self._buf[:nl])
            del self._buf[: nl + 1]
            obj = _decode_line(raw)
            if obj is not None:
                out.append(obj)
        return out

    def flush(self) -> list[dict]:
        if not self._buf:
            return []
        raw = bytes(self._buf)
        self._buf.clear()
        obj = _decode_line(raw)
        return [obj] if obj is not None else []


def _decode_line(raw: bytes) -> dict | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


#: UI 认识的事件名；未识别的只记录兼容性信息，不猜语义。
KNOWN_EVENTS = frozenset({"ready", "stage_started", "progress", "warning",
                          "preview_ready", "stage_finished", "completed",
                          "failed", "cancelled"})
