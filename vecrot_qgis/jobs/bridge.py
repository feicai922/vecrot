"""作业桥：QProcess 启动外部协调器 + UTF-8 事件流解析（开发文档 §4.1 / §8.2）。

要点：
* 用程序路径 + 参数列表启动，不拼 shell 命令；中文/空格路径必须可用。
* 异步读取 stdout/stderr，绝不在 UI 线程阻塞等待。
* 按 seq 去重；未知事件只记兼容性信息，不猜语义。
* 取消走 cancel-file（Windows 友好的跨进程信号），配合超时强制回收进程树。
"""
from __future__ import annotations

import json
import os
import sys
import time

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from .protocol import KNOWN_EVENTS, JsonlStreamDecoder

LOG_TAG = "Vecrot"


def _clean_path(python_exe: str, env_overrides: dict | None = None) -> str:
    """为外部 python 构造 PATH：环境自身目录 + 系统目录，**不含 QGIS**。

    QGIS 进程的 PATH 里有 ``<QGIS>/bin``，那里是 QGIS 打包的 GDAL/PROJ DLL。
    子进程加载到它们会和自己的 rasterio/pyogrio 期望的版本冲突。
    """
    exe_dir = os.path.dirname(os.path.abspath(python_exe))
    # conda 环境里 python.exe 就在环境根（<env>/python.exe），exe_dir 即环境根。
    # 早期版本误写成 dirname(exe_dir)，得到 <...>/envs 这一级，于是
    # Library\bin / DLLs / Scripts 全都探测不到 —— PATH 里只剩 python 目录，
    # 依赖 GDAL 等 DLL 的包（rasterio/pyogrio）在子进程里就加载失败。
    env_root = exe_dir
    parts = [exe_dir]
    for sub in ("Library\\bin", "Library\\mingw-w64\\bin", "DLLs", "Scripts", "bin"):
        cand = os.path.join(env_root, sub)
        if os.path.isdir(cand):
            parts.append(cand)
    sys_root = os.environ.get("SystemRoot") or r"C:\Windows"
    parts += [os.path.join(sys_root, "System32"), sys_root,
              os.path.join(sys_root, "System32", "Wbem")]
    return os.pathsep.join(dict.fromkeys(parts))


class EventBridge(QObject):
    """一次作业的进程桥。"""

    event_received = pyqtSignal(dict)
    finished = pyqtSignal(int, dict)          # (退出码, 汇总)
    log_line = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: QProcess | None = None
        self._decoder = JsonlStreamDecoder()
        self._seen: set = set()
        self._buffer = bytearray()
        self._err_buffer = bytearray()
        self._cancel_path = ""
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._force_kill)
        self.summary: dict = {}

    # ---- 生命周期
    def start(self, python_exe: str, coordinator_argv: list[str], cwd: str,
              cancel_file: str, env_overrides: dict | None = None) -> None:
        if self._proc is not None:
            raise RuntimeError("已有作业在运行")
        self._cancel_path = cancel_file
        self._seen.clear()
        self._buffer.clear()
        self._err_buffer.clear()
        self.summary = {}

        proc = QProcess(self)
        proc.setProgram(python_exe)
        proc.setArguments(coordinator_argv)
        proc.setWorkingDirectory(cwd)
        proc.setProcessChannelMode(QProcess.SeparateChannels)

        env = QProcessEnvironment()
        # 只传协调器需要的环境；模型 worker 由协调器另行构造干净环境
        for key in ("SystemRoot", "TEMP", "TMP", "USERPROFILE", "HOMEDRIVE",
                    "HOMEPATH", "APPDATA", "LOCALAPPDATA", "COMSPEC"):
            val = os.environ.get(key)
            if val:
                env.insert(key, val)
        # PATH 必须按协调器自己的环境重建，不能继承 QGIS 的。
        # QGIS 的 bin 里带自己的 GDAL/PROJ DLL；协调器 import rasterio 时会先
        # 命中那套版本不匹配的 DLL，报
        # `ImportError: DLL load failed while importing _base: 找不到指定的程序`
        # （实测：QGIS PATH 或干净 PATH 单独用都正常，混在一起必崩）。
        env.insert("PATH", _clean_path(python_exe, env_overrides))
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        for k, v in (env_overrides or {}).items():
            env.insert(k, str(v))
        proc.setProcessEnvironment(env)

        proc.readyReadStandardOutput.connect(self._on_stdout)
        proc.readyReadStandardError.connect(self._on_stderr)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)
        self._proc = proc
        proc.start()

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.state() != QProcess.NotRunning

    def request_cancel(self, grace_ms: int = 20000) -> None:
        """写 cancel-file；超时后回收整棵进程树。"""
        if not self.is_running():
            return
        try:
            os.makedirs(os.path.dirname(self._cancel_path), exist_ok=True)
            with open(self._cancel_path, "w", encoding="utf-8") as f:
                f.write("cancel")
        except OSError as exc:
            self.log_line.emit(f"写取消标志失败：{exc}")
        self._kill_timer.start(grace_ms)

    def _force_kill(self) -> None:
        if self.is_running() and self._proc is not None:
            self.log_line.emit("取消超时，强制结束进程树")
            self._proc.kill()

    # ---- 读取
    def _on_stdout(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardOutput())
        if data:
            for obj in self._decoder.feed(data):
                self._handle_event(obj)

    def _on_stderr(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardError())
        if not data:
            return
        self._err_buffer.extend(data)
        # 只转发完整行，避免半行噪音
        while True:
            nl = self._err_buffer.find(b"\n")
            if nl < 0:
                break
            line = bytes(self._err_buffer[:nl]).decode("utf-8", errors="replace").strip()
            del self._err_buffer[: nl + 1]
            if line:
                self.log_line.emit(line)
                QgsMessageLog.logMessage(line, LOG_TAG, Qgis.Info)

    def _on_error(self, err) -> None:
        self.log_line.emit(f"QProcess 错误：{err}")
        if err == QProcess.FailedToStart:
            self.finished.emit(-1, {"code": "PROCESS_FAILED_TO_START",
                                    "message": "协调器进程无法启动，请检查部署 manifest 的解释器路径"})
            self._proc = None

    def _handle_event(self, obj: dict) -> None:
        seq = obj.get("seq")
        if seq is not None:
            if seq in self._seen:
                return
            self._seen.add(seq)
        event = obj.get("event")
        if event not in KNOWN_EVENTS:
            self.log_line.emit(f"未识别事件 {event!r}（协议 {obj.get('schema_version')}）")
        if event == "completed":
            self.summary = obj.get("extra") or {}
        self.event_received.emit(obj)

    def _on_finished(self, code: int, status) -> None:
        for obj in self._decoder.flush():
            self._handle_event(obj)
        if self._err_buffer:
            tail = bytes(self._err_buffer).decode("utf-8", errors="replace").strip()
            del self._err_buffer[:]
            if tail:
                self.log_line.emit(tail)
        self._kill_timer.stop()
        try:
            os.remove(self._cancel_path)
        except OSError:
            pass
        self._proc = None
        self.finished.emit(int(code), dict(self.summary))


def read_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    dep_dir = os.path.dirname(os.path.abspath(path))
    # 与 backend.cli 一致：相对路径以仓库根为基准
    manifest["_root"] = os.path.dirname(dep_dir)
    manifest["_manifest_dir"] = dep_dir
    return manifest


def resolve(manifest: dict, rel: str | None) -> str | None:
    if not rel:
        return None
    if os.path.isabs(rel):
        return rel
    return os.path.normpath(os.path.join(manifest.get("_root") or "", rel))


def coordinator_python(manifest: dict) -> str:
    env = (manifest.get("environments") or {}).get("coordinator") or {}
    exe = env.get("python")
    if not exe:
        raise ValueError("部署 manifest 缺少 environments.coordinator.python")
    if not os.path.isabs(exe):
        exe = os.path.join(manifest.get("_root") or "", exe)
    if not os.path.isfile(exe):
        raise ValueError(f"协调器解释器不存在：{exe}")
    return exe


def repo_root_from(manifest_path: str) -> str:
    """确定后端代码仓库根（``python -m backend.cli`` 的执行目录）。

    优先用 manifest 里显式写的 ``repo_root``：部署时 manifest 常被复制到仓库之外
    （如 ``%USERPROFILE%\\vecrot\\deployment``），此时「上一层目录」不是代码所在位置，
    推出来的 cwd 会让 ``-m backend.cli`` 找不到模块
    （实测：ModuleNotFoundError: No module named 'backend'）。

    没有显式字段时回退到「manifest 的上一层」并做存在性校验。
    """
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        manifest = {}

    explicit = manifest.get("repo_root")
    if isinstance(explicit, str) and explicit:
        cand = explicit if os.path.isabs(explicit) else os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(manifest_path)), explicit))
        if os.path.isdir(os.path.join(cand, "backend")):
            return cand

    dep_dir = os.path.dirname(os.path.abspath(manifest_path))
    guess = os.path.dirname(dep_dir)
    if os.path.isdir(os.path.join(guess, "backend")):
        return guess
    if os.path.isdir(os.path.join(dep_dir, "backend")):
        return dep_dir
    return guess


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
