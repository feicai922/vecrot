"""外部进程启动与环境隔离（开发文档 §4.1）。

要点：
* 用程序路径 + 参数列表启动，绝不拼 shell 命令；路径含中文/空格必须可用。
* 清理父进程污染：移除 PYTHONHOME/PYTHONPATH/QT_*/QGIS_*，按环境重建 PATH。
* 异步读 stdout/stderr，不阻塞。
* 取消：先写控制文件，worker 在批次边界退出；超时回收整棵进程树。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time

#: 从父进程继承会毒害外部环境（尤其是从 QGIS 启动时）的变量
_POISONED_VARS = (
    "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONEXECUTABLE",
    "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "QGIS_PREFIX_PATH",
    "GDAL_DRIVER_PATH", "PYQGIS_STARTUP",
)


def build_clean_env(python_exe: str, gis_env_dir: str | None = None,
                    extra_path: list[str] | None = None,
                    extra_env: dict | None = None) -> dict:
    """为外部 worker 构造干净环境。

    ``gis_env_dir`` 用于指向 GIS 环境的 GDAL/PROJ 资源；检测/生成环境不应继承
    QGIS 的 GIS DLL 搜索目录，因此这里只加调用方显式指定的目录。
    """
    env = {k: v for k, v in os.environ.items() if k not in _POISONED_VARS}
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    exe_dir = os.path.dirname(os.path.abspath(python_exe))
    path_parts = [exe_dir]
    # conda 环境：Library/bin 是 DLL 的实际位置，必须显式加（不假定激活脚本已跑）。
    # python.exe 就在环境根，所以环境根 == exe_dir；早期写成 dirname(exe_dir)
    # 会指到 <...>/envs 上，下面这些子目录一个都探测不到。
    env_root = exe_dir
    for sub in ("Library\\bin", "Library\\mingw-w64\\bin", "DLLs", "Scripts", "bin"):
        cand = os.path.join(env_root, sub)
        if os.path.isdir(cand):
            path_parts.append(cand)
    for p in (extra_path or []):
        if p:
            path_parts.append(p)
    # 保留系统目录，但排在环境目录之后
    sys_root = os.environ.get("SystemRoot", r"C:\Windows")
    path_parts += [os.path.join(sys_root, "System32"), sys_root,
                   os.path.join(sys_root, "System32", "Wbem")]
    env["PATH"] = os.pathsep.join(dict.fromkeys(path_parts))

    if gis_env_dir:
        for name, sub in (("GDAL_DATA", ("share", "gdal")),
                          ("PROJ_DATA", ("share", "proj")),
                          ("PROJ_LIB", ("share", "proj"))):
            cand = os.path.join(gis_env_dir, *sub)
            if os.path.isdir(cand):
                env[name] = cand
    for k, v in (extra_env or {}).items():
        env[k] = str(v)
    return env


class WorkerProcess:
    """一个受管外部进程：异步收集 stdout/stderr，支持取消与进程树回收。"""

    def __init__(self, argv: list[str], env: dict, cwd: str | None = None,
                 on_stdout: "callable | None" = None,
                 on_stderr: "callable | None" = None):
        self.argv = argv
        self._out_chunks: list[bytes] = []
        self._err_chunks: list[bytes] = []
        self._on_stdout = on_stdout
        self._on_stderr = on_stderr
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            cwd=cwd,
            env=env,
            # Windows：不弹窗、独立进程组，便于按树回收
            creationflags=(subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
            if os.name == "nt" else 0,
            bufsize=0,
        )
        self._t_out = threading.Thread(target=self._pump, args=(self._proc.stdout, True),
                                       daemon=True)
        self._t_err = threading.Thread(target=self._pump, args=(self._proc.stderr, False),
                                       daemon=True)
        self._t_out.start()
        self._t_err.start()

    def _pump(self, pipe, is_stdout: bool) -> None:
        chunks = self._out_chunks if is_stdout else self._err_chunks
        cb = self._on_stdout if is_stdout else self._on_stderr
        try:
            while True:
                data = pipe.read(4096)
                if not data:
                    break
                with self._lock:
                    chunks.append(data)
                if cb is not None:
                    cb(data)
        except (ValueError, OSError):
            pass
        finally:
            try:
                pipe.close()
            except OSError:
                pass

    @property
    def pid(self) -> int:
        return self._proc.pid

    def poll(self) -> int | None:
        return self._proc.poll()

    def wait(self, timeout: float | None = None) -> int:
        code = self._proc.wait(timeout=timeout)
        self._t_out.join(timeout=5)
        self._t_err.join(timeout=5)
        return code

    def stdout_bytes(self) -> bytes:
        with self._lock:
            return b"".join(self._out_chunks)

    def stderr_text(self) -> str:
        with self._lock:
            return b"".join(self._err_chunks).decode("utf-8", errors="replace")

    def terminate_tree(self, grace: float = 5.0) -> None:
        """先优雅终止，超时再强杀整棵进程树。"""
        if self._proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                self._proc.terminate()
            else:
                self._proc.terminate()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(self._proc.pid)],
                               capture_output=True, check=False,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                self._proc.kill()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass

    def kill(self) -> None:
        self.terminate_tree(grace=2.0)


def write_request(payload: dict, work_dir: str, name: str) -> str:
    """把请求写成 UTF-8 JSON 文件（避免命令行长度限制与编码问题）。"""
    os.makedirs(work_dir, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix=f"{name}_", suffix=".json", dir=work_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        import json

        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def python_for(manifest: dict, role: str) -> str:
    """按 manifest 取角色对应的 python.exe；没有就明确报错。"""
    envs = (manifest.get("environments") or {})
    entry = envs.get(role) or {}
    exe = entry.get("python")
    if not exe:
        raise RuntimeError(f"deployment manifest 未配置环境 {role!r} 的 python 路径")
    if not os.path.isabs(exe):
        base = manifest.get("_root") or ""
        exe = os.path.join(base, exe)
    if not os.path.isfile(exe):
        raise RuntimeError(f"环境 {role!r} 的 python 不存在：{exe}")
    return exe
