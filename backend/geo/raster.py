"""栅格预检与分块读取（开发文档 §6.1 / §6.2）。

预检必须拒绝而不是猜：CRS 缺失不猜 EPSG，非简单仿射（RPC/GCP）要求在受控预处理后
生成正射工作栅格。读取一律走窗口，不整景驻留。
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np


class RasterError(RuntimeError):
    """预检失败。code 供 UI 分类。"""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass
class RasterInfo:
    path: str
    width: int
    height: int
    count: int
    dtypes: tuple[str, ...]
    crs: str | None
    transform: tuple[float, float, float, float, float, float]
    nodata: list[float | None]
    colorinterp: tuple[str, ...] = ()
    block_shapes: tuple[tuple[int, int], ...] = ()
    bounds: tuple[float, float, float, float] | None = None
    warnings: list[str] = field(default_factory=list)
    #: 非简单仿射（RPC/GCP）时为 True；首版拒绝推理
    has_rpc: bool = False

    @property
    def is_affine_ok(self) -> bool:
        gt = self.transform
        return abs(gt[1] * gt[5] - gt[2] * gt[4]) > 1e-18

    @property
    def approx_resolution(self) -> tuple[float, float]:
        gt = self.transform
        rx = math.hypot(gt[1], gt[4])
        ry = math.hypot(gt[2], gt[5])
        return rx, ry

    @property
    def is_rotated(self) -> bool:
        gt = self.transform
        return abs(gt[2]) > 1e-12 or abs(gt[4]) > 1e-12


def _is_local_path(path: str) -> bool:
    """源 URI 不能直接当本地路径（§6.1）。"""
    lowered = path.lower()
    if "://" in lowered:
        return lowered.startswith("file://")
    return True


def inspect_raster(path: str, bands_rgb: tuple[int, int, int] = (1, 2, 3),
                   require_local: bool = True) -> RasterInfo:
    """打开栅格做完整预检；任何一项不满足就抛 RasterError。"""
    import rasterio

    if not path:
        raise RasterError("RASTER_MISSING", "未提供影像路径")
    if require_local and not _is_local_path(path):
        raise RasterError("RASTER_NOT_LOCAL",
                          f"首版只支持本地文件，请先导出带地理参考的栅格：{path}")
    if not os.path.exists(path):
        raise RasterError("RASTER_MISSING", f"影像不存在：{path}")
    if not os.access(path, os.R_OK):
        raise RasterError("RASTER_UNREADABLE", f"影像不可读：{path}")

    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise RasterError("CRS_MISSING", "影像缺少 CRS；本工具不猜 EPSG，请在源数据补齐定义")
            gt = tuple(float(v) for v in src.transform.to_gdal())
            if abs(gt[1] * gt[5] - gt[2] * gt[4]) < 1e-18:
                raise RasterError("AFFINE_DEGENERATE", "影像仿射矩阵不可逆")
            for b in bands_rgb:
                if b < 1 or b > src.count:
                    raise RasterError("BAND_MISSING",
                                      f"波段 {b} 不存在（影像只有 {src.count} 个波段）")
            has_rpc = bool(getattr(src, "rpcs", None))
            # rasterio 的 gcps 返回 (list, crs)，空时是 ([], None)——元组本身恒为真
            gcps = getattr(src, "gcps", None)
            if gcps:
                gcp_list = gcps[0] if isinstance(gcps, (tuple, list)) else gcps
                has_rpc = has_rpc or bool(gcp_list)
            ci = src.colorinterp or ()
            block_shapes = []
            for item in src.block_windows(1):
                # rasterio 产生 ((block_row, block_col), Window)
                win = item[1] if isinstance(item, tuple) else item
                block_shapes.append((win.height, win.width))
            ci_names = tuple(getattr(c, "name", str(c)) for c in ci)
            info = RasterInfo(
                path=path,
                width=src.width,
                height=src.height,
                count=src.count,
                dtypes=tuple(src.dtypes),
                crs=src.crs.to_string(),
                transform=gt,
                nodata=list(src.nodatavals),
                colorinterp=ci_names,
                block_shapes=tuple(block_shapes),
                bounds=tuple(src.bounds),
                has_rpc=has_rpc,
            )
            if src.width < 32 or src.height < 32:
                info.warnings.append(f"影像很小（{src.width}×{src.height}），结果可能无意义")
            # 波段→RGB 映射是否可疑：给了明确波段号就不猜，只提示
            if ci_names and len(ci_names) >= max(bands_rgb):
                picked = [ci_names[b - 1] for b in bands_rgb]
                if len(set(picked)) < 3:
                    info.warnings.append(
                        f"所选波段解释为 {picked}，不是三个不同的颜色通道；请确认 bands_rgb")
    except RasterError:
        raise
    except Exception as exc:                      # noqa: BLE001 - 统一转成可诊断错误
        raise RasterError("RASTER_UNREADABLE", f"无法读取影像：{type(exc).__name__}: {exc}") from exc

    if info.has_rpc:
        raise RasterError("RASTER_RPC_UNSUPPORTED",
                          "影像依赖 RPC/GCP 而非简单仿射；请先生成正射工作栅格再运行")
    return info


# ------------------------------------------------------- 读取与 RGB 合成


@dataclass
class StretchParams:
    """拉伸参数：整景或 AOI 级固定，绝不逐瓦片独立拉伸（§6.2）。"""

    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    dtype: str = "uint8"

    def to_dict(self) -> dict:
        return {"lo": list(self.lo), "hi": list(self.hi), "dtype": self.dtype}


def to_rio_window(window):
    """我们的 PixelWindow → rasterio.windows.Window（rasterio 只接受自己的类型）。"""
    from rasterio.windows import Window

    return Window(window.xoff, window.yoff, window.width, window.height)


def compute_stretch(src, window, bands_rgb, sample_max: int = 2_000_000) -> StretchParams:
    """在一个固定窗口上统计拉伸上下界（2%/98%），供所有瓦片共用。"""
    data = src.read([bands_rgb[0], bands_rgb[1], bands_rgb[2]], window=to_rio_window(window))
    lo, hi = [], []
    nodata = src.nodatavals
    for i, b in enumerate(bands_rgb):
        arr = data[i].ravel()
        nd = nodata[b - 1] if nodata and len(nodata) >= b else None
        if nd is not None:
            arr = arr[arr != nd]
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            lo.append(0.0)
            hi.append(1.0)
            continue
        if arr.size > sample_max:
            step = max(1, arr.size // sample_max)
            arr = arr[::step]
        lo.append(float(np.percentile(arr, 2)))
        hi.append(float(np.percentile(arr, 98)))
    # 常值波段：给出非退化区间，避免除零
    for i in range(3):
        if hi[i] <= lo[i]:
            hi[i] = lo[i] + 1.0
    return StretchParams(tuple(lo), tuple(hi), dtype="uint8")


def read_rgb_window(src, window, bands_rgb, stretch: StretchParams,
                    is_8bit: bool) -> tuple[np.ndarray, np.ndarray]:
    """读取一个窗口的 RGB uint8 数组与有效掩码。

    * 8 位输入尽量保持原值（不重拉伸，只做越界裁剪）。
    * 更高位深用固定 stretch 参数线性拉伸。
    * 通道顺序一律输出 RGB；OpenCV 的 BGR 只在需要时由调用方显式转换。
    """
    data = src.read([bands_rgb[0], bands_rgb[1], bands_rgb[2]],
                    window=to_rio_window(window))
    valid = np.ones(data.shape[1:], dtype=bool)
    nodata = src.nodatavals
    for i, b in enumerate(bands_rgb):
        nd = nodata[b - 1] if nodata and len(nodata) >= b else None
        band = data[i].astype(np.float64, copy=True)
        if nd is not None:
            m = np.isfinite(band) & (band != nd)
            valid &= m
            band[~m] = 0.0
        else:
            valid &= np.isfinite(band)
            band[~np.isfinite(band)] = 0.0
        data[i] = band

    if is_8bit:
        out = data
    else:
        lo = np.asarray(stretch.lo, dtype=np.float64)[:, None, None]
        hi = np.asarray(stretch.hi, dtype=np.float64)[:, None, None]
        out = (data - lo) / (hi - lo) * 255.0

    out = np.clip(out, 0, 255).astype(np.uint8)
    out = np.transpose(out, (1, 2, 0))            # (H, W, 3)
    return out, valid


def check_disk_space(dir_path: str, need_bytes: int) -> None:
    import shutil

    parent = dir_path
    while parent and not os.path.exists(parent):
        nxt = os.path.dirname(parent)
        if nxt == parent:
            break
        parent = nxt
    if not parent or not os.path.exists(parent):
        raise RasterError("OUTPUT_UNWRITABLE", f"输出目录不存在且无法回溯到可写父目录：{dir_path}")
    free = shutil.disk_usage(parent).free
    if free < need_bytes:
        raise RasterError(
            "DISK_FULL",
            f"磁盘空间不足：需要约 {need_bytes/1e6:.0f} MB，可用 {free/1e6:.0f} MB（{parent}）")


def check_writable(dir_path: str) -> None:
    try:
        os.makedirs(dir_path, exist_ok=True)
        probe = os.path.join(dir_path, ".vecrot_write_probe")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
    except OSError as exc:
        raise RasterError("OUTPUT_UNWRITABLE", f"输出目录不可写：{dir_path}（{exc}）") from exc
