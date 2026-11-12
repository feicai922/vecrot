"""几何工具：栅格/地理坐标变换与 AOI 处理。

坐标约定（开发文档 §6.4）：
    生成坐标 → 实际裁剪像素 → 源影像像素 → 源地理坐标 → 可选输出 CRS

所有变换都显式记录，绝不从文件名反推偏移。
"""

from .transform import (
    PixelWindow,
    aoi_transform_densified,
    geo_to_pixel_window,
    pixel_corners_to_geo,
    window_affine,
)
from .raster import RasterInfo, inspect_raster

__all__ = [
    "PixelWindow",
    "RasterInfo",
    "inspect_raster",
    "aoi_transform_densified",
    "geo_to_pixel_window",
    "pixel_corners_to_geo",
    "window_affine",
]
