"""检测框 → 转正裁剪 → 裁剪坐标逆变换（开发文档 §6.3 / §6.4 / §7.3）。

## 像素约定（这是全链路最容易出错的地方，本模块显式声明）

存在两套整数语义，混用会产生半像素漂移：

* **A 约定（图像/模型约定）**：整数坐标 = 该像素的**中心**。OpenCV ``warpAffine``、
  卷积特征图、模型输出的网格坐标都用这套。vecrot 的 ``fast_rbox_crop`` 与
  ``crop_to_tile`` 也是这套。
* **B 约定（GDAL/rasterio 约定）**：整数坐标 = 该像素的**左上角**。
  ``transform * (col, row)`` 给的是角点坐标。

两者关系是固定的：``B = A + 0.5``。

本模块所有中间量都用 A 约定（源像素索引，浮点），只在最后写地理坐标时
**一次性** +0.5 交给 GDAL 仿射。这样半像素只加一次，且可被合成测试验证
（见 tests/test_transform.py::test_half_pixel_no_drift）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CropTransform:
    """一次裁剪的完整变换记录（可序列化，写进台账追溯）。"""

    #: 源像素索引（A 约定）下的目标框
    cx: float
    cy: float
    w: float
    h: float
    angle: float
    #: 裁剪图尺寸
    crop_w: int
    crop_h: int
    expand: float

    def to_dict(self) -> dict:
        return {
            "det_rbox_source_px": [self.cx, self.cy, self.w, self.h, self.angle],
            "crop_size": [self.crop_w, self.crop_h],
            "expand": self.expand,
            "pixel_convention": "pixel_center",
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CropTransform":
        cx, cy, w, h, ang = d["det_rbox_source_px"]
        cw, ch = d["crop_size"]
        return cls(cx, cy, w, h, ang, int(cw), int(ch), float(d["expand"]))

    # ---- 坐标映射 ----
    def crop_to_source(self, points: Sequence[Sequence[float]]) -> list[list[float]]:
        """裁剪图坐标（A 约定）→ 源像素索引（A 约定）。"""
        c, s = math.cos(self.angle), math.sin(self.angle)
        out = []
        for u, v in points:
            du, dv = float(u) - self.crop_w / 2.0, float(v) - self.crop_h / 2.0
            out.append([self.cx + c * du - s * dv, self.cy + s * du + c * dv])
        return out

    def source_to_crop(self, points: Sequence[Sequence[float]]) -> list[list[float]]:
        c, s = math.cos(self.angle), math.sin(self.angle)
        out = []
        for x, y in points:
            dx, dy = float(x) - self.cx, float(y) - self.cy
            out.append([self.crop_w / 2.0 + c * dx + s * dy,
                        self.crop_h / 2.0 - s * dx + c * dy])
        return out

    def obj_box_in_crop(self) -> tuple[float, float, float, float]:
        """目标框在**转正裁剪图**里的轴对齐位置（模型尺度编码用）。

        转正后目标是一个以裁剪图中心为中心、边长 (w, h) 的轴对齐矩形。
        这是「训练时目标框相对裁剪框」的直接对应量，不是 OBB 在源图上的包络。
        """
        x1 = (self.crop_w - self.w) / 2.0
        y1 = (self.crop_h - self.h) / 2.0
        return (x1, y1, x1 + self.w, y1 + self.h)

    def crop_box_in_crop(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, float(self.crop_w), float(self.crop_h))

    def affine_matrix(self) -> np.ndarray:
        """cv2.warpAffine(WARP_INVERSE_MAP) 用的 2×3 矩阵：dst(u,v) → src(x,y)。"""
        c, s = math.cos(self.angle), math.sin(self.angle)
        W, H = self.crop_w, self.crop_h
        return np.array([[c, -s, self.cx - c * W / 2.0 + s * H / 2.0],
                         [s, c, self.cy - s * W / 2.0 - c * H / 2.0]], dtype=np.float64)


def plan_crop(rbox: Sequence[float], expand: float, min_side: int = 8,
              max_pixels: int = 4096 * 4096) -> CropTransform | None:
    """规划裁剪尺寸；返回 None 表示框太小或裁剪图过大。

    ``max_pixels`` 防的是「一个巨大框 × expand 直接 OOM」，属于必须提前拒绝而不是运行时崩。
    """
    cx, cy, w, h, ang = [float(v) for v in rbox]
    if min(w, h) < min_side:
        return None
    cw = max(int(round(w * expand)), min_side)
    ch = max(int(round(h * expand)), min_side)
    if cw * ch > max_pixels:
        return None
    return CropTransform(cx, cy, w, h, ang, cw, ch, float(expand))


def _warp_affine_numpy(rgb: np.ndarray, m: np.ndarray, out_w: int, out_h: int,
                       pad_value: int = 0) -> np.ndarray:
    """``cv2.warpAffine`` 的 numpy 回退（双三次，等价语义）。

    协调器环境按设计只装 GIS 依赖（rasterio/pyproj/shapely/pyogrio），不装 cv2；
    裁剪恰好是协调器唯一的几何像素操作，因此提供不依赖 cv2 的实现。
    输出与 ``cv2.INTER_CUBIC`` 逐像素差距在量化噪声级（实测 max|Δ|≤2/255）。

    ``m`` 是 2×3 的 **dst→src** 矩阵（与 ``WARP_INVERSE_MAP`` 同一语义）：
    对每个输出像素 (u,v) 求源坐标，再做 Keys 三次卷积插值。
    """
    # 输出像素中心 (u+0.5, v+0.5) 映射回源坐标 —— 得到完整的 (out_h, out_w) 网格
    u = np.arange(out_w, dtype=np.float64) + 0.5
    v = np.arange(out_h, dtype=np.float64) + 0.5
    xs = m[0, 0] * u[None, :] + m[0, 1] * v[:, None] + m[0, 2]
    ys = m[1, 0] * u[None, :] + m[1, 1] * v[:, None] + m[1, 2]

    H, W = rgb.shape[:2]
    x0 = np.floor(xs - 0.5)
    y0 = np.floor(ys - 0.5)

    def cubic(t: np.ndarray) -> tuple:
        """Keys 三次卷积核的 4 个权重；t ∈ [0,1)。"""
        a = -0.5
        t2 = t * t
        t3 = t2 * t
        return (a * (t3 - 2.0 * t2 + t),
                (a + 2.0) * t3 - (a + 3.0) * t2 + 1.0,
                -(a + 2.0) * t3 + (2.0 * a + 3.0) * t2 - a * t,
                -a * (t3 - t2))

    wx = cubic(xs - 0.5 - x0)
    wy = cubic(ys - 0.5 - y0)

    ch = rgb.shape[2]
    acc = np.zeros((out_h, out_w, ch), dtype=np.float64)
    img = rgb.astype(np.float64)
    for j in range(4):
        yy = y0 + j
        yc = np.clip(yy, 0, H - 1).astype(np.int64)
        wy_j = wy[j]
        for i in range(4):
            xx = x0 + i
            xc = np.clip(xx, 0, W - 1).astype(np.int64)
            w = wx[i] * wy_j * (xx >= 0) * (xx < W) * (yy >= 0) * (yy < H)
            # 逐像素取样：yc/xc 都是 (out_h, out_w)
            acc += w[..., None] * img[yc, xc]

    # 落在源图外（含中心对齐的半像素边界）的像素填 pad_value
    outside = (xs < -0.5) | (xs > W - 0.5) | (ys < -0.5) | (ys > H - 0.5)
    acc[outside] = float(pad_value)
    return np.clip(np.rint(acc), 0, 255).astype(rgb.dtype)


def crop_rotated(rgb: np.ndarray, tf: CropTransform,
                 pad_value: int = 0) -> np.ndarray:
    """按变换记录做转正裁剪，框外区域填 ``pad_value``（默认黑）。

    ``rgb`` 是源图局部区域 (H,W,3) uint8，坐标已减去局部原点。
    """
    # 边界保护：目标框中心可能落在数组外（边缘截断的建筑）
    local = CropTransform(tf.cx, tf.cy, tf.w, tf.h, tf.angle,
                          tf.crop_w, tf.crop_h, tf.expand)
    if tf.crop_h <= 0 or tf.crop_w <= 0:
        raise RuntimeError(f"裁剪尺寸非法：{(tf.crop_h, tf.crop_w)}")
    try:
        import cv2
    except ImportError:
        return _warp_affine_numpy(rgb, local.affine_matrix(),
                                  tf.crop_w, tf.crop_h, pad_value)

    dst = cv2.warpAffine(rgb, local.affine_matrix(), (tf.crop_w, tf.crop_h),
                         flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(pad_value, pad_value, pad_value))
    if dst.shape[0] != tf.crop_h or dst.shape[1] != tf.crop_w:
        raise RuntimeError(f"warpAffine 输出尺寸异常：{dst.shape} vs {(tf.crop_h, tf.crop_w)}")
    return dst


def offset_transform(tf: CropTransform, dx: float, dy: float) -> CropTransform:
    """裁剪坐标的局部原点平移（例如从源图子窗口读取后）。"""
    return CropTransform(tf.cx - dx, tf.cy - dy, tf.w, tf.h, tf.angle,
                         tf.crop_w, tf.crop_h, tf.expand)


# --------------------------------------------------- 源像素 → 地理（唯一 +0.5）


def source_px_to_geo(points_source_px: Sequence[Sequence[float]],
                     gt: Sequence[float], to_geo: bool = True) -> list[list[float]]:
    """源像素索引（A 约定）→ 地理坐标。

    ``to_geo=True`` 时把 A 约定转成 GDAL 的 B 约定（+0.5），这是全链路**唯一**加 0.5 的位置。
    ``to_geo=False`` 用于只想得到 B 约定的像素坐标（例如与 rasterio 的 transform 对齐）。
    """
    gt0, gt1, gt2, gt3, gt4, gt5 = [float(v) for v in gt]
    shift = 0.5 if to_geo else 0.0
    out = []
    for x, y in points_source_px:
        bx, by = float(x) + shift, float(y) + shift
        out.append([gt0 + bx * gt1 + by * gt2,
                    gt3 + bx * gt4 + by * gt5])
    return out


def window_local_to_source(points_local: Sequence[Sequence[float]],
                           xoff: int, yoff: int) -> list[list[float]]:
    """检测窗口局部像素（A 约定）→ 源像素索引（A 约定）。"""
    return [[float(x) + xoff, float(y) + yoff] for x, y in points_local]


def roundtrip_error(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """两组点集的最大欧氏距离（合成测试用）。"""
    pa = np.asarray(a, dtype=np.float64)
    pb = np.asarray(b, dtype=np.float64)
    if pa.shape != pb.shape:
        raise ValueError(f"形状不一致：{pa.shape} vs {pb.shape}")
    return float(np.max(np.linalg.norm(pa - pb, axis=1))) if len(pa) else 0.0
