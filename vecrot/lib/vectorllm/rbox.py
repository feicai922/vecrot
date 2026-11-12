"""旋转框（oriented bbox）裁剪工具：把建筑按其最小外接矩形转正、裁剪、框外填黑。

为什么要它（遥感专用）：
1. **框外填黑**：密集联排/城中村里，邻楼像素进了裁剪框是"一个 prompt 画出两栋楼"的直接来源；
   把 rbox 外的内容涂黑，等于把"邻居"这个干扰源从输入里物理删掉。
2. **转正**：把建筑的最长边对齐到图像坐标轴，消除"斜着放"带来的最优裁剪框松紧差异
   （我们实测外扩比 1.0 vs 1.2 就会让 pass@0.75 从 0.20 跳到 0.78，说明框紧度极其敏感）。

rbox 定义与 `gzda_v1/annotations/*_rbox.json` 的 `bbox` 字段一致：`[cx, cy, w, h, angle]`，
`angle` 为弧度，表示从 x 轴正方向起的最小外接矩形旋转角（范围 (-π/2, π/2]）。

坐标约定：
* 裁剪图坐标 `(u, v)`：左上角为原点；
* 反变换 `crop_to_tile()` 把裁剪图坐标映射回瓦片坐标（预测要落回瓦片算 IoU）。
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from PIL import Image, ImageDraw


def min_area_rect(poly: Sequence[Sequence[float]]) -> Tuple[float, float, float, float, float]:
    """多边形 → 最小外接旋转框 `(cx, cy, w, h, angle)`（angle ∈ (-π/2, π/2]）。"""
    n = len(poly)
    best = None
    for i in range(n):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        ang = math.atan2(y2 - y1, x2 - x1)
        c, s = math.cos(-ang), math.sin(-ang)
        rs = [(c * float(p[0]) - s * float(p[1]), s * float(p[0]) + c * float(p[1])) for p in poly]
        xs = [p[0] for p in rs]
        ys = [p[1] for p in rs]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        if best is None or w * h < best[0]:
            best = (w * h, ang, w, h, (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2)
    """用 (area, ang, w, h, cx', cy') 记录，最后把中心转回原坐标。"""
    _, ang, w, h, cxr, cyr = best
    c2, s2 = math.cos(ang), math.sin(ang)
    cx = c2 * cxr - s2 * cyr
    cy = s2 * cxr + c2 * cyr
    # 归一化角度到 (-90°, 90°]，并把 w 放成"长边"
    if w < h:
        w, h = h, w
        ang += math.pi / 2
    while ang <= -math.pi / 2:
        ang += math.pi
    while ang > math.pi / 2:
        ang -= math.pi
    return float(cx), float(cy), float(w), float(h), float(ang)


def rbox_crop(image: Image.Image, rbox, expand: float = 1.35, mask_outside: bool = True,
              mask_expand: float = 1.12, min_side: int = 8) -> Tuple[Optional[Image.Image], dict]:
    """按旋转框把图像转正并裁剪；`mask_outside=True` 时把 rbox（略放大 `mask_expand`）外涂黑。

    Returns: `(crop | None, transform)`
        `transform = {"c": (cx, cy), "angle": a, "size": (W, H), "ok": bool}`
        反变换用 `crop_to_tile()`。
    """
    cx, cy, w, h, ang = [float(v) for v in rbox]
    if min(w, h) < min_side:
        return None, {"ok": False}
    # 裁剪窗口（在转正后的图像里，以 (cx,cy) 为中心、轴对齐）
    W = int(round(w * expand))
    H = int(round(h * expand))
    W, H = max(W, min_side), max(H, min_side)
    deg = math.degrees(ang)
    # PIL 的 rotate(A) 把内容点映射到 R(-A)·p（y 向下坐标系）；实测要让 +θ 的矩形转正必须传 +deg
    rot = image.rotate(deg, resample=Image.BICUBIC, center=(cx, cy), expand=False)
    box = (int(round(cx - W / 2)), int(round(cy - H / 2)),
           int(round(cx - W / 2)) + W, int(round(cy - H / 2)) + H)
    crop = rot.crop(box)
    if crop.size != (W, H):      # 越界裁剪时 PIL 会给小图 → 补黑到目标尺寸
        pad = Image.new("RGB", (W, H), (0, 0, 0))
        pad.paste(crop, (0, 0))
        crop = pad
    if mask_outside:
        # rbox 在转正坐标系里是一个以 (cx,cy) 为中心、边长 (w,h) 的轴对齐矩形
        mw, mh = w * mask_expand, h * mask_expand
        left = cx - W / 2
        top = cy - H / 2
        cover = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(cover)
        d.rectangle([cx - mw / 2 - left, cy - mh / 2 - top,
                     cx + mw / 2 - left, cy + mh / 2 - top], fill=255)
        black = Image.new("RGB", (W, H), (0, 0, 0))
        crop = Image.composite(crop, black, cover)
    return crop, {"ok": True, "c": (cx, cy), "angle": ang, "size": (W, H)}


def fast_rbox_crop(image: Image.Image, rbox, expand: float = 1.35,
                   min_side: int = 8) -> Tuple[Optional[Image.Image], dict]:
    """`rbox_crop(..., mask_outside=False)` 的 OpenCV 窗口仿射加速版。

    只 warp 目标窗口，不旋转整幅 1024² 图；坐标 transform 与 PIL 版完全一致。
    注意 bicubic 插值实现不同，输出并非逐像素一致，正式对比须统一 crop 实现。
    """
    import cv2
    import numpy as np

    cx, cy, w, h, ang = [float(v) for v in rbox]
    if min(w, h) < min_side:
        return None, {"ok": False}
    W = max(int(round(w * expand)), min_side)
    H = max(int(round(h * expand)), min_side)
    c, s = math.cos(ang), math.sin(ang)
    # dst→src: src = R(ang) @ (u-W/2, v-H/2) + (cx,cy)。
    M = np.array([[c, -s, cx - c * W / 2.0 + s * H / 2.0],
                  [s, c, cy - s * W / 2.0 - c * H / 2.0]], dtype=np.float64)
    src = np.asarray(image.convert("RGB"))
    dst = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return Image.fromarray(dst), {"ok": True, "c": (cx, cy), "angle": ang, "size": (W, H)}


def crop_to_tile(points: Sequence[Sequence[float]], transform: dict) -> list:
    """裁剪图坐标 → 瓦片坐标（`rbox_crop` 的逆变换）。"""
    if not transform.get("ok"):
        return []
    cx, cy = transform["c"]
    ang = transform["angle"]
    W, H = transform["size"]
    c, s = math.cos(ang), math.sin(ang)
    out = []
    for u, v in points:
        du, dv = float(u) - W / 2.0, float(v) - H / 2.0
        # 裁剪图是"转正后"的图：先转回去（+ang），再加中心
        out.append((cx + c * du - s * dv, cy + s * du + c * dv))
    return out


def tile_to_crop(points, transform: dict) -> list:
    """瓦片坐标 → 裁剪图坐标（`crop_to_tile` 的正变换）。"""
    if not transform.get("ok"):
        return []
    cx, cy = transform["c"]
    ang = transform["angle"]
    W, H = transform["size"]
    c, s = math.cos(ang), math.sin(ang)
    out = []
    for x, y in points:
        dx, dy = float(x) - cx, float(y) - cy
        # 逆：crop_rel = R(-ang)·tile_rel
        out.append((W / 2.0 + c * dx + s * dy, H / 2.0 - s * dx + c * dy))
    return out


def aabb_mask_crop(image: Image.Image, box, expand: float = 1.35, mask_expand: float = 1.02):
    """**不转正**、只把轴对齐框外涂黑的版本（部署友好：检测框就是轴对齐的）。"""
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    W = int(round(w * expand))
    H = int(round(h * expand))
    left, top = int(round(cx - W / 2)), int(round(cy - H / 2))
    crop = Image.new("RGB", (W, H), (0, 0, 0))
    src = image.crop((left, top, left + W, top + H))
    crop.paste(src, (0, 0))
    mw, mh = w * mask_expand, h * mask_expand
    cover = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(cover)
    d.rectangle([cx - mw / 2 - left, cy - mh / 2 - top,
                 cx + mw / 2 - left, cy + mh / 2 - top], fill=255)
    black = Image.new("RGB", (W, H), (0, 0, 0))
    crop = Image.composite(crop, black, cover)
    return crop, {"ok": True, "c": (cx, cy), "angle": 0.0, "size": (W, H)}
