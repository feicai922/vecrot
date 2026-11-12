"""生成插件图标（icon.png），与 icon.svg 同一设计。

    python deployment/make_icon.py

QGIS 的插件管理器与工具栏优先用 PNG（某些构建对 SVG 的 Qt 插件缺失时显示空白），
所以两套都提供：`icon.png` 给 QGIS，`icon.svg` 留作矢量源。

设计：一个旋转检测框（橙色斜'矩形'）+ 内部建筑轮廓（蓝色半透明多边形）。
含义是「检测框 → 轮廓」，与插件的两步流程对应。
"""
from __future__ import annotations

import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN = os.path.join(REPO, "qgis_plugin", "vecrot_qgis")

#: QGIS 在 24px 工具栏与 32px 列表里显示图标，多尺寸一起打
SIZES = (16, 24, 32, 48, 64)


def draw(size: int):
    from PIL import Image, ImageDraw

    S = size * 4                      # 先大后缩，边缘更干净
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def P(x, y):
        return (x / 100 * S, y / 100 * S)

    # ---- 旋转检测框（橙色，逆时针 18°）
    ang = math.radians(-18)
    cx, cy = 42, 55
    bw, bh = 52, 40
    corners = []
    for dx, dy in ((-bw / 2, -bh / 2), (bw / 2, -bh / 2),
                   (bw / 2, bh / 2), (-bw / 2, bh / 2)):
        rx = cx + dx * math.cos(ang) - dy * math.sin(ang)
        ry = cy + dx * math.sin(ang) + dy * math.cos(ang)
        corners.append(P(rx, ry))
    d.line(corners + [corners[0]], fill=(224, 118, 42, 255), width=max(2, S // 22))

    # ---- 建筑轮廓（蓝色半透明，斜置五边形）
    poly = [(46, 24), (86, 33), (78, 66), (50, 72), (38, 44)]
    d.polygon([P(x, y) for x, y in poly], fill=(70, 150, 210, 120))
    d.line([P(x, y) for x, y in poly] + [P(*poly[0])],
           fill=(43, 107, 152, 255), width=max(2, S // 24))

    # ---- 高光块（让 16px 下也能看出"建筑"）
    hl = [(52, 32), (78, 38), (73, 58), (56, 62)]
    d.polygon([P(x, y) for x, y in hl], fill=(70, 150, 210, 220))

    return img.resize((size, size), Image.LANCZOS)


def main() -> int:
    from PIL import Image

    images = [draw(s) for s in SIZES]
    out = os.path.join(PLUGIN, "icon.png")
    # 用最大尺寸为主图，多尺寸信息通过 ICO 风格的 PNG 就够了（QGIS 会缩放）
    images[-1].save(out, "PNG")
    print(f"已写 {out}  {images[-1].size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
