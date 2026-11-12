"""推理后处理组合：**边证据吸附（snap）+ 拓扑精修（refine）**。

两者解决的是不同的错误，串起来才算完整：
- `snap`（`src/vectorllm/snap.py`）：看**影像边缘证据**，把整条边拉回真实墙体（治"整体偏移 2~5px"）；
- `refine`（`src/vectorllm/topo.py`）：看**几何规则**，做直角化与成对共墙吸附（治"角不方、共墙不齐"）。

顺序有讲究，两种都值得评估（见 `tools/eval_snap.py --order`）：

| 顺序 | 直觉 | 风险 |
|---|---|---|
| `snap_first` | 先把形状贴到真实边缘，再把不规整的角拉正 | refine 的直角化可能把吸附结果重新拉偏 |
| `refine_first` | 先规则化，再用边缘证据吸附 | 直角化会改动顶点位置，吸附要在此基础上二次修正 |

门控在 `snap` 内部（面积/有效性/边缘支撑），refine 本身确定，所以整条链路不需要 GT、也不需要训练。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .postprocess import refine_polygons  # noqa: F401  (保持原有入口可见)


def snap_and_refine(polygons: Sequence[Sequence], image=None, snap_radius: float = 3.0,
                    min_gain: float = 0.05, mode: str = "pair", order: str = "snap_first",
                    do_snap: bool = True) -> List[List]:
    """对一个**瓦片内**的全部多边形做 snap + refine。`image` 为该瓦片原图（PIL，瓦片像素坐标）。

    `image=None` 或 `do_snap=False` 时退化成纯 refine；`mode="off"` 时退化成纯 snap。
    """
    polys: List[List] = [list(map(tuple, p)) if p else [] for p in polygons]

    def _snap(ps):
        if image is None or not do_snap:
            return ps
        from .snap import snap_polygon

        out = []
        for p in ps:
            if len(p) < 3:
                out.append(p)
                continue
            new, _info = snap_polygon(image, p, radius=snap_radius, min_gain=min_gain)
            out.append([(float(x), float(y)) for x, y in new])
        return out

    if order == "refine_first":
        return _snap(refine_polygons(polys, mode))
    if order != "snap_first":
        raise ValueError(f"未知 order={order!r}，可选 snap_first / refine_first")
    return refine_polygons(_snap(polys), mode)
