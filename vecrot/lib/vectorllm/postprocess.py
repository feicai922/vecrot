"""推理侧统一后处理入口：把 `topo.refine()` 包成一个"档位"概念，供各推理脚本共用。

为什么需要这一层：`topo.refine()` 有多个开关（直角化 / 成对共墙 / 全局共线），
而推理脚本（`tools/run_oracle_inference.py`、`detect_speedup/backends/vectorllm.py`）
要能一键切换、且**默认走已验证安全的档位**。把"档位名 → 参数"的映射写在一处，
避免两个调用点各写一套、慢慢漂移。

档位（`TOPOREFINE_MODES`）：

===========  ==========================  ==========================================
档位          做什么                      实测（40 张密瓦片，v2 权重）
===========  ==========================  ==========================================
``off``      原样返回                    IoU 0.8249 / p@0.75 0.808 / 共墙 1.42°
``rect``     只直角化                    IoU 0.8276 / p@0.75 0.820 / 共墙 1.04°
``pair``     **直角化 + 成对共墙吸附（默认）**  IoU 0.8276 / p@0.75 0.820 / 共墙 **0.86°**
``global``   直角化 + 全局边聚类（❌负结果）   IoU 0.6961 / p@0.75 0.606 / 共墙 0.91°
===========  ==========================  ==========================================

⚠️ **必须按瓦片成组调用**：共墙吸附要让同一瓦片内的多个多边形互相看得见；
一条队列里跨瓦片混着传，会把不同瓦片的边当邻居（几何上无意义，但白烧 CPU）。

⚠️ `global` 档位留着只作对照 —— 它会把**不相邻**建筑的近共线边也拉直，IoU 掉 13pt。
"""

from __future__ import annotations

from typing import List, Sequence

from .topo import refine as _refine

#: 可选档位（顺序即"从弱到强"，`off` 必须存在，其它可增删）
TOPOREFINE_MODES = ("off", "rect", "pair", "global")

#: 默认档位：零训练收益，核心 IoU 不掉、共墙夹角 −0.56°
DEFAULT_TOPOREFINE_MODE = "pair"


def refine_polygons(polygons: Sequence[Sequence], mode: str = DEFAULT_TOPOREFINE_MODE) -> List[List]:
    """按档位精修一组**同一瓦片内**的多边形；返回与输入等长的列表。

    `mode="off"` 时原样返回（不复制、不校验），保证可与旧口径逐位对齐。
    """
    if mode in (None, "", "off"):
        return list(polygons)
    if mode not in TOPOREFINE_MODES:            # 拼错要炸，不要静默退回 off
        raise ValueError(f"未知的 topo-refine 档位 {mode!r}，可选：{TOPOREFINE_MODES}")
    polys = [list(p) if p else [] for p in polygons]
    if mode == "rect":
        return _refine(polys, do_rectify=True, do_global=False)
    if mode == "pair":
        return _refine(polys, do_rectify=True, do_global=True)
    # global：留作对照的负结果档位。`snap_global_collinear` 只返回"够格"的多边形，
    # 长度会和输入对不上 → 只在有效子集上调，再按原位写回（弃权/失败的空多边形保持原样）。
    from .topo import rectify_right_angles, snap_global_collinear

    rected = [rectify_right_angles(p) if len(p) >= 3 else [] for p in polys]
    valid = [i for i, p in enumerate(rected) if len(p) >= 3]
    if not valid:
        return rected
    snapped = snap_global_collinear([rected[i] for i in valid])
    for i, poly in zip(valid, snapped):
        rected[i] = poly
    return rected


def add_mode_arg(parser, flag: str = "--topo-refine", default: str = DEFAULT_TOPOREFINE_MODE):
    """给 argparse 挂上统一的 `--topo-refine` 参数（调用点不必各写一遍 help 文案）。"""
    parser.add_argument(
        flag, default=default, choices=list(TOPOREFINE_MODES),
        help="拓扑后处理档位：off=不处理；rect=只直角化；pair=直角化+成对共墙吸附（默认，推荐）；"
             "global=全局共线聚类（负结果，仅对照）。**按瓦片成组生效**")
    return parser
