"""插件侧的固定模型搭配（**不依赖 backend 包**）。

为什么要有这份镜像
------------------
插件安装在 QGIS profile 的 ``python/plugins`` 下，那里没有 ``backend`` 包，
`sys.path` 也不包含仓库根。所以插件不能 ``from backend.contracts.presets import ...``
（实测会在 QGIS 里 ImportError）。

因此：

* **权威定义**在 ``backend/contracts/presets.py``（后端用，负责展开成 job.json 字段）
* **本文件是它的镜像**，只保留面板展示与路径校验所需的最小信息
* 两者一致性由 ``tests/test_presets_parity.py`` 守住：任何一边改了 key/路径，
  测试会失败并提示同步

新增模型时**两处都要改**。这是刻意的冗余——插件必须能在没有后端代码的环境里
独立展示"有哪些可用模型、缺哪个文件"。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

#: 仓库根：插件在 <repo>/qgis_plugin/vecrot_qgis/jobs/ 下，上四级到仓库根。
#: 装在 profile 里时这个推导无意义，但那种情况下路径会由 manifest 或
#: 用户安装脚本写入的绝对路径决定（见 _repo_root()）。
_HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root() -> str:
    """确定权重所在仓库根。

    查找顺序（实测：插件装在 profile 里时前两条会失败，必须靠第三、四条）：

    1. 向上找含 ``backend/`` 的目录（开发时插件就在仓库内）
    2. 同上，找含 ``deployment/`` 的目录
    3. QGIS 设置 ``vecrot/manifest_path`` 指向的 manifest 里的 ``repo_root``
    4. 环境变量 ``VECROT_REPO_ROOT``

    都找不到时返回 ``_HERE`` —— 此时所有预设会显示为"缺文件"，
    用户能看到明确提示而不是静默失败。
    """
    cur = _HERE
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, "backend")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    cur = _HERE
    for _ in range(8):
        if os.path.isdir(os.path.join(cur, "deployment")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # 3) 从 QGIS 设置里的 manifest 副本读 repo_root
    root = _root_from_manifest_setting()
    if root:
        return root

    # 4) 环境变量兜底
    env = os.environ.get("VECROT_REPO_ROOT")
    if env and os.path.isdir(env):
        return env

    return _HERE


def _manifest_candidates() -> list[str]:
    """候选 manifest 路径，按可靠性排序。

    为什么不能只依赖 QgsSettings：``QgsSettings()`` 的存储位置取决于
    QGIS 应用名与组织名（实测在 headless + o4w_env 下会落到
    ``.../Roaming/python/profiles/default/Unknown Organization/python.ini``），
    而正常启动的 QGIS 写的是 ``.../QGIS/QGIS3/profiles/default/QGIS/QGIS3.ini``。
    两者不是同一个文件 —— 安装脚本写进前者、QGIS 读后者时就读不到。

    所以这里同时读多个来源，并对**已知部署位置**做文件系统扫描兜底。
    """
    out: list[str] = []

    # 1) 环境变量
    env = os.environ.get("VECROT_MANIFEST")
    if env:
        out.append(env)

    # 2) QgsSettings，两种作用域都试
    try:
        from qgis.core import QgsSettings

        for ctor_args in ((), ("QGIS", "QGIS3")):
            try:
                s = QgsSettings(*ctor_args)
                v = s.value("vecrot/manifest_path", "", type=str)
                if v:
                    out.append(v)
            except Exception:              # noqa: BLE001
                pass
    except Exception:                      # noqa: BLE001 - 无 QGIS
        pass

    # 3) 文件系统扫描：安装脚本放置的固定位置 + 仓库内
    home = os.path.expanduser("~")
    out += [
        os.path.join(home, "vecrot", "deployment", "manifest.json"),
        os.path.join("C:\\", "vecrot", "deployment", "manifest.json"),
        os.path.join("D:\\", "vecrot", "deployment", "manifest.json"),
    ]
    cur = _HERE
    for _ in range(8):
        out.append(os.path.join(cur, "deployment", "manifest.json"))
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return out


def _root_from_manifest_setting() -> str | None:
    """从候选 manifest 里读 ``repo_root``（第一个有效的胜出）。"""
    import json

    for mp in _manifest_candidates():
        if not mp or not os.path.isfile(mp):
            continue
        try:
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
        except (OSError, ValueError):
            continue
        root = m.get("repo_root")
        if isinstance(root, str) and root:
            cand = root if os.path.isabs(root) else os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(mp)), root))
            if os.path.isdir(cand):
                return cand
    return None


#: 当前使用的仓库根。模块 import 时先算一次，之后可由 refresh_repo_root() 更新。
_REPO = _repo_root()


def refresh_repo_root(root: str | None = None) -> str:
    """重新解析仓库根并**重建**所有预设的路径。

    为什么需要：模块级常量在 import 时求值，而 QGIS 插件 import 发生在
    manifest 定位之前。若 manifest 里有 repo_root（插件装在 profile 时的唯一
    可靠线索），必须等 manifest 定位后再解析一次，否则所有路径会拼到插件目录下
    （实战报错：全部显示"缺文件"）。

    ``root`` 传了就**直接用它**，跳过自动发现。自动发现是「向上找含 backend/
    或 deployment/ 的目录」，当本仓库被嵌进另一个同构仓库时（开发布局
    ``<dev_repo>/github_repo/`` 就是这种情况），会先命中外层，此时必须能指定。

    返回解析到的仓库根。
    """
    global _REPO, STRIP_GZDA, STRIP_GZDA_SCENE, RTDETR_ROOF, VECTOR_ONLY, PRESETS, _BY_KEY
    _REPO = os.path.normpath(root) if root else _repo_root()
    STRIP_GZDA = _make_strip_gzda()
    STRIP_GZDA_SCENE = _make_strip_gzda_scene()
    RTDETR_ROOF = _make_rtdetr()
    VECTOR_ONLY = _make_vector_only()
    PRESETS = (STRIP_GZDA_SCENE, RTDETR_ROOF, VECTOR_ONLY)
    _BY_KEY = {p.key: p for p in PRESETS}
    return _REPO


#: 生成器用的绝对路径版本（_r 依赖当前 _REPO）
def _abs(rel: str) -> str:
    return os.path.normpath(os.path.join(_REPO, rel))


@dataclass(frozen=True)
class ModelPreset:
    key: str
    label: str
    backend: str
    files: dict = field(default_factory=dict)
    num_classes: int | None = None
    note: str = ""

    def missing(self) -> list[str]:
        if not self.files:
            return ["（该预设尚未配置权重路径）"]
        out = []
        for label, path in self.files.items():
            if not path or not os.path.exists(path):
                out.append(f"{label}: {path or '（未配置）'}")
        return out

    def is_available(self) -> bool:
        return not self.missing()

    def to_job_fields(self) -> dict:
        f = self.files
        out: dict = {"detector_profile": self.backend}
        if f.get("weights"):
            out["detector_weights"] = f["weights"]
        if f.get("config"):
            out["detector_config"] = f["config"]
        if self.num_classes:
            out["detector_num_classes"] = int(self.num_classes)
        if f.get("vector_ckpt"):
            out["vector_ckpt"] = f["vector_ckpt"]
        if f.get("vector_base"):
            out["vector_base"] = f["vector_base"]
        return out


def _make_strip_gzda() -> ModelPreset:
    return ModelPreset(
        key="strip_s_gzda_sc055m",
        label="建筑专用 · Strip-R-CNN-S + ScaleCond 0.55M（旋转框，推荐）",
        backend="strip_s_gzda",
        files={
            "weights": _abs("vecrot/models/detector_strip_s/"
                            "best_mAP_iter_90288.pth"),
            "config": _abs("third_party/Strip-R-CNN/configs/"
                           "gzda/strip_s_gzda.py"),
            "vector_ckpt": _abs("vecrot/models/adaptor_sc_055m"),
            "vector_base": _abs("vecrot/models/official_vocab_635"),
        },
        num_classes=1,
        note="本仓库默认组合。检测权重 missing=0/unexpected=0；"
             "真实瓦片 R=0.67/P=0.74。",
    )


def _make_vector_only() -> ModelPreset:
    return ModelPreset(
        key="vector_only",
        label="只跑轮廓（读已有检测结果 jsonl）",
        backend="jsonl",
        files={
            "vector_ckpt": _abs("vecrot/models/adaptor_sc_055m"),
            "vector_base": _abs("vecrot/models/official_vocab_635"),
        },
        note="需要 manifest 的 models.detections_jsonl 指向窗口级检测结果。",
    )


def _make_rtdetr() -> ModelPreset:
    return ModelPreset(
        key="rtdetr_roof",
        label="卫星屋顶 · RT-DETR R101vd（水平框，单类别）",
        backend="rtdetr_hf",
        files={
            "weights": _abs("vecrot/models/detector_rt-detr_bbox"),
            "vector_ckpt": _abs("vecrot/models/adaptor_sc_055m"),
            "vector_base": _abs("vecrot/models/official_vocab_635"),
        },
        num_classes=1,
        note="HF 版 RT-DETR，1 类屋顶，固定 640×640。AP_small 仅 0.248，"
             "高分辨率影像小目标漏检偏多。",
    )


def _make_strip_gzda_scene() -> ModelPreset:
    return ModelPreset(
        key="strip_s_gzda_scene",
        label="建筑专用 · Strip-R-CNN-S + 场景分类（大范围推理推荐）",
        backend="strip_s_gzda",
        files={
            "weights": _abs("vecrot/models/detector_strip_s_scene/"
                            "best_mAP_iter_37620.pth"),
            "config": _abs("third_party/Strip-R-CNN/configs/gzda/strip_s_gzda_scene.py"),
            "vector_ckpt": _abs("vecrot/models/adaptor_sc_055m"),
            "vector_base": _abs("vecrot/models/official_vocab_635"),
        },
        num_classes=1,
        note="带 scene_head：先判窗口有无建筑，无建筑直接跳过检测与轮廓。"
             "纯背景窗口耗时降到 1/3.5。",
    )


# 初始构建（import 时）。manifest 定位后可调 refresh_repo_root() 重建。
STRIP_GZDA = _make_strip_gzda()
STRIP_GZDA_SCENE = _make_strip_gzda_scene()
VECTOR_ONLY = _make_vector_only()
RTDETR_ROOF = _make_rtdetr()
PRESETS: tuple[ModelPreset, ...] = (
    STRIP_GZDA_SCENE, RTDETR_ROOF, VECTOR_ONLY)
_BY_KEY = {p.key: p for p in PRESETS}


def available_presets(only_available: bool = True) -> list[ModelPreset]:
    """预设列表。

    默认只返回**文件齐备**的：精简分发包可能不含某些权重（例如
    ``base_v1`` 与旧检测权重），此时在面板上列出「缺文件」的选项只会让
    用户以为装坏了。真要排查缺件时用 ``only_available=False`` 看全量。
    """
    if not only_available:
        return list(PRESETS)
    return [p for p in PRESETS if p.is_available()]


def get_preset(key: str) -> ModelPreset | None:
    return _BY_KEY.get(key)


def describe() -> list[dict]:
    """给面板/自检用的可读清单（含缺件说明）。"""
    return [{
        "key": p.key,
        "label": p.label,
        "backend": p.backend,
        "available": p.is_available(),
        "missing": p.missing(),
        "num_classes": p.num_classes,
        "note": p.note,
    } for p in PRESETS]
