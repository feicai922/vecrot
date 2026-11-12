"""固定模型搭配（preset registry）：一组权重 + 配置 + 后端的完整组合。

## 为什么用"固定搭配"而不是自由填路径

开发文档 §2.1 与 §7.3 反复强调的约束是：**权重、配置、预处理、类别数必须成套**。
真实事故里最常见的失败不是"路径写错"，而是：

* 换了 `.pth` 没换 `.py` → 结构不匹配（`missing/unexpected` 一堆）
* 换了模型没换 `num_classes` → 分类头尺寸不符
* 换了基座没换适配器 → 词表错位，模型静默输出换行（本仓库实际踩过）

所以这里把每个可用模型写成**一个预设**：权重、配置、类别数、后端、预处理口径
全部在代码里写死。用户在面板上只选一个名字，不存在拼错的余地。

## 怎么加一个新模型（例如 RT-DETR）

在本文件底部 `PRESETS` 里加一条：

    ModelPreset(
        key="rtdetr_r50_coco",
        label="RT-DETR R50（COCO 水平框）",
        backend="mmdet_hbb",                  # 或 mmrotate_obb / strip_s_gzda
        weights=..., config=...,
        num_classes=80,
        note="...",
    )

需要的话在 `backend/detector/` 补对应的适配器。插件侧不需要改——下拉框从
``available_presets()`` 动态生成。预设里任何一项路径不存在，面板会把它显示为
不可用并给出缺哪个文件，而不是等到作业失败。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

#: 仓库根（本文件在 backend/contracts/ 下）
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _r(rel: str) -> str:
    """相对仓库根的路径转绝对路径。"""
    return os.path.normpath(os.path.join(_REPO, rel))


@dataclass(frozen=True)
class ModelPreset:
    """一个可直接运行的模型组合。"""

    key: str
    label: str
    backend: str                       # strip_s_gzda / mmrotate_obb / mmdet_hbb / jsonl
    #: 必需文件（存在性检查用），按用途命名
    files: dict = field(default_factory=dict)
    num_classes: int | None = None
    #: 会在 report 里显示的说明（来源、口径、已知限制）
    note: str = ""
    #: 后端专用参数（如 score_thr / nms_iou），留空用 manifest 默认
    detector_options: dict = field(default_factory=dict)

    def missing(self) -> list[str]:
        """返回缺失的必需文件（空列表表示可用）。

        注意 ``files`` 为空表示"这个预设还没配权重"，同样算不可用 ——
        否则占位条目会被当成能跑的模型（实测：RT-DETR 占位曾被判为 available）。
        """
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
        """展开成 job.json 的 inference 字段。"""
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


# ------------------------------------------------------------------ 预设

#: 本仓库自带的贵州建筑模型（旋转框 + ScaleCond 轮廓）
STRIP_GZDA = ModelPreset(
    key="strip_s_gzda_sc055m",
    label="建筑专用 · Strip-R-CNN-S + ScaleCond 0.55M（旋转框，推荐）",
    backend="strip_s_gzda",
    files={
        "weights": _r("vecrot/models/detector_strip_s/best_mAP_iter_90288.pth"),
        "config": _r("vecrot/models/detector_strip_s/Strip-R-CNN/configs/gzda/"
                     "strip_s_gzda.py"),
        "vector_ckpt": _r("vecrot/models/adaptor_sc_055m"),
        # 训练时的 official_vocab 缺失，用按 prune 映射重建的 635 词表基座
        "vector_base": _r("vecrot/models/official_vocab_635"),
    },
    num_classes=1,
    note="本仓库默认组合。检测权重 missing=0/unexpected=0；真实瓦片 R=0.67/P=0.74。"
         "轮廓基座由 base_v1 按 prune 映射重建（见 docs/STATUS.md）。",
    detector_options={"score_thr": 0.05, "nms_iou": 0.3, "max_per_img": 2000},
)

#: 只跑轮廓：不做检测，读用户提供的 jsonl 检测结果
VECTOR_ONLY = ModelPreset(
    key="vector_only",
    label="只跑轮廓（读已有检测结果 jsonl）",
    backend="jsonl",
    files={
        "vector_ckpt": _r("vecrot/models/adaptor_sc_055m"),
        "vector_base": _r("vecrot/models/official_vocab_635"),
    },
    note="需要 manifest 的 models.detections_jsonl 指向窗口级检测结果。",
)

#: RT-DETR（HuggingFace Transformers 版，水平框）。
#: 目录形式：model.safetensors + config.json + preprocessor_config.json。
#: 单类别（LABEL_0 = 屋顶），固定 640×640 输入，只出框无掩膜。
RTDETR_ROOF = ModelPreset(
    key="rtdetr_roof",
    label="卫星屋顶 · RT-DETR R101vd（水平框，单类别）",
    backend="rtdetr_hf",
    files={
        # RT-DETR 是"目录即模型"：把整个目录当作 weights
        "weights": _r("vecrot/models/detector_rt-detr_bbox"),
        # 轮廓仍由 VectorLLM 生成，所以需要适配器 + 基座
        "vector_ckpt": _r("vecrot/models/adaptor_sc_055m"),
        "vector_base": _r("vecrot/models/official_vocab_635"),
    },
    num_classes=1,
    note="HF Transformers 版 RT-DETR（RTDetrForObjectDetection），"
         "1 类「屋顶」，固定 640×640 输入。作者报告 AP50-95=0.434，"
         "但 AP_small 仅 0.248：高分辨率影像上小目标漏检偏多。"
         "需要用 deployment/envs/vector 作为检测环境（含 transformers+RTDetr API）。",
)


#: 带场景分类分支的检测模型：先判「本窗口有没有建筑」，无建筑直接跳过，
#: 不跑 RPN/RoI 与轮廓生成。实测纯背景窗口从 1725ms 降到 492ms（约 3.5x）。
#: 权重里 scene_head 有 6 个张量（fc1 1280->512、norm、fc2 512->2），
#: 结构按权重形状重建于 configs/gzda/scene_branch.py。
STRIP_GZDA_SCENE = ModelPreset(
    key="strip_s_gzda_scene",
    label="建筑专用 · Strip-R-CNN-S + 场景分类（大范围推理推荐）",
    backend="strip_s_gzda",
    files={
        "weights": _r("vecrot/models/detector_strip_s_scene/"
                      "best_mAP_iter_37620.pth"),
        "config": _r("vecrot/models/detector_strip_s_scene/"
                     "strip_s_gzda_scene.py"),
        "vector_ckpt": _r("vecrot/models/adaptor_sc_055m"),
        "vector_base": _r("vecrot/models/official_vocab_635"),
    },
    num_classes=1,
    note="带 scene_head。先判断窗口是否含建筑，无建筑直接跳过检测与轮廓。"
         "实测场景概率：有建筑 0.9988 / 纯背景 0.1675；纯背景窗口耗时降到 1/3.5。",
    detector_options={"score_thr": 0.05, "nms_iou": 0.3, "max_per_img": 2000,
                      "scene_threshold": 0.5},
)


PRESETS: tuple[ModelPreset, ...] = (
    STRIP_GZDA_SCENE,
    RTDETR_ROOF,
    VECTOR_ONLY,
)

_BY_KEY = {p.key: p for p in PRESETS}


def available_presets(only_available: bool = True) -> list[ModelPreset]:
    """预设列表；默认只返回文件齐备的（精简包会缺部分权重）。"""
    if not only_available:
        return list(PRESETS)
    return [p for p in PRESETS if p.is_available()]


def get_preset(key: str) -> ModelPreset | None:
    return _BY_KEY.get(key)


def describe() -> list[dict]:
    """给面板/自检用的可读清单。"""
    out = []
    for p in PRESETS:
        miss = p.missing()
        out.append({
            "key": p.key,
            "label": p.label,
            "backend": p.backend,
            "available": not miss,
            "missing": miss,
            "num_classes": p.num_classes,
            "note": p.note,
        })
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(describe(), ensure_ascii=False, indent=2))
