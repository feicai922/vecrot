"""Strip-R-CNN 检测后端（需要 detector 环境 + 固定源码）。

## 源码与定制（2026-11-10 实测核对）

```text
vecrot/models/detector_strip_s/Strip-R-CNN/     MMRotate 0.3.4 完整分支
  configs/strip_rcnn/strip_rcnn_s_fpn_1x_dota_le90.py   上游基础配置
  configs/gzda/strip_s_gzda.py                          本地训练覆盖配置
  configs/gzda/gzda_dataset.py                          GzdaRBoxDataset + GzdaConsoleHook
  mmrotate/models/{backbones/stripnet,detectors/strip_rcnn,
                   roi_heads/bbox_heads/strip_head}.py  模型实现
```

### 已核实的结构差异（gzda 覆盖相对上游 base）

| 项 | gzda 覆盖值 | 性质 |
|---|---|---|
| `roi_head.bbox_head.num_classes` | 1 | **结构**：分类头宽度 |
| `backbone.norm_cfg` | `BN`（上游 `SyncBN`） | **结构**：归一化层类型 |
| `backbone.init_cfg` | 训练机预训练骨干路径 | 训练期 |
| `train_cfg.rpn.allowed_border` | -1 | 训练期 |
| `test_cfg.rcnn.score_thr` | 0.05 | **推理** |
| `test_cfg.rcnn.nms.iou_thr` | 0.3 | **推理** |
| `test_cfg.rcnn.max_per_img` | 2000 | **推理** |

预处理（`mean=[123.675,116.28,103.53]`、`std=[58.395,57.12,57.375]`、`to_rgb=True`）
**没有**被 gzda 覆盖，因此与上游 base 一致 —— 这意味着 `rgb_to_model_array()` 的
归一化常量是对的。

**关键结论**：`gzda_dataset.py` 只做数据集读取与训练日志 Hook，**不注册任何模型模块**。
所以「检测实现缺失」这个缺口在拿到 gzda 后已闭合：权重结构 = 上游 StripRCNN +
`num_classes=1` + `BN`。

## 派生纯推理配置

不修改原训练配置（§15 第 5 条）。派生从**本地 gzda 配置**出发（更忠实于原训练），
剔除训练期字段与训练机路径，写成独立文件并记录三方 SHA256：

1. `_base_` 展开（mmcv Config.fromfile 自动处理）
2. 保留 `custom_imports`：它注册的 `GzdaRBoxDataset`/`GzdaConsoleHook` 虽然推理不用，
   但 `Config.fromfile` 需要这些模块可导入才能完成解析
3. 删除训练期字段：`data.train/val`、`optimizer`、`lr_config`、`runner`、
   `custom_hooks`、`evaluation`、`checkpoint_config`、`log_config`
4. 移除 `backbone.init_cfg`（推理不需要预训练骨干，权重由 checkpoint 提供）
5. `norm_cfg` 保持 **BN**（gzda 原本就是 BN，不是把 SyncBN 降级）

若本地 gzda 配置缺失，回退到「上游 base + 单类别覆盖」的派生路径，
并在报告里明确标注 `derivation_source: upstream_fallback`。
"""
from __future__ import annotations

import hashlib
import os
from typing import Sequence

from .backends import DetectorNotAvailable, RawBox

#: 训练期字段：派生推理配置时删除。只删这些，其它一律保留。
_TRAIN_ONLY_KEYS = ("optimizer", "lr_config", "runner", "custom_hooks",
                    "evaluation", "checkpoint_config", "log_config", "workflow",
                    "fp16", "gpu_number")

#: 本地单类别模型的推理参数（来自 configs/gzda/strip_s_gzda.py）
_INFER = {
    "num_classes": 1,
    "score_thr": 0.05,
    "nms_iou": 0.3,
    "max_per_img": 2000,
}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: 本分支的 `mmrotate/datasets/__init__.py` 导入了 `DOTAv15Dataset` / `DOTAv2Dataset`，
#: 但 `datasets/dota.py` 里只有 `DOTADataset` —— 上游把这个文件精简过却忘了同步
#: `__init__.py`。直接 `import mmrotate` 必然 ImportError。
#:
#: 我们不修改用户下载的源码（那会破坏来源可追溯性）。改为在导入前把一个
#: 空的 `mmrotate.datasets` 包装进 sys.modules，让包的 `from .datasets import *`
#: 顺利完成；之后需要的子模块（如 datasets.dota）会按正常路径加载。
_MMROTATE_IMPORT_NOTE = (
    "已绕过 mmrotate.datasets.__init__ 的缺失导出"
    "（该分支 dota.py 只有 DOTADataset，但 __init__ 仍导入 DOTAv15Dataset/DOTAv2Dataset）"
)

#: mmrotate 0.3.4 的 `multiclass_nms_rotated` 用 `torch.arange(num_classes)` 建 labels，
#: 默认落在 CPU；而 torch≥1.13 对「CPU 索引去索引 GPU 张量」开始报错。上游修法是
#: `torch.arange(num_classes, device=scores.device)`。这是本分支与 torch 2.0 的
#: 兼容问题，不是我们的代码缺陷。这里用运行期补丁覆盖该函数，不修改用户源码。
_NMS_DEVICE_FIX_NOTE = (
    "已修补 mmrotate.core.multiclass_nms_rotated 的 labels 设备问题"
    "（torch.arange 未跟随 scores.device，torch 2.0 下会 RuntimeError）"
)


def _patch_nms_rotated_device() -> list[str]:
    """让 multiclass_nms_rotated 的 labels 与输入同设备。

    做法：重新定义该模块里的函数（源码文本等价，仅把 arange 加上 device）。
    直接改源码会破坏「下载的源码保持原样」这条约束。
    """
    import sys

    import torch
    from mmcv.ops import nms_rotated

    from mmrotate.core.post_processing import bbox_nms_rotated as mod

    if getattr(mod, "_vecrot_device_fixed", False):
        return []

    def multiclass_nms_rotated(multi_bboxes,
                               multi_scores,
                               score_thr,
                               nms,
                               max_num=-1,
                               score_factors=None,
                               return_inds=False):
        """与 mmrotate 0.3.4 源码逐行等价，唯一差别：labels 的 arange 带 device。

        源码在 `bbox_nms_rotated.py`：`nms` 是 cfg 对象（用 nms.iou_thr 访问），
        不是 {type: 'nms'} 字典。
        """
        num_classes = multi_scores.size(1) - 1
        if multi_bboxes.shape[1] > 5:
            bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
        else:
            bboxes = multi_bboxes[:, None].expand(
                multi_scores.size(0), num_classes, 5)
        scores = multi_scores[:, :-1]

        # ← 唯一修正：device=scores.device
        labels = torch.arange(num_classes, dtype=torch.long,
                              device=scores.device)
        labels = labels.view(1, -1).expand_as(scores)
        bboxes = bboxes.reshape(-1, 5)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        valid_mask = scores > score_thr
        if score_factors is not None:
            score_factors = score_factors.view(-1, 1).expand(
                multi_scores.size(0), num_classes).reshape(-1)
            scores = scores * score_factors

        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

        if bboxes.numel() == 0:
            dets = torch.cat([bboxes, scores[:, None]], -1)
            return (dets, labels, inds) if return_inds else (dets, labels)

        max_coordinate = bboxes[:, :2].max() + bboxes[:, 2:4].max()
        offsets = labels.to(bboxes) * (max_coordinate + 1)
        if bboxes.size(-1) == 5:
            bboxes_for_nms = bboxes.clone()
            bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
        else:
            bboxes_for_nms = bboxes + offsets[:, None]
        _, keep = nms_rotated(bboxes_for_nms, scores, nms.iou_thr)

        if max_num > 0:
            keep = keep[:max_num]

        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        dets = torch.cat([bboxes, scores[:, None]], 1)
        return (dets, labels, keep) if return_inds else (dets, labels)

    mod.multiclass_nms_rotated = multiclass_nms_rotated
    mod._vecrot_device_fixed = True

    # 已 import 了旧函数对象的模块要一并替换
    try:
        from mmrotate.models.roi_heads.bbox_heads import rotated_bbox_head

        if getattr(rotated_bbox_head, "multiclass_nms_rotated", None) is not None:
            rotated_bbox_head.multiclass_nms_rotated = multiclass_nms_rotated
    except Exception:                          # noqa: BLE001
        pass
    for name in ("mmrotate.models.roi_heads.bbox_heads.strip_head",):
        m = sys.modules.get(name)
        if m is not None and hasattr(m, "multiclass_nms_rotated"):
            m.multiclass_nms_rotated = multiclass_nms_rotated

    return [_NMS_DEVICE_FIX_NOTE]


def _mmrotate_datasets_dir(repo_dir: str | None) -> str | None:
    """找到 mmrotate/datasets 的实体目录。

    两种部署形态都要支持：
    * **源码目录**（开发/本机）：``<repo>/mmrotate/datasets``
    * **site-packages**（分发包或 ``pip install .`` 后）：环境里已装好，
      位置由 importlib 决定，不能假设和 repo_dir 同层。

    返回 None 表示找不到（此时不做桩，让 import 自然失败并报出真实原因）。
    """
    if repo_dir:
        cand = os.path.join(repo_dir, "mmrotate", "datasets")
        if os.path.isdir(cand):
            return cand
    # 已安装形态：用 importlib 的路径查找，不 import 包本身（那会触发失败的 __init__）
    try:
        import importlib.util

        spec = importlib.util.find_spec("mmrotate")
        if spec and spec.submodule_search_locations:
            for base in spec.submodule_search_locations:
                cand = os.path.join(base, "datasets")
                if os.path.isdir(cand):
                    return cand
    except (ImportError, ValueError):
        pass
    return None


def import_mmrotate(repo_dir: str | None) -> tuple[object, list[str]]:
    """导入 mmrotate，绕过本分支 datasets/__init__ 的缺失导出。

    返回 (mmrotate 模块, 已应用的兼容补丁说明列表)。

    顺序很重要：必须在任何 `import mmrotate` 之前预置 sys.modules 桩，
    否则包的 __init__ 会先执行并失败。

    ``repo_dir`` 可为 None（site-packages 布局），此时靠 importlib 定位。
    """
    import sys
    import types

    if repo_dir and repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    applied: list[str] = []
    if "mmrotate.datasets" not in sys.modules:
        import mmcv  # noqa: F401 - mmrotate.__init__ 依赖它做版本断言
        import mmdet  # noqa: F401

        pkg_dir = _mmrotate_datasets_dir(repo_dir)
        if pkg_dir is None:
            # 找不到就不打桩，让 import 失败并暴露真实错误
            import mmrotate as _m
            return _m, applied
        stub = types.ModuleType("mmrotate.datasets")
        stub.__path__ = [pkg_dir]
        stub.__package__ = "mmrotate.datasets"
        stub.__all__ = []
        sys.modules["mmrotate.datasets"] = stub
        applied.append(_MMROTATE_IMPORT_NOTE)

    import mmrotate

    # 把真正需要的子模块挂回桩上，行为与正常包一致
    stub = sys.modules.get("mmrotate.datasets")
    if stub is not None and not hasattr(stub, "DOTADataset"):
        from mmrotate.datasets import dota as _dota

        stub.DOTADataset = _dota.DOTADataset
        stub.__all__ = ["DOTADataset"]

    # 旋转 NMS 的设备兼容补丁（torch 2.0 + mmrotate 0.3.4）
    applied.extend(_patch_nms_rotated_device())

    return mmrotate, applied


class StripRcnnDetector:
    """把 Strip-R-CNN 权重包成 DetectorBackend。"""

    name = "strip_s_gzda"

    def __init__(self, options: dict):
        self.repo_dir = options.get("strip_rcnn_repo") or ""
        self.weights = options.get("detector_weights") or ""
        self.local_config = options.get("detector_config") or ""
        self.device = options.get("device") or "cuda:0"
        self.score_thr = float(options.get("score_thr", _INFER["score_thr"]))
        self.nms_iou = float(options.get("nms_iou", _INFER["nms_iou"]))
        self.max_per_img = int(options.get("max_per_img", _INFER["max_per_img"]))
        self.derived_dir = options.get("derived_dir") or ""
        #: 场景分类阈值：低于它直接判为「无建筑」，跳过 RPN/RoI/轮廓。
        #: 只在模型带 scene_head 时生效（见 StripRCNNScene）。
        self.scene_threshold = float(options.get("scene_threshold", 0.5))
        self._model = None
        self._cfg = None
        #: 最近一次推理的场景统计（供 report 汇总）
        self.scene_stats = {"has_scene_head": False, "skipped_windows": 0,
                            "evaluated_windows": 0}

    # ---------------------------------------------------------- 前置检查

    def _upstream_base_config(self) -> str:
        return os.path.join(self.repo_dir, "configs", "strip_rcnn",
                            "strip_rcnn_s_fpn_1x_dota_le90.py")

    def _gzda_config(self) -> str:
        """本地 gzda 覆盖配置：优先用 manifest 指定的，否则用仓库内的约定位置。"""
        if self.local_config and os.path.isfile(self.local_config):
            return self.local_config
        cand = os.path.join(self.repo_dir, "configs", "gzda", "strip_s_gzda.py")
        return cand if os.path.isfile(cand) else ""

    def _check_prereqs(self) -> None:
        if not self.repo_dir or not os.path.isdir(self.repo_dir):
            raise DetectorNotAvailable(
                "DETECTOR_REPO_MISSING",
                "未配置 Strip-R-CNN 源码目录（manifest models.detector.repo）；"
                "期望位于 vecrot/models/detector_strip_s/Strip-R-CNN")
        if not os.path.isfile(self._upstream_base_config()):
            raise DetectorNotAvailable(
                "DETECTOR_CONFIG_MISSING",
                f"上游基础配置不存在：{self._upstream_base_config()}；"
                "请确认 clone 的是 YXB-NKU/Strip-R-CNN")

        # 三个模型实现必须都在，否则权重加载必然对不上
        missing_mods = []
        for mod in ("stripnet", "strip_rcnn", "strip_head"):
            found = False
            for root, _dirs, names in os.walk(os.path.join(self.repo_dir, "mmrotate")):
                if f"{mod}.py" in names:
                    found = True
                    break
            if not found:
                missing_mods.append(f"{mod}.py")
        if missing_mods:
            raise DetectorNotAvailable(
                "DETECTOR_SOURCE_INCOMPLETE",
                f"源码中找不到 {missing_mods}；这不是完整的 Strip-R-CNN 分支")

        if not self.weights or not os.path.isfile(self.weights):
            raise DetectorNotAvailable("DETECTOR_WEIGHTS_MISSING",
                                       f"检测权重不存在：{self.weights!r}")
        try:
            import torch        # noqa: F401
            import mmcv         # noqa: F401
            import mmdet        # noqa: F401
        except ImportError as exc:
            raise DetectorNotAvailable(
                "DETECTOR_ENV_INCOMPLETE",
                f"当前解释器不是 detector 环境（缺 {exc.name}）；"
                "请用 deployment manifest 里 environments.detector 的 python.exe") from exc

        import mmdet

        major = int(str(mmdet.__version__).split(".")[0])
        if major >= 3:
            raise DetectorNotAvailable(
                "DETECTOR_MMDET_INCOMPATIBLE",
                f"检测到 mmdet {mmdet.__version__}；Strip-R-CNN 分支要求 mmdet 2.x"
                "（上游 runtime.txt: mmdet>=2.25.1,<3.0.0）")

    # ---------------------------------------------------------- 派生配置

    def _derive_config(self) -> tuple[object, dict]:
        import sys

        from mmcv import Config

        gzda_cfg = self._gzda_config()
        source = "local_gzda" if gzda_cfg else "upstream_fallback"
        cfg_path = gzda_cfg or self._upstream_base_config()

        # custom_imports 需要 configs.gzda.gzda_dataset 能被导入，源码根必须在 sys.path
        if self.repo_dir not in sys.path:
            sys.path.insert(0, self.repo_dir)

        notes = {
            "derivation_source": source,
            "source_config": cfg_path,
            "source_config_sha256": _sha256(cfg_path),
            "upstream_base_config": self._upstream_base_config(),
            "upstream_base_sha256": _sha256(self._upstream_base_config()),
            "removed_keys": [],
            "overrides": {},
        }
        if gzda_cfg:
            ds = os.path.join(os.path.dirname(gzda_cfg), "gzda_dataset.py")
            if os.path.isfile(ds):
                notes["gzda_dataset_sha256"] = _sha256(ds)

        try:
            cfg = Config.fromfile(cfg_path)
        except Exception as exc:                   # noqa: BLE001
            if source == "local_gzda":
                raise DetectorNotAvailable(
                    "DETECTOR_GZDA_UNIMPORTABLE",
                    f"本地 gzda 配置无法解析：{exc}。常见原因：custom_imports 里的 "
                    "configs.gzda.gzda_dataset 不在 sys.path，或 mmrotate 未安装") from exc
            raise

        for key in _TRAIN_ONLY_KEYS:
            if key in cfg:
                cfg.pop(key)
                notes["removed_keys"].append(key)

        data = cfg.get("data") or {}
        for key in ("train", "val", "samples_per_gpu", "workers_per_gpu"):
            if key in data:
                data.pop(key)
                notes["removed_keys"].append(f"data.{key}")
        if "test" in data:
            # test 的 ann_file/img_prefix 只用于离线评测；推理直接喂数组
            data["test"] = {k: v for k, v in data["test"].items()
                            if k not in ("ann_file", "img_prefix")}
        cfg.data = data

        # 推理必需的结构与阈值：以本地 gzda 的值为准，不依赖配置里是否写全
        cfg.model.roi_head.bbox_head.num_classes = _INFER["num_classes"]
        cfg.model.roi_head.bbox_head.pop("init_cfg", None)
        cfg.model.backbone.pop("init_cfg", None)     # 训练机预训练骨干路径，推理不需要
        # gzda 原本就是 BN（不是把上游 SyncBN 降级），这里保持；
        # 若走了 upstream_fallback 才需要真正替换 SyncBN
        norm = (cfg.model.backbone.get("norm_cfg") or {}).get("type")
        if norm == "SyncBN":
            cfg.model.backbone.norm_cfg = dict(type="BN", requires_grad=False)
            notes["overrides"]["norm_cfg"] = "SyncBN -> BN (fallback)"
        else:
            notes["overrides"]["norm_cfg"] = f"kept {norm}"
        cfg.model.backbone.norm_cfg["requires_grad"] = False

        cfg.model.test_cfg = dict(
            rpn=dict(nms_pre=2000, max_per_img=2000,
                     nms=dict(type="nms", iou_threshold=0.8), min_bbox_size=0),
            rcnn=dict(nms_pre=2000, min_bbox_size=0,
                      score_thr=self.score_thr,
                      nms=dict(iou_thr=self.nms_iou),
                      max_per_img=self.max_per_img))
        notes["overrides"].update({
            "num_classes": _INFER["num_classes"],
            "score_thr": self.score_thr,
            "nms_iou": self.nms_iou,
            "max_per_img": self.max_per_img,
        })

        derived_path = ""
        if self.derived_dir:
            os.makedirs(self.derived_dir, exist_ok=True)
            derived_path = os.path.join(self.derived_dir, "detector_infer.py")
            cfg.dump(derived_path)
            notes["derived_config"] = derived_path
            notes["derived_config_sha256"] = _sha256(derived_path)
        return cfg, notes

    # ---------------------------------------------------------- 加载

    def prepare(self) -> dict:
        self._check_prereqs()
        import sys

        from mmcv.runner import load_checkpoint
        from mmdet.models import build_detector

        mmrotate, import_patches = import_mmrotate(self.repo_dir)

        cfg, notes = self._derive_config()
        # 把用户给的场景阈值写进模型配置，让 StripRCNNScene 在 simple_test 里用
        if "scene_head" in (cfg.model or {}):
            sh = dict(cfg.model["scene_head"] or {})
            sh["score_threshold"] = self.scene_threshold
            cfg.model["scene_head"] = sh
            self.scene_stats["has_scene_head"] = True
        notes["import_patches"] = import_patches
        # mmdet 2.x 的 build_detector 不允许 test_cfg 同时出现在 cfg.model 内部
        # 与外层参数里（会 assert 失败）。只保留 cfg.model.test_cfg，
        # 且必须是 ConfigDict（TwoStageDetector 用 test_cfg.rpn 属性访问）。
        from mmcv import ConfigDict

        cfg.model.test_cfg = ConfigDict(cfg.model.test_cfg)
        model = build_detector(cfg.model)
        load_report = self._load_weights(model, load_checkpoint)
        model.to(self.device)
        model.eval()
        self._model = model
        self._cfg = cfg

        return {
            "backend": self.name,
            "repo_dir": self.repo_dir,
            "mmrotate_version": getattr(mmrotate, "__version__", "?"),
            "weights": self.weights,
            "weights_sha256": _sha256(self.weights),
            "device": self.device,
            "num_classes": _INFER["num_classes"],
            "config_derivation": notes,
            "weight_loading": load_report,
            "scene_branch": {
                "enabled": bool(self.scene_stats.get("has_scene_head")),
                "threshold": self.scene_threshold,
            },
            "verified_on_this_machine": False,
        }

    def _load_weights(self, model, load_checkpoint) -> dict:
        """加载权重并**报告**不匹配，绝不静默忽略（§2.1 第 6 条 / §7.3）。"""
        result = load_checkpoint(model, self.weights, map_location="cpu")
        missing = list(getattr(result, "missing_keys", None) or [])
        unexpected = list(getattr(result, "unexpected_keys", None) or [])
        # 推理期允许的唯一「不匹配」是训练专用的辅助结构（采样器/分配器没有权重）
        benign = ("loss", "assigner", "sampler")
        serious_missing = [k for k in missing if not any(t in k.lower() for t in benign)]
        if serious_missing:
            raise DetectorNotAvailable(
                "DETECTOR_WEIGHTS_INCOMPATIBLE",
                f"权重有 {len(serious_missing)} 个结构未匹配（示例 {serious_missing[:3]}）；"
                "通常意味着派生配置与原训练的网络结构不一致，不能用忽略加载错误的方式跑通")
        return {
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "serious_missing": serious_missing,
            "matched_all_structural": not serious_missing,
        }

    # ---------------------------------------------------------- 推理

    def predict_windows(self, windows: Sequence[dict]) -> list[list[RawBox]]:
        if self._model is None:
            raise DetectorNotAvailable("DETECTOR_NOT_PREPARED", "prepare() 未成功调用")
        return [self._predict_one(w) for w in windows]

    def _predict_one(self, w: dict) -> list[RawBox]:
        import numpy as np
        import torch

        from .backends import rgb_to_model_array

        rgb = w["rgb"]
        arr = rgb_to_model_array(rgb)                 # (3,H,W) float32, RGB, 已归一化
        h, w_ = arr.shape[1], arr.shape[2]
        img_meta = dict(
            img_shape=(h, w_, 3), ori_shape=(h, w_, 3), pad_shape=(h, w_, 3),
            scale_factor=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            flip=False, filename=str(w.get("window_id", "")),
        )
        # mmdet 的 simple_test 收到的是 img[0]，即带 batch 维的 (1,3,H,W)。
        # 只给 (3,H,W) 会让 StripNet 的 PatchEmbed 在 `_,_,H,W = x.shape` 处崩。
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            result = self._model(return_loss=False,
                                 img=[tensor],
                                 img_metas=[[img_meta]],
                                 rescale=False)

        # 场景分支：带 scene_head 的模型会把判定结果写回 img_meta。
        # 低于阈值时模型内部已直接返回空结果（不跑 RPN/RoI），
        # 这里只做统计，供 report 汇总「跳过了多少无建筑窗口」。
        if "scene_prob" in img_meta:
            self.scene_stats["has_scene_head"] = True
            self.scene_stats["evaluated_windows"] += 1
            if not img_meta.get("scene_has_building", True):
                self.scene_stats["skipped_windows"] += 1
                self.scene_stats.setdefault("probs", []).append(
                    round(float(img_meta["scene_prob"]), 4))
                return []
        # rbbox2result 返回的是**按类别分组**的列表：result[0] 是单张图，
        # 形如 [class0_boxes(N×6), class1_boxes(...), ...]。
        # 旧写法把 result[0] 直接当 N×6，遇到 (1,9,6) 这种嵌套形状会因
        # ndim!=2 被整批跳过 —— 这正是「真实影像 0 框」的直接原因。
        per_image = result[0] if isinstance(result, (list, tuple)) else result
        out: list[RawBox] = []
        for per_class in per_image:
            arr_boxes = np.asarray(per_class)
            if arr_boxes.size == 0:
                continue
            if arr_boxes.ndim == 3:          # 多一层 batch 维时先摊平
                arr_boxes = arr_boxes.reshape(-1, arr_boxes.shape[-1])
            if arr_boxes.ndim != 2 or arr_boxes.shape[1] < 6:
                continue
            for row in arr_boxes:
                cx, cy, bw, bh, ang, score = [float(v) for v in row[:6]]
                if score < self.score_thr:
                    continue
                out.append(RawBox(cx, cy, bw, bh, ang, score))
        return out

    def close(self) -> None:
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                          # noqa: BLE001
            pass
