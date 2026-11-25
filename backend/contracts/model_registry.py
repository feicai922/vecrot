"""模型文件夹注册表：**一个文件夹 = 一个可选模型**。

## 为什么要从"代码里写死预设"改成"扫文件夹"

旧做法在 ``presets.py`` 里把每个模型的权重/配置/类别数写成 Python 常量。
它确实挡住了"换 .pth 忘了换 .py"这类错配，但也带来两个新问题：

* 加一个模型要改两处代码（backend 侧 + 插件侧镜像），还要同步测试；
* 权重和配置分散在 ``vecrot/models/`` 与 ``third_party/`` 两个地方，
  想确认"这对权重配的是哪个配置"必须去读代码。

现在改成：模型目录自己声明自己。每个模型一个文件夹，文件夹里放
``model.json``（元信息）+ 权重 + 配置，注册表启动时扫一遍生成列表。
**加模型 = 拷文件夹，不改代码**；错配依然被挡住，因为权重与配置在同一个
``model.json`` 里成对声明。

## 目录约定

```text
<repo>/vecrot/models/<任意名>/
├─ model.json                 # 必须：本文件描述的元信息
├─ best_mAP_iter_37620.pth    # 权重（.pth / .safetensors / HF 目录）
└─ strip_s_fusion.py          # 配置（mmrotate .py / rio .yml，可选）
```

``model.json`` 字段（``spec_version: 1``）::

    {
      "spec_version": 1,
      "key": "strip_s_fusion",            // 可选，默认取文件夹名
      "label": "融合数据集 · Strip-R-CNN-S（旋转框）",
      "backend": "strip_s_gzda",          // 必填，见 SUPPORTED_BACKENDS
      "num_classes": 1,                   // 可选
      "repo": "third_party/Strip-R-CNN",  // 可选：源码树（相对仓库根）
      "weights": "best_mAP_iter_37620.pth",
      "config": "repo:configs/gzda/strip_s_gzda_fusion.py",
      "detector_options": {"score_thr": 0.05, "nms_iou": 0.3},
      "vector": {"ckpt": "../adaptor_sc_055m", "base": "../official_vocab_635"},
      "order": 10,
      "enabled": true,
      "note": "面板上显示的一句话说明"
    }

路径写法（``weights`` / ``config`` / ``repo`` / ``vector.*`` 通用）：

| 写法 | 含义 |
|---|---|
| ``best.pth`` | 相对**本文件夹** |
| ``.`` | 本文件夹自身（HF 那种"目录即模型"） |
| ``repo:third_party/Strip-R-CNN/configs/x.py`` | 相对**仓库根** |
| ``src:configs/gzda/x.py`` | 相对本模型声明的 ``repo``（源码树里的配置，最常用） |
| ``models:other/best.pth`` | 相对 **models 根**（共用别的文件夹的权重） |
| ``/abs/path`` | 绝对路径（Windows 盘符路径同样适用） |

## 这个文件有两份逐字节相同的副本

插件装在 QGIS profile 里，那里没有 ``backend`` 包，所以插件必须自带一份：

* 权威：``backend/contracts/model_registry.py``
* 镜像：``vecrot_qgis/jobs/model_registry.py``

两份**内容必须完全一致**（``tests/test_registry_parity.py`` 比对文件哈希）。
本模块只用标准库，不 import numpy/backend 任何东西，否则镜像会拖垮插件加载。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

#: 每个模型文件夹里的元信息文件名
SPEC_FILENAME = "model.json"

#: 元信息格式版本（不匹配的文件夹会被跳过并给出原因，而不是猜）
SPEC_VERSION = 1

#: 可用的检测后端 → 需要的文件角色。
#: 角色名与 ``files`` / job.json 的字段一一对应（见 ``_JOB_FIELD``）。
BACKEND_REQUIRED_FILES = {
    "strip_s_gzda": ("weights", "config"),
    "strip_rcnn": ("weights", "config"),
    "mmrotate_obb": ("weights", "config"),
    "mmdet_hbb": ("weights", "config"),
    "rtdetr_hf": ("weights",),
    "rio_detr": ("weights", "config"),
    "jsonl": (),
}

#: 后端 → 面板上显示的中文类型说明
BACKEND_LABELS = {
    "strip_s_gzda": "Strip-R-CNN（MMRotate 旋转框）",
    "strip_rcnn": "Strip-R-CNN（MMRotate 旋转框）",
    "mmrotate_obb": "MMRotate 旋转框",
    "mmdet_hbb": "MMDetection 水平框",
    "rtdetr_hf": "RT-DETR（HF Transformers 水平框）",
    "rio_detr": "RiO-DETR（YAML 配置 + checkpoint 旋转框）",
    "jsonl": "只跑轮廓（读已有检测结果）",
}

#: 文件角色 → job.json 的 inference 字段名
_JOB_FIELD = {
    "weights": "detector_weights",
    "config": "detector_config",
    "repo": "detector_repo",
    "vector_ckpt": "vector_ckpt",
    "vector_base": "vector_base",
}

#: 面板上文件角色的中文名
_ROLE_LABELS = {
    "weights": "权重",
    "config": "配置",
    "repo": "源码",
    "vector_ckpt": "轮廓适配器",
    "vector_base": "轮廓基座",
}

#: 已知的**资产目录**（轮廓适配器、词表基座等）：它们也是 vecrot/models 下的子目录，
#: 但不是"检测模型文件夹"，扫描时静默忽略 —— 否则面板会提示
#: "official_vocab_635 缺少 model.json"，纯属误导。
ASSET_FOLDERS = ("adaptor_sc_055m", "official_vocab_635", "base_v1")

#: 覆盖模型根目录的环境变量（模型可以放在仓库外，避免几个 GB 的权重进 git 目录）
MODELS_ROOT_ENV = "VECROT_MODELS_ROOT"

#: 路径前缀
_REPO_PREFIX = "repo:"
_MODELS_PREFIX = "models:"
_SRC_PREFIX = "src:"


class ModelSpecError(ValueError):
    """model.json 不合法。"""


def find_repo_root(start: str | None = None) -> str:
    """向上找含 ``backend/`` 或 ``deployment/`` 的目录（仓库根）。

    这是**兜底**手段：插件侧真正可靠的线索是 manifest 里的 ``repo_root``，
    由 ``presets.refresh_repo_root()`` 显式传进来。找不到时返回 ``start``
    （此时所有路径都会显示为缺文件，用户能看到明确提示，而不是静默失败）。
    """
    cur = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    probe = cur
    for _ in range(8):
        if os.path.isdir(os.path.join(probe, "backend")) or \
                os.path.isdir(os.path.join(probe, "deployment")):
            return probe
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return cur


def default_models_root(repo_root: str | None = None) -> str:
    """默认模型根目录：``<repo>/vecrot/models``。"""
    root = repo_root or find_repo_root()
    return os.path.normpath(os.path.join(root, "vecrot", "models"))


def resolve_models_root(repo_root: str | None = None, declared: str | None = None,
                        env: dict | None = None) -> str:
    """确定模型根目录。优先级：显式声明 > 环境变量 > 默认位置。

    为什么允许换位置：模型包有几个 GB，很多人不想让它躺在 clone 出来的
    git 目录里（备份、同步、换盘都别扭）。所以三种都支持：

    * ``declared``：部署 manifest 里的 ``models_root``（绝对路径，或相对仓库根）
    * 环境变量 ``VECROT_MODELS_ROOT``
    * 默认 ``<repo>/vecrot/models``（网盘包解压到这里就零配置）
    """
    root = os.path.normpath(repo_root or find_repo_root())
    env_map = os.environ if env is None else env
    for cand in (declared, env_map.get(MODELS_ROOT_ENV)):
        if not cand or not isinstance(cand, str):
            continue
        path = cand if os.path.isabs(cand) else os.path.join(root, cand)
        if os.path.isdir(path):
            return os.path.normpath(path)
    return default_models_root(root)


@dataclass(frozen=True)
class ModelPreset:
    """一个可直接运行的模型组合（来自一个模型文件夹）。"""

    key: str
    label: str
    backend: str
    #: 文件角色 → 绝对路径（只含已配置且已解析的角色）
    files: dict = field(default_factory=dict)
    num_classes: int | None = None
    #: 面板/report 里显示的一句话说明
    note: str = ""
    #: 后端专用参数（score_thr / nms_iou / max_per_img / scene_threshold …）
    detector_options: dict = field(default_factory=dict)
    #: 来源文件夹（存疑时可回溯到磁盘）
    models_dir: str = ""
    #: 下拉排序（越小越前）与启用开关，直接来自 model.json
    order: int = 100
    enabled: bool = True
    #: 声明里就缺东西（例如没写 weights）时的错误说明
    spec_error: str = ""

    # ---- 可用性
    def missing(self) -> list[str]:
        """缺失的必需文件（空列表 = 可用）。

        ``files`` 为空同样算不可用 —— 否则占位条目会被当成能跑的模型。
        """
        out: list[str] = []
        if self.spec_error:
            out.append(self.spec_error)
        if not self.files and not self.spec_error:
            out.append("（该模型文件夹未配置权重路径）")
        for role in BACKEND_REQUIRED_FILES.get(self.backend, ()):
            path = self.files.get(role)
            if not path:
                out.append(f"{_ROLE_LABELS.get(role, role)}: （未配置）")
        for role, path in self.files.items():
            if not os.path.exists(path):
                out.append(f"{_ROLE_LABELS.get(role, role)}: {path}")
        return out

    def is_available(self) -> bool:
        return not self.missing()

    # ---- 展开成 job.json 字段
    def to_job_fields(self) -> dict:
        out: dict = {"detector_profile": self.backend}
        for role, key in _JOB_FIELD.items():
            path = self.files.get(role)
            if path:
                out[key] = path
        if self.num_classes:
            out["detector_num_classes"] = int(self.num_classes)
        if self.detector_options:
            out["detector_options"] = dict(self.detector_options)
        return out

    def describe(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "backend": self.backend,
            "backend_label": BACKEND_LABELS.get(self.backend, self.backend),
            "available": self.is_available(),
            "missing": self.missing(),
            "num_classes": self.num_classes,
            "note": self.note,
            "models_dir": self.models_dir,
            "detector_options": dict(self.detector_options),
            # 面板要显示"权重/配置/源码"这种中文角色名，集中在这里定义
            "files": {role: {"label": _ROLE_LABELS.get(role, role), "path": path}
                      for role, path in self.files.items()},
        }


# ------------------------------------------------------------------ 路径解析


def resolve_path(value, folder: str, repo_root: str, models_root: str,
                 src_root: str | None = None) -> str | None:
    """把 model.json 里的路径写法解析成绝对路径（见模块文档的表格）。

    ``src_root`` 是本模型声明的源码树目录，只有 ``src:`` 前缀会用到它；
    声明了 ``src:`` 却没声明 ``repo`` 时报错，而不是退化成相对文件夹。
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ModelSpecError(f"路径必须是字符串，实际 {type(value).__name__}")
    v = value.strip()
    if not v:
        return None
    if v.startswith(_SRC_PREFIX):
        rest = v[len(_SRC_PREFIX):].strip().lstrip("/\\")
        if not rest:
            raise ModelSpecError(f"路径前缀 {_SRC_PREFIX} 后面没有内容")
        if not src_root:
            raise ModelSpecError(
                f"路径 {value!r} 用了 {_SRC_PREFIX} 前缀，但本模型没有声明 repo（源码树）")
        return os.path.normpath(os.path.join(src_root, rest))
    if v.startswith(_REPO_PREFIX):
        rest = v[len(_REPO_PREFIX):].strip().lstrip("/\\")
        if not rest:
            raise ModelSpecError(f"路径前缀 {_REPO_PREFIX} 后面没有内容")
        return os.path.normpath(os.path.join(repo_root, rest))
    if v.startswith(_MODELS_PREFIX):
        rest = v[len(_MODELS_PREFIX):].strip().lstrip("/\\")
        if not rest:
            raise ModelSpecError(f"路径前缀 {_MODELS_PREFIX} 后面没有内容")
        return os.path.normpath(os.path.join(models_root, rest))
    if v == ".":
        return os.path.normpath(folder)
    if os.path.isabs(v):
        return os.path.normpath(v)
    return os.path.normpath(os.path.join(folder, v))


# ------------------------------------------------------------------ 读 model.json


def _as_number_or_str(value):
    return isinstance(value, (int, float, bool, str))


def _check_options(raw, where: str) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ModelSpecError(f"{where}.detector_options 必须是对象")
    out = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not k:
            raise ModelSpecError(f"{where}.detector_options 的键必须是非空字符串")
        if not _as_number_or_str(v):
            raise ModelSpecError(
                f"{where}.detector_options.{k} 只接受数字/布尔/字符串，实际 {type(v).__name__}")
        out[k] = v
    return out


def load_spec(folder: str, repo_root: str, models_root: str) -> ModelPreset:
    """读一个模型文件夹的 model.json。非法时抛 ModelSpecError。"""
    spec_path = os.path.join(folder, SPEC_FILENAME)
    if not os.path.isfile(spec_path):
        raise ModelSpecError(f"缺少 {SPEC_FILENAME}")
    try:
        with open(spec_path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as exc:
        raise ModelSpecError(f"{SPEC_FILENAME} 无法解析：{exc}") from exc
    if not isinstance(raw, dict):
        raise ModelSpecError(f"{SPEC_FILENAME} 顶层必须是对象")

    version = raw.get("spec_version", SPEC_VERSION)
    if version != SPEC_VERSION:
        raise ModelSpecError(
            f"spec_version={version!r} 不受支持（本版本只认 {SPEC_VERSION}）")

    name = os.path.basename(os.path.normpath(folder))
    key = raw.get("key") or name
    if not isinstance(key, str) or not key.strip():
        raise ModelSpecError("key 必须是非空字符串")
    key = key.strip()

    backend = raw.get("backend")
    if not isinstance(backend, str) or not backend.strip():
        raise ModelSpecError("缺少 backend（例如 strip_s_gzda / rio_detr / rtdetr_hf）")
    backend = backend.strip()
    if backend not in BACKEND_REQUIRED_FILES:
        raise ModelSpecError(
            f"未知 backend {backend!r}（可选 {sorted(BACKEND_REQUIRED_FILES)}）")

    label = raw.get("label", key)
    if not isinstance(label, str) or not label.strip():
        raise ModelSpecError("label 必须是非空字符串（不写则默认用文件夹名）")

    num_classes = raw.get("num_classes")
    if num_classes is not None:
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
            raise ModelSpecError(f"num_classes 必须是 ≥1 的整数，实际 {num_classes!r}")

    files: dict = {}
    # repo 先解析：config/weights 里的 ``src:`` 前缀要相对它
    repo_path = resolve_path(raw.get("repo"), folder, repo_root, models_root)
    if repo_path:
        files["repo"] = repo_path
    for role in ("weights", "config"):
        path = resolve_path(raw.get(role), folder, repo_root, models_root,
                            src_root=repo_path)
        if path:
            files[role] = path
    vector = raw.get("vector") or {}
    if vector and not isinstance(vector, dict):
        raise ModelSpecError("vector 必须是对象（{\"ckpt\": …, \"base\": …}）")
    for role, decl in (("vector_ckpt", vector.get("ckpt")),
                       ("vector_base", vector.get("base"))):
        path = resolve_path(decl, folder, repo_root, models_root)
        if path:
            files[role] = path

    options = _check_options(raw.get("detector_options"), SPEC_FILENAME)

    note = raw.get("note") or ""
    if not isinstance(note, str):
        raise ModelSpecError("note 必须是字符串")

    order = raw.get("order", 100)
    if isinstance(order, bool) or not isinstance(order, int):
        raise ModelSpecError(f"order 必须是整数，实际 {order!r}")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ModelSpecError(f"enabled 必须是 true/false，实际 {enabled!r}")

    spec_error = ""
    req = BACKEND_REQUIRED_FILES.get(backend, ())
    lacking = [r for r in req if r not in files]
    if lacking:
        spec_error = "model.json 缺少必需项：" + "、".join(
            _ROLE_LABELS.get(r, r) for r in lacking)

    return ModelPreset(
        key=key, label=label.strip(), backend=backend, files=files,
        num_classes=num_classes, note=note, detector_options=options,
        models_dir=os.path.normpath(folder), order=order, enabled=enabled,
        spec_error=spec_error)


def _sort_key(p: ModelPreset, folder_name: str):
    return (p.order, folder_name.lower())


# ------------------------------------------------------------------ 扫描


def scan(models_root: str, repo_root: str | None = None) -> tuple[list[ModelPreset], list[dict]]:
    """扫 ``models_root`` 下的每个子文件夹。

    返回 ``(presets, skipped)``：

    * ``presets``：按 ``order``（model.json 里的字段，缺省 100）再按文件夹名排序
    * ``skipped``：``[{"folder": …, "reason": …}]``，面板上要显示出来，
      否则用户会觉得"我把文件夹放进去了却没出现"
    """
    repo = repo_root or find_repo_root()
    models_root = os.path.normpath(models_root)
    presets: list[ModelPreset] = []
    skipped: list[dict] = []
    if not os.path.isdir(models_root):
        return presets, skipped

    ordered: list[tuple[ModelPreset, tuple]] = []
    for name in sorted(os.listdir(models_root)):
        folder = os.path.join(models_root, name)
        if not os.path.isdir(folder) or name.startswith((".", "_")):
            continue
        if name in ASSET_FOLDERS:
            continue        # 轮廓资产，不是检测模型
        try:
            preset = load_spec(folder, repo, models_root)
        except ModelSpecError as exc:
            skipped.append({"folder": folder, "reason": str(exc)})
            continue
        if not preset.enabled:
            skipped.append({"folder": folder, "reason": "model.json 里 enabled=false"})
            continue
        ordered.append((preset, _sort_key(preset, name)))

    ordered.sort(key=lambda item: item[1])
    seen: dict = {}
    for preset, _key in ordered:
        if preset.key in seen:
            # 两个文件夹声明了同一个 key：保留先出现的，后者明确报出来
            skipped.append({"folder": preset.models_dir,
                            "reason": f"key {preset.key!r} 与 {seen[preset.key]} 重复"})
            continue
        seen[preset.key] = preset.models_dir
        presets.append(preset)
    return presets, skipped


def builtin_presets(models_root: str, repo_root: str | None = None) -> list[ModelPreset]:
    """不属于"检测模型文件夹"的内置条目。

    只有一个 ``vector_only``：它没有检测权重（只跑轮廓），所以没有自己的
    模型文件夹，但它仍然要在面板上可选。它的轮廓权重指向 models 根下的
    公共目录，写法与 model.json 里的 ``models:`` 前缀一致。
    """
    root = os.path.normpath(models_root)
    ckpt = os.path.join(root, "adaptor_sc_055m")
    base = os.path.join(root, "official_vocab_635")
    files = {}
    for role, path in (("vector_ckpt", ckpt), ("vector_base", base)):
        if os.path.exists(path):
            files[role] = path
    note = "需要 manifest 的 models.detections_jsonl 指向窗口级检测结果。"
    return [ModelPreset(
        key="vector_only",
        label="只跑轮廓（读已有检测结果 jsonl）",
        backend="jsonl",
        files=files,
        note=note,
        spec_error="" if files else "（未找到轮廓权重 adaptor_sc_055m / official_vocab_635）",
    )]


def validate_folder(folder: str, repo_root: str | None = None) -> tuple[bool, list[str]]:
    """校验单个模型文件夹（给部署脚本/自检用）。"""
    repo = repo_root or find_repo_root()
    models_root = os.path.dirname(os.path.normpath(folder))
    try:
        preset = load_spec(folder, repo, models_root)
    except ModelSpecError as exc:
        return False, [str(exc)]
    miss = preset.missing()
    return not miss, miss or ["所有必需文件齐备"]


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else default_models_root()
    found, dropped = scan(root)
    print(json.dumps({
        "models_root": root,
        "count": len(found),
        "presets": [p.describe() for p in found],
        "skipped": dropped,
    }, ensure_ascii=False, indent=2))
