"""VectorLLM 后端接口与内置实现（开发文档 §7.3）。

状态必须完整保留：vecrot 的 ``predict()`` 返回 ``(polygon, abstained, z)``，
旧入口只取 ``r[0]`` 会丢掉 abstained 信息。这里逐实例返回状态与原因。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


class VectorizeNotAvailable(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass
class PolygonResult:
    """单个 det 的生成结果。

    status: ``ok`` | ``abstained`` | ``empty`` | ``error`` | ``skipped``
    """

    status: str
    polygon_crop_px: list[list[float]] | None = None
    polygon_source_px: list[list[float]] | None = None
    abstain_z: float | None = None
    reason: str = ""
    elapsed_s: float = 0.0
    extra: dict = field(default_factory=dict)


class VectorizerBackend(Protocol):
    name: str

    def prepare(self) -> dict:
        ...

    def predict(self, tasks: Sequence[dict]) -> list[PolygonResult]:
        """tasks 每项: {det_id, crop_rgb(H,W,3 uint8), crop_meta{...}}。"""
        ...

    def close(self) -> None:
        ...


def _read_json(path: str) -> dict | None:
    import json

    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _safetensors_embed_rows(base_dir: str) -> int | None:
    """读基座 safetensors 里 embed_tokens 的行数，不加载整个模型。

    这是判断「基座词表规模是否与适配器 tokenizer 匹配」的唯一可靠办法：
    基座 config 里的 vocab_size 可能被改写，但权重张量的实际行数骗不了人。
    """
    try:
        from safetensors import safe_open

        path = os.path.join(base_dir, "model.safetensors")
        if not os.path.isfile(path):
            return None
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith("embed_tokens.weight"):
                    return int(f.get_slice(key).get_shape()[0])
        return None
    except Exception:                          # noqa: BLE001 - 读不到就跳过这项判断
        return None


# --------------------------------------------------------- vecrot 真后端




# --------------------------------------------------------- vecrot 真后端


class VecrotBackend:
    """封装 ``vecrot/lib/vectorizer.py`` 的 ``Vectorizer``。

    必须显式传入 base 目录：``Vectorizer`` 的默认目录推导被判为不可靠（§2.1 第 3 条），
    ``backbone.json`` 里的路径是 Linux 训练机路径，不能直接用。
    """

    name = "vecrot"

    def __init__(self, options: dict):
        self.vecrot_dir = options.get("vecrot_dir") or ""
        self.ckpt = options.get("vector_ckpt") or ""
        self.base = options.get("vector_base") or ""
        self.device = options.get("device") or "cuda:0"
        self.batch = int(options.get("vector_batch", 4))
        #: 是否启用尺度条件；None=按 arch.json 自动判定
        self.scale_cond = options.get("scale_cond")
        self.tau = options.get("tau")
        self.max_retries = int(options.get("max_retries", 2))
        self._vec = None
        self._scale_cfg = None

    def _check_prereqs(self) -> None:
        if not self.vecrot_dir or not os.path.isdir(self.vecrot_dir):
            raise VectorizeNotAvailable("VECTOR_REPO_MISSING",
                                        f"vecrot 目录不存在：{self.vecrot_dir!r}")
        if not self.ckpt or not os.path.isdir(self.ckpt):
            raise VectorizeNotAvailable("VECTOR_CKPT_MISSING",
                                        f"适配器目录不存在：{self.ckpt!r}")
        if not self.base or not os.path.isdir(self.base):
            raise VectorizeNotAvailable(
                "VECTOR_BASE_MISSING",
                f"基座目录不存在：{self.base!r}；必须显式指向经验证的 base_v1，"
                "不允许从 backbone.json 的训练机路径推断（见开发文档 §2.1）")
        try:
            import torch      # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise VectorizeNotAvailable(
                "VECTOR_ENV_INCOMPLETE",
                f"当前解释器不是 vector 环境（缺 {exc.name}）；"
                "请用 deployment manifest 里 vecrot-vllm 的 python.exe") from exc
        if not os.path.isfile(os.path.join(self.base, "config.json")):
            raise VectorizeNotAvailable("VECTOR_BASE_INCOMPLETE",
                                        f"基座目录缺 config.json：{self.base}")

    def _verify_vocab(self) -> dict:
        """核对 tokenizer / embedding / pixel_idx 三者是否**语义一致**。

        这不是「长度相等」就够的检查。2026-11-10 实机验证发现的陷阱：

        ``adaptor_sc_055m/backbone.json`` 指向训练时的
        ``/data1/wzh26/vectorllm_v2/work_dirs/vocab_prune/official_vocab`` ——
        一个已按 prune 映射**重排过**的 635 词表基座。该目录不在本仓库。

        用仓库内的 ``base_v1``（原始 Qwen3 全词表 151926 行）建模型、再
        ``resize_token_embeddings(635)``，只会**截断取前 635 行**：
        tokenizer 说 ``<pixel>`` 是 377，但 377 号 embedding 行是原词表第 377 个
        普通 BPE token 的权重，与训练时那一行完全不是同一个东西。

        实测症状：模型不报错、能正常前向，但**输出 640 个换行、0 个坐标 token**。
        这种「静默产出垃圾」正是 §2.1 第 4 条与 §7.3 明令禁止的情形，
        所以这里必须显式失败，不能跑出看似成功的空结果。
        """
        import json

        emb_rows = int(self._vec.model.get_input_embeddings().weight.shape[0])
        tok_len = len(self._vec.tok)
        pix_id = self._vec.tok.convert_tokens_to_ids("<pixel>")
        pix_model = int(getattr(self._vec.model, "pixel_idx", -1))
        arch = _read_json(os.path.join(self.ckpt, "arch.json")) or {}
        declared_base = (_read_json(os.path.join(self.ckpt, "backbone.json")) or {}).get("base")
        base_rows = _safetensors_embed_rows(self.base)

        report = {
            "tokenizer_len": tok_len,
            "embedding_rows": emb_rows,
            "base_embedding_rows": base_rows,
            "pixel_idx_tokenizer": pix_id,
            "pixel_idx_model": pix_model,
            "declared_base": declared_base,
            "actual_base": self.base,
            "arch": arch,
            "scale_cond_requested": bool(arch.get("scale_cond", False)),
        }

        if not isinstance(pix_id, int) or pix_id < 0:
            raise VectorizeNotAvailable(
                "VECTOR_VOCAB_MISMATCH",
                "tokenizer 里没有 <pixel>（pixel_idx=%r）" % (pix_id,))
        if tok_len != emb_rows:
            raise VectorizeNotAvailable(
                "VECTOR_VOCAB_MISMATCH",
                "词表长度不一致：tokenizer=%d embedding=%d" % (tok_len, emb_rows))
        if pix_id != pix_model:
            raise VectorizeNotAvailable(
                "VECTOR_VOCAB_MISMATCH",
                "pixel_idx 不一致：tokenizer=%d model=%d" % (pix_id, pix_model))

        if base_rows is not None and base_rows != tok_len:
            raise VectorizeNotAvailable(
                "VECTOR_BASE_VOCAB_MISMATCH",
                ("基座 %s 的 embedding 有 %d 行，但适配器 tokenizer 只有 %d 个 token。"
                 "该基座不是训练时用的 pruned 词表版本（backbone.json 声明的底座是 %r）。"
                 "用大词表基座加 resize 截断会让 <xN>/<yN>/<pixel> 的 embedding 行错位："
                 "模型仍能前向，但只会输出换行（已实测 640 token 全为换行符）。"
                 "必须找回训练时的 635 词表基座（official_vocab），"
                 "或按 prune 映射重排权重后另存为新的基座目录。")
                % (self.base, base_rows, tok_len, declared_base))

        report["vocab_consistent"] = True
        if arch.get("scale_cond") and self.scale_cond is False:
            raise VectorizeNotAvailable(
                "MODEL_INPUT_INCOMPATIBLE",
                "ckpt 声明 scale_cond=true，但配置禁止尺度输入；"
                "按 §7.3 不得静默退回基座行为")
        return report

    def prepare(self) -> dict:
        self._check_prereqs()
        import sys

        lib = os.path.join(self.vecrot_dir, "lib")
        if lib not in sys.path:
            sys.path.insert(0, lib)
        from vectorizer import Vectorizer        # noqa: E402  - vecrot 的封装

        vec = Vectorizer(self.ckpt, self.device, tau=self.tau, base=self.base)
        self._vec = vec
        info = self._verify_vocab()
        info.update({
            "backend": self.name,
            "vecrot_dir": self.vecrot_dir,
            "ckpt": self.ckpt,
            "base": self.base,
            "device": self.device,
            "batch": self.batch,
            "need_scale": bool(vec.need_scale),
            "hr_size": vec.hr_size,
            "zoom": bool(vec.zoom),
            "verified_on_this_machine": False,
        })
        return info

    def predict(self, tasks: Sequence[dict]) -> list[PolygonResult]:
        if self._vec is None:
            raise VectorizeNotAvailable("VECTOR_NOT_PREPARED", "prepare() 未成功调用")
        out: list[PolygonResult] = []
        for s in range(0, len(tasks), max(self.batch, 1)):
            out.extend(self._predict_with_retry(tasks[s:s + max(self.batch, 1)]))
        return out

    def _predict_with_retry(self, chunk: Sequence[dict]) -> list[PolygonResult]:
        """OOM 时减半重试，超过上限就逐条标记 error（§4.2：有限次重试，不无限循环）。"""
        max_retries = max(int(self.max_retries), 0)
        attempts = 0
        current = list(chunk)
        while current:
            try:
                return self._predict_chunk(current, attempts)
            except _OOM:
                if attempts >= max_retries:
                    return [PolygonResult("error",
                                          reason=f"OOM after {attempts} retries")
                            for _ in current]
                attempts += 1
                if len(current) == 1:
                    return [PolygonResult("error", reason="OOM on single item")]
                half = max(len(current) // 2, 1)
                failed = self._predict_with_retry(current[:half])
                rest = self._predict_with_retry(current[half:])
                return failed + rest
        return []

    def _predict_chunk(self, chunk: Sequence[dict], attempt: int = 0) -> list[PolygonResult]:
        import time

        from PIL import Image

        crops = [Image.fromarray(np.asarray(t["crop_rgb"], dtype=np.uint8)) for t in chunk]
        scale_codes = None
        if self._vec.need_scale:
            scale_codes = self._build_scale_codes(chunk)
        t0 = time.perf_counter()
        try:
            raw = self._vec.predict(crops, batch=len(crops), scale_codes=scale_codes)
        except RuntimeError as exc:
            msg = str(exc)
            if "out of memory" in msg.lower():
                raise _OOM(msg) from exc
            return [PolygonResult("error", reason=f"RuntimeError:{msg[:200]}") for _ in chunk]
        elapsed = (time.perf_counter() - t0) / max(len(chunk), 1)

        results: list[PolygonResult] = []
        for (poly, abstained, z), task in zip(raw, chunk):
            if abstained:
                results.append(PolygonResult("abstained", abstain_z=_f(z),
                                             reason="no_object", elapsed_s=elapsed))
            elif not poly:
                results.append(PolygonResult("empty", abstain_z=_f(z),
                                             reason="no_polygon", elapsed_s=elapsed))
            else:
                results.append(PolygonResult(
                    "ok",
                    polygon_crop_px=[[float(x), float(y)] for x, y in poly],
                    abstain_z=_f(z), elapsed_s=elapsed))
        if len(results) != len(chunk):
            raise VectorizeNotAvailable(
                "VECTOR_RESULT_COUNT_MISMATCH",
                f"模型返回 {len(results)} 条结果但有 {len(chunk)} 个输入")
        return results

    def _build_scale_codes(self, chunk: Sequence[dict]):
        """按训练口径构造 scale_code。

        训练侧 ``scale_code(obj_box, crop_box)`` 的输入是**轴对齐矩形**，且训练时目标框是
        标注框在「转正后裁剪图」里的位置。这里用同一个函数、同一个坐标系（裁剪图局部像素），
        目标框取「检测 OBB 转正后落在裁剪图内的那个轴对齐框」—— 即 crop_meta 里由 worker
        算好的 ``obj_box_in_crop``，而不是 OBB 在源图上的轴对齐包络（§7.3 明确禁止）。
        """
        import torch

        from vectorllm.scale_edge import scale_code

        codes = []
        for t in chunk:
            meta = t.get("crop_meta") or {}
            obj_box = meta.get("obj_box_in_crop")
            crop_box = meta.get("crop_box_in_crop")
            if obj_box is None or crop_box is None:
                raise VectorizeNotAvailable(
                    "MODEL_INPUT_INCOMPATIBLE",
                    f"det {t.get('det_id')} 缺少裁剪坐标系下的目标框/裁剪框，"
                    "无法生成与训练一致的尺度编码")
            codes.append(scale_code(obj_box, crop_box))
        # vecrot 的 predict() 期望**列表**（它内部再 torch.stack(scale_codes[s:s+batch])），
        # 传单个 stack 好的张量会在切片后变成 1 维，stack 报 "must be tuple of Tensors"。
        return codes

    def close(self) -> None:
        self._vec = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:                          # noqa: BLE001
            pass


def batch_limit(n: int) -> int:
    return max(int(n), 1)


class _OOM(RuntimeError):
    """内部信号：触发减批重试。"""


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------ 工厂


def build_vectorizer(profile: str, options: dict | None = None) -> VectorizerBackend:
    options = options or {}
    profile = (profile or "").strip()
    if profile in ("vecrot", "scalecond_verified", "vectorllm"):
        return VecrotBackend(options)
    if profile in ("null", "none"):
        return NullVectorizer()
    raise VectorizeNotAvailable("VECTOR_PROFILE_UNKNOWN",
                               f"未知 vector_profile：{profile!r}（可选 vecrot / null）")


class NullVectorizer:
    """返回确定性矩形（围绕目标框外扩 40%），仅用于无 GPU 时验证坐标链与导出。"""

    name = "null"

    def prepare(self) -> dict:
        return {"backend": "null",
                "note": "polygons are synthetic rectangles; NOT model output"}

    def predict(self, tasks: Sequence[dict]) -> list[PolygonResult]:
        out = []
        for t in tasks:
            meta = t.get("crop_meta") or {}
            box = meta.get("obj_box_in_crop")
            if box is None:
                out.append(PolygonResult("skipped", reason="no_obj_box"))
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = (x2 - x1) * 0.7, (y2 - y1) * 0.7
            poly = [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
                    [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]
            out.append(PolygonResult("ok", polygon_crop_px=poly, extra={"synthetic": True}))
        return out

    def close(self) -> None:
        pass
