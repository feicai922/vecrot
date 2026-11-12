"""作业编排主流程（开发文档 §8）。

阶段：
    VALIDATING → PREPARING → DETECTING → DEDUPLICATING → VECTORIZING → EXPORTING
    → SUCCEEDED / PARTIAL / CANCELLED / FAILED

外部进程边界：
    * 检测与轮廓各自独立进程、各自解释器（manifest 决定），绝不 import 到协调器进程。
    * 协调器自身只需要 numpy + rasterio + shapely + pyproj + geopandas/pyogrio。
    * 单卡顺序：检测进程退出后再起轮廓进程。
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
import traceback
from dataclasses import asdict

import numpy as np

from ..contracts.events import EventEmitter, JsonlStreamDecoder, worker_log
from ..contracts.jobspec import JobSpec, JobSpecError, parse_job_spec
from ..contracts.version import EXIT_CODES, JobState, STAGE_STATE, check_transition
from ..geo import boxes as bgeo
from ..geo import crop as cgeo
from ..geo import exporter
from ..geo import quality
from ..geo import raster as rgeo
from ..geo import transform as tgeo
from .ledger import open_ledger
from .procs import WorkerProcess, build_clean_env, python_for, write_request

CANCELLED_EXC = "CANCELLED"


class JobCancelled(RuntimeError):
    pass


class FatalJobError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class Coordinator:
    def __init__(self, spec: JobSpec, manifest: dict, work_root: str,
                 emitter: EventEmitter | None = None):
        self.spec = spec
        self.manifest = manifest
        self.job_id = spec.job_id
        self.work_dir = os.path.join(work_root, "jobs", self.job_id)
        self.emitter = emitter or EventEmitter(spec.job_id)
        self.ledger = open_ledger(work_root)
        self._cancel_flag = threading_flag()
        self._children: list[WorkerProcess] = []
        self._seen_seqs: dict[str, set] = {}
        #: 检测 worker 上报的场景分类统计（无 scene_head 时为空）
        self._scene_stats: dict = {}
        self.started = time.time()
        self.report: dict = {}
        self.stage_times: dict = {}

    # ---------------------------------------------------------- 控制

    def request_cancel(self) -> None:
        """异步请求取消：设置标志，worker 在批次边界退出。"""
        self._cancel_flag.set()

    def _check_cancel(self) -> None:
        if self._cancel_flag.is_set():
            raise JobCancelled(CANCELLED_EXC)

    def _set_state(self, state: JobState, stage: str | None = None, **kw) -> None:
        cur = self.ledger.get_job(self.job_id)
        prev = JobState(cur["state"]) if cur else JobState.CREATED
        if prev is not state:
            check_transition(prev, state)
        self.ledger.set_state(self.job_id, state, stage=stage, **kw)

    # ---------------------------------------------------------- 入口

    def run(self) -> int:
        """执行作业，返回协调器退出码。"""
        try:
            self.ledger.create_job(self.job_id, self.spec.to_manifest())
            self._set_state(JobState.VALIDATING, "VALIDATING")
            self._stage_start("VALIDATING")
            ctx = self._validate()
            self._stage_end("VALIDATING")

            self._set_state(JobState.PREPARING, "PREPARING")
            self._stage_start("PREPARING")
            self._prepare(ctx)
            self._stage_end("PREPARING")

            self._set_state(JobState.DETECTING, "DETECTING")
            self._stage_start("DETECTING")
            self._detect(ctx)
            self._stage_end("DETECTING")

            self._set_state(JobState.DEDUPLICATING, "DEDUPLICATING")
            self._stage_start("DEDUPLICATING")
            dets = self._dedup(ctx)
            self._stage_end("DEDUPLICATING")

            self._set_state(JobState.VECTORIZING, "VECTORIZING")
            self._stage_start("VECTORIZING")
            self._vectorize(ctx, dets)
            self._stage_end("VECTORIZING")

            self._set_state(JobState.EXPORTING, "EXPORTING")
            self._stage_start("EXPORTING")
            counts = self._export(ctx)
            self._stage_end("EXPORTING")

            self._finish_success(ctx, dets, counts)
            return EXIT_CODES["ok"]

        except JobCancelled:
            self._handle_cancel()
            return EXIT_CODES["cancelled"]
        except FatalJobError as exc:
            self._handle_failure(exc.code, exc.message)
            return EXIT_CODES["failed"]
        except Exception as exc:                   # noqa: BLE001
            worker_log(traceback.format_exc())
            self._handle_failure("INTERNAL_ERROR", f"{type(exc).__name__}: {exc}")
            return EXIT_CODES["failed"]
        finally:
            self._reap_children()
            self.ledger.close()

    # ---------------------------------------------------------- 阶段实现

    def _stage_start(self, name: str) -> None:
        self.ledger.stage_start(self.job_id, name)
        self.stage_times[name] = time.time()
        self.emitter.emit("stage_started", stage=name, message=f"{name} started")

    def _stage_end(self, name: str, metrics: dict | None = None) -> None:
        elapsed = time.time() - self.stage_times.get(name, time.time())
        merged = dict(metrics or {})
        merged.setdefault("elapsed_s", round(elapsed, 3))
        self.ledger.stage_finish(self.job_id, name, merged)
        self.emitter.emit("stage_finished", stage=name,
                          message=f"{name} finished in {elapsed:.1f}s", extra={"metrics": merged})

    # ---- VALIDATING
    def _validate(self) -> dict:
        self._check_cancel()
        p = self.spec
        try:
            info = rgeo.inspect_raster(p.raster_path, p.bands_rgb)
        except rgeo.RasterError as exc:
            raise FatalJobError(exc.code, exc.message) from exc
        for w in info.warnings:
            self.emitter.emit("warning", stage="VALIDATING", message=w)

        rgeo.check_writable(p.output_dir)
        rgeo.check_disk_space(p.output_dir, need_bytes=200 * 1024 * 1024)

        # AOI → 影像 CRS
        src_crs = info.crs
        aoi_crs = p.aoi_crs or src_crs
        aoi_geo = p.aoi_geojson
        if aoi_crs != src_crs:
            try:
                aoi_geo = tgeo.aoi_transform_densified(aoi_geo, aoi_crs, src_crs)
            except Exception as exc:                # noqa: BLE001
                raise FatalJobError("AOI_CRS_TRANSFORM_FAILED",
                                    f"AOI 从 {aoi_crs} 转到 {src_crs} 失败：{exc}") from exc

        aoi_geom = quality.aoi_geometry(aoi_geo)
        if not aoi_geom.is_valid:
            aoi_geom = aoi_geom.buffer(0)
        if aoi_geom.is_empty:
            raise FatalJobError("AOI_EMPTY", "AOI 在影像 CRS 下为空几何")

        win = tgeo.geo_to_pixel_window(aoi_geo, info.transform, (info.height, info.width))
        if win is None:
            raise FatalJobError("AOI_OUTSIDE_RASTER",
                                "AOI 与影像有效范围不相交；请确认坐标系与框选范围")

        out_crs = src_crs if p.output.crs_policy == "source" else p.output.crs_policy
        if out_crs != src_crs:
            from pyproj import CRS

            try:
                CRS.from_user_input(out_crs)
            except Exception as exc:                # noqa: BLE001
                raise FatalJobError("OUTPUT_CRS_INVALID", f"输出 CRS 无法解析：{out_crs}（{exc}）") from exc

        self.report.update({
            "schema_version": p.schema_version,
            "job_id": self.job_id,
            "started_at_utc": exporter.utc_now_iso(),
            "input": {
                "raster_path": info.path,
                "width": info.width, "height": info.height,
                "count": info.count, "dtypes": list(info.dtypes),
                "crs": src_crs,
                "transform": list(info.transform),
                "bands_rgb": list(p.bands_rgb),
                "nodata": info.nodata,
                "rotated_affine": info.is_rotated,
            },
            "aoi": {
                "crs_input": aoi_crs,
                "pixel_window": list(win.as_tuple()),
                "policy": p.output.aoi_policy,
            },
            "output_crs": out_crs,
            "pixel_area_m2": self._pixel_area_m2(info, out_crs),
        })
        self.emitter.emit("progress", stage="VALIDATING", done=1, total=1,
                          message=f"AOI → 像素窗口 {win.as_tuple()}，源 CRS {src_crs}",
                          extra={"raster": {"width": info.width, "height": info.height},
                                 "window": list(win.as_tuple())})
        return {"info": info, "aoi_geo": aoi_geo, "aoi_geom": aoi_geom,
                "window": win, "src_crs": src_crs, "out_crs": out_crs}

    @staticmethod
    def _pixel_area_m2(info, out_crs: str) -> float | None:
        """源像素 → m² 的乘数；不可靠时返回 None（§9.2 允许 area_m2 为空）。"""
        from pyproj import CRS

        try:
            crs = CRS.from_user_input(out_crs)
        except Exception:                           # noqa: BLE001
            return None
        if not crs.is_projected:
            return None
        gt = info.transform
        if abs(gt[2]) > 1e-12 or abs(gt[4]) > 1e-12:
            return None                             # 旋转仿射不适用简单相乘
        unit = crs.axis_info[0].unit_conversion_factor if crs.axis_info else 1.0
        return abs(gt[1] * gt[5]) * float(unit) ** 2

    # ---- PREPARING
    def _prepare(self, ctx: dict) -> None:
        """铺窗口 + 固定拉伸参数 + 写源影像指纹。"""
        self._check_cancel()
        info: rgeo.RasterInfo = ctx["info"]
        p = self.spec
        win = ctx["window"]
        windows = tgeo.tile_windows(win, p.inference.tile_size, p.inference.tile_overlap,
                                    raster_w=info.width, raster_h=info.height)

        import rasterio

        with rasterio.open(info.path) as src:
            is_8bit = all(src.dtypes[b - 1] in ("uint8",) for b in p.bands_rgb) and \
                all(str(src.dtypes[b - 1]) == "uint8" for b in p.bands_rgb)
            stretch = rgeo.compute_stretch(src, win, p.bands_rgb)

        # 窗口记录：补齐与边界都在这一步定死，后续不再从文件名推断
        records = []
        for i, w in enumerate(windows):
            pad_w = max(0, p.inference.tile_size - w.width)
            pad_h = max(0, p.inference.tile_size - w.height)
            records.append({
                "window_id": i,
                "xoff": w.xoff, "yoff": w.yoff,
                "width": w.width, "height": w.height,
                "pad": {"left": 0, "top": 0, "right": pad_w, "bottom": pad_h},
            })
        self.ledger.add_windows(self.job_id, records)

        import hashlib

        stat = os.stat(info.path)
        fp = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "crs": info.crs,
            "transform": list(info.transform),
            "width": info.width, "height": info.height,
        }
        self.ledger.set_checkpoint(self.job_id, "raster_fingerprint", fp)
        self.ledger.set_checkpoint(self.job_id, "stretch", stretch.to_dict())
        self.ledger.set_checkpoint(self.job_id, "is_8bit", bool(is_8bit))
        self.ledger.set_checkpoint(self.job_id, "params", {
            "tile_size": p.inference.tile_size,
            "tile_overlap": p.inference.tile_overlap,
            "crop_expand": p.inference.crop_expand,
            "score_threshold": p.inference.score_threshold,
            "dedup_iou": p.inference.det_dedup_iou,
        })

        ctx["is_8bit"] = is_8bit
        ctx["stretch"] = stretch
        ctx["window_records"] = records
        self.report["windows"] = {
            "count": len(records),
            "tile_size": p.inference.tile_size,
            "tile_overlap": p.inference.tile_overlap,
            "stretch": stretch.to_dict(),
        }
        self.emitter.emit("progress", stage="PREPARING", done=len(records), total=len(records),
                          message=f"{len(records)} 个检测窗口，{info.width}×{info.height} 像素")

    # ---- DETECTING
    def _detect(self, ctx: dict) -> None:
        self._check_cancel()
        import rasterio

        info: rgeo.RasterInfo = ctx["info"]
        p = self.spec
        det_cfg = self._detector_options()
        opts = det_cfg["options"]
        staging = os.path.join(self.work_dir, "windows")
        os.makedirs(staging, exist_ok=True)

        # 把每个窗口的 RGB 落成 .npy，检测进程不装 rasterio（§4 环境隔离）
        window_inputs = []
        with rasterio.open(info.path) as src:
            for rec in ctx["window_records"]:
                self._check_cancel()
                w = tgeo.PixelWindow(rec["xoff"], rec["yoff"], rec["width"], rec["height"])
                rgb, _valid = rgeo.read_rgb_window(src, w, p.bands_rgb, ctx["stretch"],
                                                   ctx["is_8bit"])
                pad = rec["pad"]
                if pad["right"] or pad["bottom"]:
                    rgb = np.pad(rgb, ((0, pad["bottom"]), (0, pad["right"]), (0, 0)),
                                 mode="constant", constant_values=0)
                path = os.path.join(staging, f"w{rec['window_id']:06d}.npy")
                np.save(path, rgb)
                window_inputs.append({
                    "window_id": rec["window_id"], "image": path,
                    "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                    "xoff": rec["xoff"], "yoff": rec["yoff"],
                })

        req = {
            "schema_version": self.spec.schema_version,
            "job_id": self.job_id,
            "profile": det_cfg["profile"],
            "options": det_cfg["options"],
            "windows": window_inputs,
        }
        out_path = os.path.join(self.work_dir, "detections.raw.jsonl")
        self._run_worker("detector", req, out_path, stage="DETECTING")

        # 解析结果 → Detection（局部坐标 → 源像素）
        counts = {"candidates": 0, "windows_ok": 0, "windows_error": 0}
        candidates: list[bgeo.Detection] = []
        # 注意：report["detector"] 已由 _run_worker 从 worker 的 ready 事件填入，
        # 这里不能再覆盖成空字典（否则权重加载报告丢失）
        decoder = JsonlStreamDecoder()
        by_id = {w["window_id"]: w for w in window_inputs}
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    counts["windows_error"] += 1
                    continue
                wid = int(rec["window_id"])
                if rec.get("status") != "ok":
                    counts["windows_error"] += 1
                    self.ledger.set_window_state(self.job_id, wid, "error",
                                                 error=str(rec.get("error", ""))[:500])
                    self.emitter.emit("warning", stage="DETECTING",
                                      message=f"窗口 {wid} 检测失败：{rec.get('error')}")
                    continue
                win = by_id[wid]
                dets = rec.get("detections") or []
                counts["windows_ok"] += 1
                counts["candidates"] += len(dets)
                self.ledger.set_window_state(self.job_id, wid, "ok", n_candidates=len(dets))
                for d in dets:
                    r = d.get("rbox_xywha_rad")
                    if not r or len(r) != 5:
                        continue
                    score = float(d.get("score", 0.0))
                    if score < p.inference.score_threshold:
                        continue
                    # 窗口局部（含 padding）→ 源像素：先减 pad，再加窗口偏移
                    pad_left = 0
                    pad_top = 0
                    cx = float(r[0]) - pad_left + win["xoff"]
                    cy = float(r[1]) - pad_top + win["yoff"]
                    edge = self._is_edge_truncated(cx, cy, float(r[2]), float(r[3]),
                                                   float(r[4]), ctx["window"], info)
                    candidates.append(bgeo.Detection(
                        window_id=wid, score=score,
                        rbox=(cx, cy, float(r[2]), float(r[3]), float(r[4])),
                        edge_truncated=edge))

        ctx["candidates"] = candidates
        self.report["detection"] = counts
        if self._scene_stats:
            self.report["scene_classification"] = self._scene_stats
        self.ledger.set_checkpoint(self.job_id, "candidate_count", counts["candidates"])
        self.emitter.emit("progress", stage="DETECTING", done=counts["windows_ok"],
                          total=len(window_inputs),
                          message=f"{counts['windows_ok']} 窗口成功 / "
                                  f"{counts['windows_error']} 失败；候选框 {counts['candidates']}",
                          extra={"candidates": counts["candidates"]})
        if counts["windows_ok"] == 0 and window_inputs:
            raise FatalJobError("DETECTION_ALL_FAILED", "所有窗口检测失败，无法继续")

    @staticmethod
    def _is_edge_truncated(cx, cy, w, h, ang, aoi_win: tgeo.PixelWindow, info) -> bool:
        """框是否贴到源影像边界（贴边说明建筑可能被影像边界截断）。"""
        corners = bgeo.rbox_corners((cx, cy, w, h, ang))
        x1, y1 = corners[:, 0].min(), corners[:, 1].min()
        x2, y2 = corners[:, 0].max(), corners[:, 1].max()
        margin = 1.0
        if x1 <= margin and aoi_win.xoff <= 0:
            return True
        if y1 <= margin and aoi_win.yoff <= 0:
            return True
        if x2 >= info.width - margin and aoi_win.x1 >= info.width:
            return True
        if y2 >= info.height - margin and aoi_win.y1 >= info.height:
            return True
        return False

    # ---- DEDUPLICATING
    def _dedup(self, ctx: dict) -> list[dict]:
        self._check_cancel()
        candidates: list[bgeo.Detection] = ctx["candidates"]
        result = bgeo.dedup_detections(
            candidates, iou_threshold=self.spec.inference.det_dedup_iou,
            prefer_edge_complete=True)
        dets = bgeo.assign_det_ids(result.kept)

        # 记录抑制关系（可追溯，§7.2）：kept_idx 直接指向保留列表下标
        kept_ids = [d["det_id"] for d in dets]
        sup_rows = []
        for cand_idx, (det, kept_idx) in enumerate(result.suppressed):
            sup_rows.append((f"s{cand_idx:06d}", kept_ids[kept_idx],
                             bgeo.rbox_iou(det.rbox, result.kept[kept_idx].rbox)))
        self.ledger.add_detections(self.job_id, dets)
        self.ledger.add_suppressions(self.job_id, sup_rows)
        self.report["dedup"] = {
            "candidates_in": len(candidates),
            "kept": len(dets),
            "suppressed": len(result.suppressed),
            "iou_threshold": self.spec.inference.det_dedup_iou,
        }
        self.emitter.emit("progress", stage="DEDUPLICATING", done=len(dets),
                          total=len(candidates),
                          message=f"候选 {len(candidates)} → 保留 {len(dets)}",
                          extra={"suppressed": len(result.suppressed)})
        return dets

    # ---- VECTORIZING
    def _vectorize(self, ctx: dict, dets: list[dict]) -> None:
        self._check_cancel()
        p = self.spec
        info: rgeo.RasterInfo = ctx["info"]
        if not dets:
            self.report["vectorization"] = {"requested": 0, "ok": 0, "abstained": 0,
                                            "empty": 0, "error": 0, "skipped": 0}
            ctx["polygons"] = {}
            self.emitter.emit("progress", stage="VECTORIZING", done=0, total=0,
                              message="无检测框，跳过轮廓生成")
            return

        crops_dir = os.path.join(self.work_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)
        tasks = []
        import rasterio

        with rasterio.open(info.path) as src:
            for d in dets:
                self._check_cancel()
                rbox = d["rbox"]
                tf = cgeo.plan_crop(rbox, p.inference.crop_expand)
                if tf is None:
                    self.ledger.set_det_status(self.job_id, d["det_id"], "skipped",
                                               reasons="crop_plan_rejected")
                    continue
                half_w, half_h = tf.crop_w / 2, tf.crop_h / 2
                # 读取包含整个裁剪框的源图局部区域（不截断内容，边缘用 0 补齐）
                x0 = int(math.floor(tf.cx - half_w))
                y0 = int(math.floor(tf.cy - half_h))
                x1 = int(math.ceil(tf.cx + half_w))
                y1 = int(math.ceil(tf.cy + half_h))
                rx0, ry0 = max(x0, 0), max(y0, 0)
                rx1, ry1 = min(x1, info.width), min(y1, info.height)
                if rx1 <= rx0 or ry1 <= ry0:
                    self.ledger.set_det_status(self.job_id, d["det_id"], "skipped",
                                               reasons="outside_raster")
                    continue
                w = tgeo.PixelWindow(rx0, ry0, rx1 - rx0, ry1 - ry0)
                rgb, _valid = rgeo.read_rgb_window(src, w, p.bands_rgb, ctx["stretch"],
                                                   ctx["is_8bit"])
                # 补到完整裁剪区域大小，并记录偏移
                left_pad, top_pad = rx0 - x0, ry0 - y0
                right_pad, bottom_pad = x1 - rx1, y1 - ry1
                if left_pad or top_pad or right_pad or bottom_pad:
                    rgb = np.pad(rgb, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
                                 mode="constant", constant_values=0)
                local_tf = cgeo.offset_transform(tf, x0, y0)
                cropped = cgeo.crop_rotated(rgb, local_tf)
                crop_path = os.path.join(crops_dir, f"{d['det_id']}.npy")
                np.save(crop_path, cropped)
                tasks.append({
                    "det_id": d["det_id"],
                    "crop_image": crop_path,
                    "crop_meta": {
                        "obj_box_in_crop": list(tf.obj_box_in_crop()),
                        "crop_box_in_crop": list(tf.crop_box_in_crop()),
                        "transform": tf.to_dict(),
                    },
                })
                ctx.setdefault("transforms", {})[d["det_id"]] = tf

        if not tasks:
            self.report["vectorization"] = {"requested": 0, "ok": 0, "abstained": 0,
                                            "empty": 0, "error": 0, "skipped": len(dets)}
            ctx["polygons"] = {}
            return

        vec_cfg = self._vector_options()
        req = {
            "schema_version": self.spec.schema_version,
            "job_id": self.job_id,
            "profile": vec_cfg["profile"],
            "options": vec_cfg["options"],
            "tasks": tasks,
        }
        out_path = os.path.join(self.work_dir, "polygons.raw.jsonl")
        self._run_worker("vector", req, out_path, stage="VECTORIZING")

        counts = {"requested": len(tasks), "ok": 0, "abstained": 0, "empty": 0,
                  "error": 0, "skipped": len(dets) - len(tasks)}
        polys: dict[str, dict] = {}
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                det_id = rec.get("det_id")
                status = rec.get("status", "error")
                if status not in counts:
                    status = "error"
                counts[status] += 1
                self.ledger.set_det_status(self.job_id, det_id, status,
                                           reasons=str(rec.get("reason") or "")[:300],
                                           elapsed_s=rec.get("elapsed_s"))
                if status == "ok" and rec.get("polygon_crop_px"):
                    polys[det_id] = rec
        self.report["vectorization"] = counts
        ctx["polygons"] = polys
        self.emitter.emit("progress", stage="VECTORIZING", done=counts["ok"] + counts["error"],
                          total=len(tasks),
                          message=f"轮廓成功 {counts['ok']} / 弃权 {counts['abstained']} "
                                  f"/ 空 {counts['empty']} / 失败 {counts['error']}",
                          extra={"counts": counts})

    # ---- EXPORTING
    def _export(self, ctx: dict) -> dict:
        self._check_cancel()
        info: rgeo.RasterInfo = ctx["info"]
        p = self.spec
        src_crs = ctx["src_crs"]
        out_crs = ctx["out_crs"]
        gt = info.transform
        polygons: dict = ctx.get("polygons") or {}
        transforms: dict = ctx.get("transforms") or {}
        # AOI 必须先转成**源像素**几何，才能与模型输出（源像素坐标）求交；
        # 直接拿地理坐标 AOI 与像素几何相交会永远为空，从而静默丢光所有建筑。
        aoi_geom_geo = ctx["aoi_geom"]
        aoi_geom_px = _geo_geometry_to_pixels(aoi_geom_geo, gt)

        dets = self.ledger.detections(self.job_id)
        box_rows = []
        building_rows = []
        # 被 AOI 策略排除的检测（框与轮廓都不写出，保持两图层一致）
        dropped_outside_aoi = 0
        polygons_in_aoi = 0
        for d in dets:
            det_id = d["det_id"]
            rbox = d["rbox"]
            corners_src = bgeo.rbox_corners(rbox).tolist()
            status = d["vector_status"]
            poly_rec = polygons.get(det_id)
            valid_poly = None
            reasons: list[str] = []
            geom_status = ""
            clipped = False

            # 检测框与轮廓必须走**同一套** AOI 判定。
            # 之前只过滤轮廓、框全量输出，导致 boxes 比 buildings 多出一批
            # AOI 外的检测（实测 143 框 / 56 轮廓），看起来像"轮廓没显示"。
            # 检测窗口按 tile_size 铺，必然覆盖到 AOI 之外，这些结果不属于本次作业。
            if p.output.aoi_policy == "intersect_keep_complete":
                from shapely.geometry import Polygon as _Poly
                box_poly = _Poly([(float(x), float(y)) for x, y in corners_src])
                if not box_poly.intersects(aoi_geom_px):
                    dropped_outside_aoi += 1
                    self.ledger.set_det_status(self.job_id, det_id, "outside_aoi",
                                               reasons="box_outside_aoi",
                                               elapsed_s=d.get("elapsed_s"))
                    continue

            if poly_rec is not None:
                tf = transforms.get(det_id)
                if tf is None:
                    geom_status = quality.STATUS_INVALID
                    reasons = ["missing_crop_transform"]
                else:
                    crop_pts = poly_rec["polygon_crop_px"]
                    src_pts = tf.crop_to_source(crop_pts)
                    chk = quality.check_polygon(src_pts)
                    geom_status = chk.status
                    reasons = list(chk.reasons)
                    if chk.geometry is not None:
                        geom = chk.geometry
                        if p.output.aoi_policy == "intersect_keep_complete":
                            if not geom.intersects(aoi_geom_px):
                                reasons.append("outside_aoi")
                                geom = None
                        if geom is not None and p.output.aoi_policy == "clip_strict":
                            clipped_geom = quality.clip_polygon_to_aoi(geom, aoi_geom_px)
                            if clipped_geom.is_empty:
                                reasons.append("clipped_empty")
                                geom = None
                            else:
                                clipped = True
                                geom = clipped_geom
                        valid_poly = geom

            # 坐标换算：源像素（A 约定）→ 可选输出 CRS 的地理坐标
            if valid_poly is not None:
                from shapely.ops import transform as shp_transform

                Xg, Yg, Zg = _make_px_to_geo(gt, src_crs, out_crs)
                geom_out = shp_transform(
                    lambda x, y, z=None: (Xg(x, y, z), Yg(x, y, z)), valid_poly)
                if geom_out.is_empty:
                    geom_out = None
                    reasons.append("empty_after_crs")
            else:
                geom_out = None

            # 框：始终输出（轮廓失败也保留，§9.3）
            box_corners = cgeo.source_px_to_geo(corners_src, gt)
            if out_crs != src_crs:
                from pyproj import Transformer

                tr = _cached_transformer(src_crs, out_crs)
                xs, ys = tr.transform([c[0] for c in box_corners],
                                      [c[1] for c in box_corners])
                box_corners = [[float(a), float(b)] for a, b in zip(xs, ys)]

            box_rows.append({
                "det_id": det_id, "det_score": d["det_score"],
                "class_name": "building", "box_type": "obb",
                "angle_rad": rbox[4], "vector_status": status,
                "source_window": d["source_window"],
                "edge_truncated": d["edge_truncated"], "clipped": clipped,
                "corners": box_corners,
            })

            if geom_out is not None:
                polygons_in_aoi += 1
                # 面积用**像素面积 × 每像素面积**，不用输出 CRS 里的几何面积：
                # 输出 CRS 可能是经纬度，取面积会得到平方度。
                area_px = float(valid_poly.area)
                building_rows.append({
                    "det_id": det_id, "det_score": d["det_score"],
                    "geom_status": geom_status or quality.STATUS_OK,
                    "clipped": clipped, "edge_truncated": d["edge_truncated"],
                    "area_px": area_px,
                    "geom": geom_out, "reasons": reasons,
                })

        # 计数守恒校验（§9.4）
        status_counts = self.ledger.det_status_counts(self.job_id)
        total_kept = len(dets)
        accounted = sum(status_counts.values())
        if total_kept != accounted:
            self.report["count_warning"] = {
                "kept": total_kept, "with_status": accounted, "detail": status_counts}

        self.report.update({
            "aoi_filter": {
                "policy": p.output.aoi_policy,
                "dropped_outside_aoi": dropped_outside_aoi,
            },
            "counts": {
                "windows": self.report.get("windows", {}).get("count", 0),
                "candidates": self.report.get("dedup", {}).get("candidates_in", 0),
                # kept_boxes / polygons_ok 都按「落在 AOI 内」口径统计，
                # 与 buildings_written 可直接对照
                "kept_boxes": total_kept - dropped_outside_aoi,
                "polygons_ok": polygons_in_aoi,
                "buildings_written": len(building_rows),
                "status": status_counts,
            },
            "completed_at_utc": exporter.utc_now_iso(),
        })
        self.report["elapsed_s"] = round(time.time() - self.started, 3)
        self.report["stages"] = self._stage_metrics()
        # 记录**实际生效**的 profile（job.json 未指定时由 deployment manifest 决定），
        # 否则 report 里会是 null，事后无法追溯这次到底跑了哪个后端
        det_effective = (self.report.get("detector") or {}).get("backend")             or self._detector_options()["profile"]
        vec_effective = (self.report.get("vectorizer") or {}).get("backend")             or self._vector_options()["profile"]
        self.report["manifest"] = {
            "detector_profile": det_effective,
            "vector_profile": vec_effective,
            "requested_detector_profile": p.inference.detector_profile,
            "requested_vector_profile": p.inference.vector_profile,
            "detector": self.report.get("detector"),
            "vectorizer": self.report.get("vectorizer"),
        }

        try:
            result = exporter.export_outputs(
                job_id=self.job_id,
                detections=box_rows,
                buildings=building_rows,
                crs=out_crs,
                output_dir=p.output_dir,
                src_crs=src_crs,
                aoi_policy=p.output.aoi_policy,
                report=self.report,
            )
        except exporter.ExportError as exc:
            raise FatalJobError(exc.code, exc.message) from exc

        self.report["outputs"] = {
            "boxes": result.boxes_path,
            "buildings": result.buildings_path,
            "manifest": result.manifest_path,
            "report": result.report_path,
        }
        # report.json 已由 exporter 写过一次；补上最终计数后再写一遍
        with open(result.report_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, ensure_ascii=False, indent=2)
        self.emitter.emit("progress", stage="EXPORTING", done=2, total=2,
                          message=f"写出 {result.counts['boxes']} 框 / "
                                  f"{result.counts['buildings']} 轮廓")
        return result.counts

    # ---------------------------------------------------------- 收尾

    def _finish_success(self, ctx: dict, dets: list[dict], counts: dict) -> None:
        status_counts = self.ledger.det_status_counts(self.job_id)
        n_kept = len(dets)
        n_ok = len(ctx.get("polygons") or {})
        partial = False
        if n_kept and n_ok < n_kept:
            partial = True
        win_counts = self.ledger.window_counts(self.job_id)
        if win_counts["error"]:
            partial = True

        state = JobState.PARTIAL if partial else JobState.SUCCEEDED
        exit_code = EXIT_CODES["partial"] if partial else EXIT_CODES["ok"]
        self._set_state(state, exit_code=exit_code,
                        error_code="PARTIAL_COMPLETION" if partial else None)
        self.emitter.emit("completed", stage="EXPORTING", done=n_ok, total=n_kept,
                          message=("部分成功：框已全部保留，"
                                   f"{n_kept - n_ok} 个轮廓未生成" if partial else "作业完成"),
                          extra={
                              "state": state.value,
                              "exit_code": exit_code,
                              "outputs": self.report.get("outputs"),
                              "counts": self.report.get("counts"),
                              "status": status_counts,
                          })

    def _handle_cancel(self) -> None:
        try:
            self._set_state(JobState.CANCELLING)
            self._set_state(JobState.CANCELLED, exit_code=EXIT_CODES["cancelled"],
                            error_code="CANCELLED", error_msg="用户取消")
        except ValueError:
            self.ledger.set_state(self.job_id, JobState.CANCELLED,
                                  exit_code=EXIT_CODES["cancelled"],
                                  error_code="CANCELLED", error_msg="用户取消")
        self.emitter.emit("cancelled", stage=None, message="作业已取消；已完成部分保留在 work/ 下",
                          extra={"exit_code": EXIT_CODES["cancelled"]})

    def _handle_failure(self, code: str, message: str) -> None:
        try:
            cur = self.ledger.get_job(self.job_id)
            if cur and JobState(cur["state"]) is not JobState.FAILED:
                self._set_state(JobState.FAILED, error_code=code, error_msg=message,
                                exit_code=EXIT_CODES["failed"])
        except Exception:                          # noqa: BLE001
            self.ledger.set_state(self.job_id, JobState.FAILED, error_code=code,
                                  error_msg=message, exit_code=EXIT_CODES["failed"])
        self.emitter.emit("failed", stage=None, message=f"{code}: {message}",
                          code=code, extra={"exit_code": EXIT_CODES["failed"]})

    def _reap_children(self) -> None:
        for child in self._children:
            if child.poll() is None:
                child.kill()
        self._children.clear()

    def _stage_metrics(self) -> dict:
        out = {}
        for name in ("VALIDATING", "PREPARING", "DETECTING", "DEDUPLICATING",
                     "VECTORIZING", "EXPORTING"):
            out[name] = self.ledger.stage_metrics(self.job_id, name)
        return out

    # ---------------------------------------------------------- worker 调度

    def _env_role_for(self, role: str, request: dict) -> str:
        """某些后端需要不同的解释器。

        RT-DETR 是 HF Transformers 模型，需要 transformers + RTDetr API；
        而 mmdet/mmrotate 系需要 mmcv-full，两者不能装在同一环境
        （本机实测：detector 环境没有 transformers，vector 环境有但没 mmcv）。
        因此按 worker 的后端 profile 覆盖环境。
        """
        if role == "detector":
            profile = (request.get("profile") or "").strip()
            if profile in ("rtdetr_hf",):
                override = (self.manifest.get("environments") or {}).get(
                    "detector_hf") or {}
                if override.get("python"):
                    return "detector_hf"
            elif profile in ("strip_s_gzda", "strip_rcnn", "mmrotate_obb",
                             "mmdet_hbb", "jsonl", "null", ""):
                pass        # 用默认 detector 环境（mmcv 系 / 无需模型）
        return role

    def _run_worker(self, role: str, request: dict, out_path: str, stage: str) -> None:
        """按 manifest 启动外部 worker，转发它的事件，校验退出码。"""
        # 两个概念要分开：
        #   role       = worker 类型（detector / vector），决定跑哪个模块
        #   env_role   = 用哪套解释器（可能因后端不同，如 detector_hf）
        # 之前把 role 直接改成 detector_hf，随后用它索引模块字典 → KeyError。
        module_role = role
        env_role = self._env_role_for(role, request)
        try:
            python_exe = python_for(self.manifest, env_role)
        except RuntimeError as exc:
            # 部署配置问题给专用错误码，而不是笼统的 INTERNAL_ERROR。
            # 附上恢复路径：这条错误最常出现在「环境被清理 / 换机器」之后。
            hint = (
                "恢复方法：按 docs/安装教程-自建环境.md 建好 "
                + env_role
                + " 环境，再把 python.exe 绝对路径填进 "
                + "deployment/manifest.json 的 environments."
                + env_role + ".python"
            )
            raise FatalJobError(
                "WORKER_ENV_MISSING",
                str(exc) + chr(10) + hint) from exc
        env_cfg = (self.manifest.get("environments") or {}).get(env_role) or {}
        env = build_clean_env(
            python_exe,
            gis_env_dir=env_cfg.get("gis_env_dir"),
            extra_path=env_cfg.get("extra_path") or [],
            extra_env=env_cfg.get("env") or {},
        )
        module = {"detector": "backend.workers.detector_worker",
                  "vector": "backend.workers.vector_worker"}[module_role]
        req_path = write_request(request, self.work_dir, f"req_{module_role}")
        argv = [python_exe, "-u", "-m", module, "--request", req_path, "--out", out_path]

        repo_root = self.manifest.get("_repo_root") or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env["PYTHONPATH"] = repo_root

        decoder = JsonlStreamDecoder()
        backend_info: dict = {}

        def on_stdout(data: bytes) -> None:
            for ev in decoder.feed(data):
                self._relay_event(ev, role)
                # worker 把后端加载信息放在 ready 事件的 extra.backend 里
                payload = (ev.get("extra") or {}).get("backend") or ev.get("backend")
                if payload:
                    backend_info.update(payload)

        proc = WorkerProcess(argv, env, cwd=repo_root, on_stdout=on_stdout)
        self._children.append(proc)
        self.ledger.set_checkpoint(self.job_id, f"worker_{role}_pid", proc.pid)
        worker_log(f"{module_role} worker started pid={proc.pid} "
                   f"env={env_role} python={python_exe}")

        deadline = None
        while True:
            code = proc.poll()
            if code is not None:
                break
            if self._cancel_flag.is_set():
                worker_log(f"cancel requested; terminating {role} worker")
                proc.terminate_tree(grace=5.0)
                raise JobCancelled(CANCELLED_EXC)
            if deadline is not None and time.time() > deadline:
                proc.terminate_tree(grace=2.0)
                raise FatalJobError("WORKER_TIMEOUT", f"{role} worker 超时未退出")
            time.sleep(0.05)

        for ev in decoder.flush():
            self._relay_event(ev, role)
            payload = (ev.get("extra") or {}).get("backend") or ev.get("backend")
            if payload:
                backend_info.update(payload)

        stderr_text = proc.stderr_text()
        if stderr_text.strip():
            log_path = os.path.join(self.work_dir, f"{role}_worker.stderr.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n===== run at {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
                f.write(stderr_text)
            self.report.setdefault("logs", {})[f"{role}_stderr"] = log_path

        if code == 20:
            raise JobCancelled(CANCELLED_EXC)
        if code != 0:
            tail = "\n".join(stderr_text.strip().splitlines()[-8:])
            raise FatalJobError("WORKER_FAILED",
                                f"{role} worker 退出码 {code}。stderr 尾部：\n{tail}")
        if not os.path.exists(out_path):
            raise FatalJobError("WORKER_NO_OUTPUT", f"{role} worker 未生成结果文件 {out_path}")

        key = "detector" if role == "detector" else "vectorizer"
        if backend_info:
            self.report[key] = backend_info

    def _relay_event(self, ev: dict, role: str) -> None:
        """把 worker 事件转发给 UI，按 seq 去重 + 兼容性记录（§8.2）。"""
        sv = ev.get("schema_version")
        if sv != self.spec.schema_version:
            self.emitter.emit("warning", stage=ev.get("stage"),
                              message=f"{role} worker 使用 schema {sv}，与本端"
                                      f" {self.spec.schema_version} 不同",
                              extra={"compat": True})
        seen = self._seen_seqs.setdefault(role, set())
        seq = ev.get("seq")
        if seq is not None:
            if seq in seen:
                return
            seen.add(seq)
        event = ev.get("event")
        # 记录 worker 上报的场景分类统计（带 scene_head 的模型）
        extra = ev.get("extra") or {}
        if isinstance(extra, dict) and "scene" in extra:
            self._scene_stats = extra["scene"]
        if event in ("ready", "progress", "warning"):
            self.emitter.emit(event, stage=ev.get("stage") or None,
                              done=ev.get("done"), total=ev.get("total"),
                              message=str(ev.get("message") or "")[:500],
                              extra={"source": role})

    # ---------------------------------------------------------- 参数

    def _detector_options(self) -> dict:
        """检测配置。profile 解析顺序：job.json 显式指定 > manifest 默认。"""
        p = self.spec
        models = self.manifest.get("models") or {}
        det = (models.get("detector") or {})
        profile = p.inference.detector_profile or det.get("profile") or "strip_s_gzda"
        base = {
            "detections_jsonl": self._resolve(models.get("detections_jsonl")),
            "device": f"cuda:{p.inference.gpu_index}",
            "score_thr": det.get("score_thr", 0.05),
            "nms_iou": det.get("nms_iou", 0.3),
            "max_per_img": det.get("max_per_img", 2000),
        }
        if profile in ("strip_s_gzda", "strip_rcnn", "mmrotate_obb"):
            # job.json 里显式给的路径优先于 manifest（用户自选模型）
            base.update({
                "strip_rcnn_repo": self._resolve(det.get("repo")),
                "detector_weights": p.inference.detector_weights
                or self._resolve(det.get("weights")),
                "detector_config": p.inference.detector_config
                or self._resolve(det.get("config")),
                # 场景分类阈值：job.json > manifest > 0.5（只有带 scene_head 的模型生效）
                "scene_threshold": (
                    p.inference.scene_threshold
                    if p.inference.scene_threshold is not None
                    else det.get("scene_threshold", 0.5)),
                "derived_dir": os.path.join(self.work_dir, "derived"),
            })
        elif profile == "mmdet_hbb":
            base.update({
                "mmdet_repo": self._resolve(det.get("mmdet_repo")),
                "detector_weights": p.inference.detector_weights,
                "detector_config": p.inference.detector_config,
                "num_classes": p.inference.detector_num_classes,
                "derived_dir": os.path.join(self.work_dir, "derived"),
            })
        elif profile == "rtdetr_hf":
            # RT-DETR 是「目录即模型」：detector_weights 指向整个模型目录
            # （含 model.safetensors + config.json + preprocessor_config.json）。
            # 不需要 .py 配置与 num_classes（从 config.json 的 id2label 读）。
            base.update({
                "detector_weights": p.inference.detector_weights
                or self._resolve(det.get("rtdetr_dir")),
            })
        return {"profile": profile, "options": base}

    def _vector_options(self) -> dict:
        p = self.spec
        models = self.manifest.get("models") or {}
        vec = models.get("vectorizer") or {}
        profile = p.inference.vector_profile or vec.get("profile") or "vecrot"
        return {"profile": profile, "options": {
            "vecrot_dir": self._resolve(vec.get("vecrot_dir")),
            "vector_ckpt": p.inference.vector_ckpt or self._resolve(vec.get("ckpt")),
            "vector_base": p.inference.vector_base or self._resolve(vec.get("base")),
            "device": f"cuda:{p.inference.gpu_index}",
            "vector_batch": p.inference.vector_batch,
            "max_retries": p.inference.max_retries,
            "scale_cond": vec.get("scale_cond"),
        }}

    def _resolve(self, path: str | None) -> str | None:
        if not path:
            return None
        if os.path.isabs(path):
            return path
        root = self.manifest.get("_root") or ""
        return os.path.normpath(os.path.join(root, path))


# ------------------------------------------------------------ 工具


def threading_flag():
    """轻量取消标志（比 Event 更好用：可跨线程安全读写）。"""
    import threading

    return threading.Event()


_TRANSFORMERS: dict = {}


def _geo_geometry_to_pixels(geom_geo, gt):
    """地理几何（影像 CRS）→ 源像素索引几何（A 约定）。

    先做仿射逆变换（得到角点约定），再 -0.5 换到像素中心约定，与模型输出同坐标系。
    shapely 的 transform 逐点调用，因此这里必须同时支持标量与数组输入。
    """
    from shapely.ops import transform as shp_transform

    gt0, gt1, gt2, gt3, gt4, gt5 = [float(v) for v in gt]
    det = gt1 * gt5 - gt2 * gt4
    if abs(det) < 1e-18:
        raise FatalJobError("AFFINE_DEGENERATE", "仿射矩阵不可逆，无法把 AOI 转到像素坐标")

    def _to_px(X, Y, z=None):
        dx = np.asarray(X, dtype=np.float64) - gt0
        dy = np.asarray(Y, dtype=np.float64) - gt3
        px = (gt5 * dx - gt2 * dy) / det - 0.5
        py = (-gt4 * dx + gt1 * dy) / det - 0.5
        if np.ndim(px) == 0:
            px, py = float(px), float(py)
        return (px, py, z) if z is not None else (px, py)

    return shp_transform(_to_px, geom_geo)


def _make_px_to_geo(gt, src_crs: str, out_crs: str):
    """构造标量版「源像素（A 约定）→ 输出 CRS 地理坐标」函数。

    shapely 的 transform 逐点调用；这里只在标量层面做一次 +0.5，
    避免把数组/列表约定混进几何变换。
    """
    gt0, gt1, gt2, gt3, gt4, gt5 = [float(v) for v in gt]
    tr = _cached_transformer(src_crs, out_crs) if out_crs != src_crs else None

    def Xg(x, y, z=None):
        bx, by = float(x) + 0.5, float(y) + 0.5
        X = gt0 + bx * gt1 + by * gt2
        Y = gt3 + bx * gt4 + by * gt5
        if tr is None:
            return X
        return float(tr.transform(X, Y)[0])

    def Yg(x, y, z=None):
        bx, by = float(x) + 0.5, float(y) + 0.5
        X = gt0 + bx * gt1 + by * gt2
        Y = gt3 + bx * gt4 + by * gt5
        if tr is None:
            return Y
        return float(tr.transform(X, Y)[1])

    return Xg, Yg, None


def _cached_transformer(src: str, dst: str):
    from pyproj import Transformer

    key = (src, dst)
    tr = _TRANSFORMERS.get(key)
    if tr is None:
        tr = Transformer.from_crs(src, dst, always_xy=True)
        _TRANSFORMERS[key] = tr
    return tr
