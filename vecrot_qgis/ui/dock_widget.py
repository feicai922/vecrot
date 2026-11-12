"""面板逻辑：输入、AOI、运行、进度、结果（开发文档 §10）。

线程约束：QGIS 操作与图层加载只在主线程执行；外部进程只接触普通数据。
"""
from __future__ import annotations

import os

from qgis.core import QgsCoordinateReferenceSystem, QgsProject, QgsRasterLayer
from qgis.PyQt import uic
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import QDockWidget, QFileDialog, QMessageBox

from ..jobs.bridge import (EventBridge, coordinator_python, now_stamp, read_manifest,
                           repo_root_from)
from ..jobs.jobfile import JobSpecProblem, build_job_payload, write_job_file
from ..layers.result_loader import load_results, open_report
from ..maptools.aoi_tool import AoiRectangleTool, ensure_polygon_in_crs, geometry_area_m2

FORM_CLASS, _ = uic.loadUiType(os.path.join(os.path.dirname(__file__), "dock_widget_base.ui"))

STAGE_TEXT = {
    "VALIDATING": "校验输入",
    "PREPARING": "准备窗口",
    "DETECTING": "检测中",
    "DEDUPLICATING": "去重",
    "VECTORIZING": "生成轮廓",
    "EXPORTING": "写出成果",
}


class VecrotDockWidget(QDockWidget, FORM_CLASS):
    closed = pyqtSignal()

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.iface = iface
        self.bridge: EventBridge | None = None
        self.aoi_geometry = None
        self.aoi_crs = None
        self.last_out_dir: str | None = None
        self.manifest_path: str = ""

        self._aoi_tool = None
        #: 当前作业的成果目录（增量提取时复用）
        self.last_out_dir: str | None = None
        #: 已加载到项目的图层，增量追加时写回这些文件
        self._box_layer_path: str | None = None
        self._building_layer_path: str | None = None
        #: 增量模式下累计的 det_id 前缀计数，避免与已有记录冲突
        self._incremental_seq = 0
        self.drawAoiButton.setEnabled(False)
        self.clearAoiButton.setEnabled(False)

        self.browseButton.clicked.connect(self._browse_raster)
        self.rasterCombo.currentIndexChanged.connect(self._on_combo_changed)
        self.refreshLayersButton.clicked.connect(self._refresh_layers)
        self.continueButton.clicked.connect(self._continue_extraction)
        QgsProject.instance().layersAdded.connect(self._refresh_layers)
        QgsProject.instance().layersRemoved.connect(self._refresh_layers)
        self.outputBrowseButton.clicked.connect(self._browse_output)
        self.manifestBrowseButton.clicked.connect(self._browse_manifest)
        self.modelPresetCombo.currentIndexChanged.connect(self._on_preset_changed)
        # 注意顺序：必须**先**定位 manifest，再加载预设。
        # 预设的权重路径依赖 manifest 里的 repo_root（插件装在 profile 里时，
        # 向上找不到仓库根，只能靠 manifest 指路）。之前顺序反了，
        # 导致预设把所有路径拼到插件目录下、全部显示"缺文件"（实战报错）。
        self.drawAoiButton.toggled.connect(self._toggle_aoi_tool)
        self.clearAoiButton.clicked.connect(self._clear_aoi)
        self.runButton.clicked.connect(self._run)
        self.cancelButton.clicked.connect(self._cancel)
        self.loadResultsButton.clicked.connect(self._load_results)
        self.openReportButton.clicked.connect(self._open_report)

        self._load_default_manifest()
        self._reload_presets()          # 必须在 manifest 之后
        self._refresh_layers()

        # 限制面板宽度：组合详情、日志、路径这些长文本不应把停靠面板撑到超屏。
        # 之前详情显示完整绝对路径（100+ 字符），面板被撑到 1000+ px。
        self._limit_panel_width()

    def _limit_panel_width(self) -> None:
        """让长文本在内部换行，而不是把停靠面板撑到超出屏幕。

        注意：**不要**对 QDockWidget 自身调 setMaximumWidth/setMinimumWidth ——
        实测会在 __init__ 里触发布局重算并卡死（进程无输出直接挂住）。
        只约束内部控件即可：给长文本控件设置可压缩的宽度策略。
        """
        from qgis.PyQt.QtWidgets import QSizePolicy

        for name in ("modelDetailView", "rasterInfoLabel", "aoiInfoLabel"):
            w = getattr(self, name, None)
            if w is None:
                continue
            w.setWordWrap(True)
            w.setMinimumWidth(1)
            sp = w.sizePolicy()
            sp.setHorizontalPolicy(QSizePolicy.Ignored)
            w.setSizePolicy(sp)

        # 详情标签：限制最大高度，避免它按内容把面板撑很高
        detail = getattr(self, "modelDetailView", None)
        if detail is not None:
            detail.setMaximumHeight(150)

        # 日志区：固定高度区间，按宽度换行
        log = getattr(self, "logEdit", None)
        if log is not None:
            log.setMinimumWidth(1)
            log.setMinimumHeight(80)
            log.setMaximumHeight(160)
            sp = log.sizePolicy()
            sp.setHorizontalPolicy(QSizePolicy.Ignored)
            log.setSizePolicy(sp)

    # ---------------------------------------------------------- 初始化

    def _load_default_manifest(self) -> None:
        """找到 deployment/manifest.json。

        插件通常装在 QGIS profile 的 python/plugins 下，向上找不到仓库目录，
        所以按以下顺序找：
        1. 环境变量 ``VECROT_MANIFEST``（部署时显式指定，最可靠）
        2. QGIS 设置 ``vecrot/manifest_path``（用户在面板里选过一次就记住）
        3. 插件目录向上若干层（开发时插件就在仓库内）
        4. 常见部署位置
        """
        from qgis.core import QgsSettings

        candidates: list[str] = []
        env = os.environ.get("VECROT_MANIFEST")
        if env:
            candidates.append(env)
        saved = QgsSettings().value("vecrot/manifest_path", "", type=str)
        if saved:
            candidates.append(saved)

        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        for _ in range(5):
            candidates.append(os.path.join(here, "deployment", "manifest.json"))
            parent = os.path.dirname(here)
            if parent == here:
                break
            here = parent

        home = os.path.expanduser("~")
        candidates += [
            os.path.join(home, "vecrot", "deployment", "manifest.json"),
            os.path.join("C:\\", "vecrot", "deployment", "manifest.json"),
            os.path.join("D:\\", "vecrot", "deployment", "manifest.json"),
        ]

        for path in candidates:
            if path and os.path.isfile(path):
                self.manifest_path = os.path.normpath(path)
                self.manifestEdit.setText(self.manifest_path)
                self._log(f"使用部署 manifest：{self.manifest_path}")
                return

        self._log("⚠ 未找到 deployment/manifest.json。请到「高级设置 → 部署 manifest」"
                  "选择，或设置环境变量 VECROT_MANIFEST 指向它。")

    # ---------------------------------------------------------- 输入影像

    def _browse_raster(self) -> None:
        """直接选本地文件（影像不在当前项目里时用）。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择影像", "", "栅格影像 (*.tif *.tiff *.vrt);;所有文件 (*)")
        if not path:
            return
        self._apply_raster(path)

    # ---- 从项目图层选择

    def _project_rasters(self) -> list:
        """当前项目里可作为输入的栅格图层。

        排除 WMS/WMTS/WCS 等网络服务层：它们没有可读的本地仿射，
        首版要求先用受控预处理导出带地理参考栅格（§6.1）。
        """
        from qgis.core import QgsMapLayer, QgsRasterLayer

        out = []
        for layer in QgsProject.instance().mapLayers().values():
            if not isinstance(layer, QgsRasterLayer):
                continue
            if layer.type() != QgsMapLayer.RasterLayer:
                continue
            provider = (layer.providerType() or "").lower()
            if provider in ("wms", "wmts", "wcs"):
                continue
            src = layer.source() or ""
            path = src.split("|")[0]
            if not path or not os.path.isabs(path) or not os.path.isfile(path):
                continue
            if not layer.crs().isValid():
                continue
            out.append((layer.name(), path, layer))
        return out

    def _refresh_layers(self, *_args) -> None:
        """把项目里的栅格填进下拉框，尽量保持当前选择。"""
        current = self.rasterPathEdit.text().strip()
        items = self._project_rasters()

        self.rasterCombo.blockSignals(True)
        self.rasterCombo.clear()
        if not items:
            self.rasterCombo.addItem("（项目里没有可用的栅格图层）", "")
            self.rasterCombo.setEnabled(False)
        else:
            self.rasterCombo.setEnabled(True)
            for name, path, _layer in items:
                self.rasterCombo.addItem(name, path)
            idx = self.rasterCombo.findData(current)
            self.rasterCombo.setCurrentIndex(idx if idx >= 0 else 0)
        self.rasterCombo.blockSignals(False)

        if not current and items:
            self._on_combo_changed(0)

    def _on_combo_changed(self, index: int) -> None:
        if index < 0:
            return
        path = self.rasterCombo.itemData(index) or ""
        if path:
            self._apply_raster(path, sync_combo=False)

    def _apply_raster(self, path: str, sync_combo: bool = True) -> None:
        """统一入口：本地文件与项目图层都走这里。"""
        self.rasterPathEdit.setText(path)
        self._inspect_raster(path)
        if sync_combo:
            idx = self.rasterCombo.findData(path)
            if idx >= 0 and idx != self.rasterCombo.currentIndex():
                self.rasterCombo.blockSignals(True)
                self.rasterCombo.setCurrentIndex(idx)
                self.rasterCombo.blockSignals(False)
        if not self.outputDirEdit.text().strip():
            self.outputDirEdit.setText(os.path.dirname(path))

    def _inspect_raster(self, path: str) -> None:
        """用 QGIS 自己读元信息做即时反馈；权威预检在协调器里。"""
        # 只读元信息，不加入项目。必须显式断开数据源再丢弃：
        # 局部 QgsRasterLayer 若带着打开的 GDAL 数据源被 Python GC 回收，
        # 会在 QGIS 关闭/插件卸载时段错误（已实测 exit=139）。
        layer = QgsRasterLayer(path, os.path.basename(path), "gdal")
        try:
            if not layer.isValid():
                self.rasterInfoLabel.setText("⚠ 影像无法加载（不是有效栅格或缺少驱动）")
                self.drawAoiButton.setEnabled(False)
                return
            crs = layer.crs()
            if not crs.isValid():
                self.rasterInfoLabel.setText("⚠ 影像缺少 CRS：本工具不猜 EPSG，请先补齐定义")
                self.drawAoiButton.setEnabled(False)
                return
            ext = layer.extent()
            res_x = layer.rasterUnitsPerPixelX()
            msg = (f"{layer.width()}×{layer.height()} px · {layer.bandCount()} 波段 · "
                   f"{crs.authid() or crs.description()}")
            if res_x:
                msg += f" · 分辨率 {res_x:.3g}"
            if crs.isGeographic():
                msg += "\n⚠ 地理坐标系：请确认栅格拉伸与 AOI 面积含义"
            msg += (f"\n范围 {ext.xMinimum():.6g},{ext.yMinimum():.6g} → "
                    f"{ext.xMaximum():.6g},{ext.yMaximum():.6g}")
            self.rasterInfoLabel.setText(msg)
            self.drawAoiButton.setEnabled(True)
            self.clearAoiButton.setEnabled(True)
        finally:
            layer.setDataSource("", layer.name(), "gdal")

    # ---------------------------------------------------------- AOI

    def _toggle_aoi_tool(self, checked: bool) -> None:
        canvas = self.iface.mapCanvas()
        if checked:
            if self._aoi_tool is None:
                self._aoi_tool = AoiRectangleTool(canvas)
                self._aoi_tool.aoi_drawn.connect(self._on_aoi_drawn)
            canvas.setMapTool(self._aoi_tool)
            self._log("按住左键拖出矩形范围；框选会保留在图上，点本按钮退出框选")
        elif self._aoi_tool is not None:
            # 用 unsetMapTool 恢复上一个工具（通常是平移/缩放），
            # 不要 setMapTool(None)：那会让画布彻底失去工具，鼠标无法平移。
            # 退出时**保留** AOI 预览，用户要看到框的范围。
            canvas.unsetMapTool(self._aoi_tool)

    def _on_aoi_drawn(self, geometry, canvas_crs) -> None:
        try:
            self.aoi_geometry = geometry
            self.aoi_crs = canvas_crs
            area = geometry_area_m2(geometry, canvas_crs)
            text = f"已框选（{canvas_crs.authid() or canvas_crs.description()}）"
            if area:
                text += f"，面积约 {area/1e4:,.1f} 公顷"
            self.aoiInfoLabel.setText(text)
        except Exception as exc:                    # noqa: BLE001
            QMessageBox.warning(self, "AOI 不可用", str(exc))
        finally:
            # 框完自动退出框选模式：回到平移工具，矩形预览保留在画布上
            self.drawAoiButton.setChecked(False)

    def _clear_aoi(self) -> None:
        self.aoi_geometry = None
        self.aoi_crs = None
        if self._aoi_tool is not None:
            self._aoi_tool.clear()          # 清掉画布上的预览矩形
        self.aoiInfoLabel.setText("未框选")

    # ---------------------------------------------------------- 输出

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.outputDirEdit.setText(path)

    # ---------------------------------------------------------- 模型搭配

    def _reload_presets(self) -> None:
        """读固定模型组合，填进下拉框。

        会让 presets 模块**重新解析**仓库根：模块级的 ``_REPO`` 在 import 时
        就算好了，而那时 manifest 可能还没定位（插件装在 profile 里时尤其如此）。
        所以这里先调 ``refresh_repo_root()`` 再取预设列表。
        """
        from ..jobs import presets as preset_mod

        preset_mod.refresh_repo_root()
        available_presets = preset_mod.available_presets
        describe = preset_mod.describe

        self._presets = available_presets()
        self._preset_info = {d["key"]: d for d in describe()}

        self.modelPresetCombo.blockSignals(True)
        self.modelPresetCombo.clear()
        for p in self._presets:
            self.modelPresetCombo.addItem(p.label, p.key)
        self.modelPresetCombo.blockSignals(False)

        if self._presets:
            self._on_preset_changed(0)
        else:
            self.modelDetailView.setText("⚠ 没有可用的模型组合，请检查部署包是否完整")

    def _current_preset(self):
        key = self.modelPresetCombo.currentData()
        if not key:
            return None
        from ..jobs.presets import get_preset

        return get_preset(key)

    def _on_preset_changed(self, index: int) -> None:
        preset = self._current_preset()
        if preset is None:
            self.modelDetailView.setText("（未选择）")
            return
        info = self._preset_info.get(preset.key, {})
        lines = [
            "后端：%s%s" % (
                preset.backend,
                (" · 类别数 %d" % preset.num_classes) if preset.num_classes else ""),
        ]
        for label, path in preset.files.items():
            mark = "✓" if os.path.exists(path) else "✗"
            # 只显示文件名/目录名：完整绝对路径上百字符会把面板撑到超出屏幕
            # （实测面板被撑到 1000+px）。完整路径放 tooltip。
            shown = os.path.basename(os.path.normpath(path)) or "（未配置）"
            lines.append("%s %s：%s" % (mark, label, shown))
        if info.get("missing"):
            short = []
            for item in info["missing"]:
                if ":" in item:
                    role, full = item.split(":", 1)
                    base = os.path.basename(os.path.normpath(full.strip())) or full.strip()
                    short.append("%s（%s）" % (role.strip(), base))
                else:
                    short.append(item)
            lines.append("⚠ 缺：" + "；".join(short))
            lines.append("（完整路径见下方日志或 tooltip）")
        if preset.note:
            lines.append(preset.note)
        self.modelDetailView.setText(chr(10).join(lines))
        tip = ["完整路径："]
        for label, path in preset.files.items():
            tip.append("%s = %s" % (label, path))
        self.modelDetailView.setToolTip(chr(10).join(tip))
        can_run = bool(info.get("available"))
        self.runButton.setEnabled(can_run and self.bridge is None)
        if not can_run:
            self._log("⚠ 模型组合 %s 缺少文件，无法运行" % preset.key)
            for miss in info["missing"]:
                self._log("   " + miss)

    def _model_overrides(self) -> dict:
        """把选中预设展开成 job.json 的 inference 字段。"""
        preset = self._current_preset()
        if preset is None:
            raise JobSpecProblem("MODEL_PRESET_MISSING",
                                 "请先在下拉框选择一个模型搭配")
        miss = preset.missing()
        if miss:
            raise JobSpecProblem("MODEL_PRESET_INCOMPLETE",
                                 "所选模型组合缺少文件：" + "；".join(miss))
        return preset.to_job_fields()

    def _browse_manifest(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择部署 manifest", "", "manifest (*.json)")
        if not path:
            return
        try:
            read_manifest(path)                 # 立即验证可解析
        except Exception as exc:                # noqa: BLE001
            QMessageBox.warning(self, "manifest 无效", str(exc))
            return
        from qgis.core import QgsSettings

        self.manifest_path = path
        self.manifestEdit.setText(path)
        # 记住这次选择，下次启动插件不用再选
        QgsSettings().setValue("vecrot/manifest_path", path)
        self._log(f"已切换部署 manifest：{path}（已记住）")

    # ---------------------------------------------------------- 运行

    def _run(self) -> None:
        self._start_job(incremental=False)

    def _continue_extraction(self) -> None:
        """在已有成果上继续推理新框选的区域。"""
        if not self.last_out_dir:
            QMessageBox.information(
                self, "还没有成果",
                "请先完成一次完整提取，然后再框选新区域继续。")
            return
        if self.aoi_geometry is None:
            QMessageBox.information(
                self, "请先框选", "请在地图上框选一个新的区域，再点「继续提取」。")
            return
        self._start_job(incremental=True)

    def _aoi_payload(self) -> tuple[dict, str]:
        """把当前框选转成 (GeoJSON, CRS)。保持画布 CRS，跨 CRS 由协调器处理。"""
        if self.aoi_geometry is None:
            raise ValueError("请先在地图上框选范围")
        canvas_crs = self.aoi_crs
        if canvas_crs is None or not canvas_crs.isValid():
            raise ValueError("框选时拿不到画布 CRS，请重新框选")
        return ensure_polygon_in_crs(self.aoi_geometry, canvas_crs, canvas_crs)

    def _start_job(self, incremental: bool) -> None:
        if self.bridge is not None and self.bridge.is_running():
            QMessageBox.information(self, "作业进行中", "请等待当前作业结束或先取消")
            return
        raster = self.rasterPathEdit.text().strip()
        out_dir = self.outputDirEdit.text().strip()
        if not raster:
            QMessageBox.warning(self, "缺少影像",
                                "请在上方下拉框选择项目里的栅格图层，或用 … 选本地文件")
            return
        if not out_dir:
            QMessageBox.warning(self, "缺少输出目录", "请选择输出目录")
            return

        try:
            manifest = read_manifest(self.manifest_path)
            python_exe = coordinator_python(manifest)
        except Exception as exc:                    # noqa: BLE001
            QMessageBox.critical(self, "部署配置不可用", str(exc))
            return

        try:
            aoi_geojson, aoi_crs = self._aoi_payload()
        except Exception as exc:                    # noqa: BLE001
            QMessageBox.warning(self, "AOI 不可用", str(exc))
            return

        if incremental:
            job_id = "vecrot_inc_" + now_stamp()
            self._incremental_seq += 1
        else:
            job_id = "vecrot_" + now_stamp()

        run_dir = os.path.join(out_dir, job_id)
        try:
            overrides = self._model_overrides()
        except JobSpecProblem as exc:
            QMessageBox.warning(self, "模型设置有误", f"{exc.code}: {exc.message}")
            return
        try:
            payload = build_job_payload(
                raster_path=raster, aoi_geojson=aoi_geojson, aoi_crs=aoi_crs,
                output_dir=run_dir,
                score_threshold=self.scoreSpin.value(),
                vector_batch=self.batchSpin.value(),
                aoi_policy=("clip_strict" if self.aoiPolicyCombo.currentIndex() == 1
                            else "intersect_keep_complete"),
                crs_policy=self.crsPolicyEdit.text().strip() or "source",
                job_id=job_id,
                detector_profile=overrides.pop("detector_profile", None),
                model_overrides=overrides,
            )
        except JobSpecProblem as exc:
            QMessageBox.warning(self, "参数有问题", f"{exc.code}: {exc.message}")
            return

        repo_root = repo_root_from(self.manifest_path)
        work_root = os.path.join(repo_root, "work")
        job_path = write_job_file(payload, work_root)

        # 记录本次是否为增量，收尾时据此决定「追加」还是「新建图层组」
        self._pending_incremental = incremental
        self._pending_run_dir = run_dir

        self.bridge = EventBridge(self)
        self.bridge.event_received.connect(self._on_event)
        self.bridge.log_line.connect(self._log)
        self.bridge.finished.connect(self._on_finished)
        cancel_file = os.path.join(work_root, "jobs", job_id, "cancel.flag")

        self._set_running(True)
        self.progressBar.setValue(0)
        self.progressBar.setMaximum(0)
        mode = "增量" if incremental else "完整"
        self.stageLabel.setText(f"启动外部协调器（{mode}）…")
        self._log(f"{mode}作业 {job_id} 启动；输出 {run_dir}")
        try:
            self.bridge.start(
                python_exe,
                ["-u", "-m", "backend.cli", "run",
                 "--job", job_path,
                 "--manifest", self.manifest_path,
                 "--work", work_root,
                 "--cancel-file", cancel_file],
                cwd=repo_root,
                cancel_file=cancel_file,
            )
        except Exception as exc:                    # noqa: BLE001
            self._set_running(False)
            QMessageBox.critical(self, "无法启动作业", str(exc))

    def _cancel(self) -> None:
        if self.bridge is not None and self.bridge.is_running():
            self.stageLabel.setText("正在结束当前批次…")
            self.cancelButton.setEnabled(False)
            self.bridge.request_cancel()

    # ---------------------------------------------------------- 事件

    def _on_event(self, ev: dict) -> None:
        event = ev.get("event")
        stage = ev.get("stage")
        if event == "stage_started":
            self.stageLabel.setText(f"{STAGE_TEXT.get(stage, stage)}…")
            self.progressBar.setMaximum(0)
        elif event == "progress":
            total = ev.get("total")
            done = ev.get("done")
            if total:
                self.progressBar.setMaximum(int(total))
                self.progressBar.setValue(int(done or 0))
            else:
                self.progressBar.setMaximum(0)
            if ev.get("message"):
                self.stageLabel.setText(f"{STAGE_TEXT.get(stage, stage)}："
                                        f"{ev['message']}")
        elif event == "warning":
            self._log(f"⚠ {ev.get('message', '')}")
        elif event == "failed":
            self.stageLabel.setText("失败")
            code = ev.get("code") or "UNKNOWN"
            self._log(f"✗ {code}: {ev.get('message', '')}")
            QMessageBox.critical(self, "作业失败",
                                 f"{code}\n\n{ev.get('message', '')}")
        elif event == "cancelled":
            self.stageLabel.setText("已取消")
            self._log("作业已取消；已完成部分保留在 work/ 下")
        elif event == "completed":
            extra = ev.get("extra") or {}
            outputs = extra.get("outputs") or {}
            this_run = os.path.dirname(outputs["manifest"]) if outputs.get("manifest") else None
            counts = extra.get("counts") or {}
            self.stageLabel.setText(
                f"{ev.get('message', '完成')}（框 {counts.get('kept_boxes', 0)} / "
                f"建筑 {counts.get('buildings_written', 0)}）")
            if getattr(self, "_pending_incremental", False):
                # 增量作业：last_out_dir 必须保持为**基准成果目录**，
                # 否则 _append_incremental 会把增量目录追加到它自己，等于没追加
                # （实测症状：日志报「框 +8」，磁盘数量不变）。
                self._log(f"✓ 增量成果：{this_run}（待并入 {self.last_out_dir}）")
                self.loadResultsButton.setEnabled(bool(self.last_out_dir))
                self.openReportButton.setEnabled(bool(self.last_out_dir))
            else:
                if this_run:
                    self.last_out_dir = this_run
                self.loadResultsButton.setEnabled(bool(self.last_out_dir))
                self.openReportButton.setEnabled(bool(self.last_out_dir))
                self._log(f"✓ 成果：{self.last_out_dir}")

    def _on_finished(self, code: int, summary: dict) -> None:
        self._set_running(False)
        self.progressBar.setMaximum(1)
        self.progressBar.setValue(1)
        if code == 0:
            self.stageLabel.setText("完成")
        elif code == 10:
            self.stageLabel.setText("部分成功（部分轮廓未生成，框已保留）")
        elif code == 20:
            self.stageLabel.setText("已取消")
        elif code == -1:
            self.stageLabel.setText("协调器无法启动")
        else:
            self.stageLabel.setText(f"失败（退出码 {code}）")
            self._log(f"进程退出码 {code}")

        # 作业成功结束后自动把成果加载到地图（用户不必再点一次）
        if code in (0, 10):
            run_dir = getattr(self, "_pending_run_dir", None)
            if getattr(self, "_pending_incremental", False) and run_dir:
                # 增量：合并进基准成果并刷新已有图层
                self._append_incremental(run_dir)
            elif run_dir:
                # 完整：记住成果目录，然后自动加载
                self.last_out_dir = run_dir
                self.continueButton.setEnabled(True)
                self._auto_load(run_dir)
        self._pending_incremental = False

    def _auto_load(self, out_dir: str) -> None:
        """作业完成后自动加载成果图层（失败只记日志，不弹窗打断）。"""
        if not self.autoLoadCheck.isChecked():
            self._log("（自动加载已关闭，可手动点「加载成果到地图」）")
            return
        try:
            from ..layers.result_loader import load_results

            group = load_results(QgsProject.instance(), out_dir)
            n = len(group.children())
            self._log(f"已自动加载 {n} 个成果图层到地图")
            self.loadResultsButton.setEnabled(True)
            self.openReportButton.setEnabled(True)
        except Exception as exc:                    # noqa: BLE001
            self._log(f"⚠ 自动加载成果失败：{type(exc).__name__}: {exc}")
            self._log("   可手动点「加载成果到地图」重试")

    # ---------------------------------------------------------- 增量合并

    def _append_incremental(self, run_dir: str) -> None:
        """把一次增量作业的 GPKG 追加到当前成果文件，并刷新项目图层。"""
        if not self.last_out_dir:
            self._log("⚠ 增量作业结束但找不到基准成果目录，改为按新成果加载")
            self._load_results_from(run_dir)
            return
        new_boxes = os.path.join(run_dir, "boxes.gpkg")
        new_buildings = os.path.join(run_dir, "buildings.gpkg")
        if not os.path.isfile(new_boxes):
            self._log(f"⚠ 增量作业没有产出 boxes.gpkg：{new_boxes}")
            return
        try:
            from ..layers.result_loader import append_gpkg, refresh_layers_from

            added_boxes = append_gpkg(
                os.path.join(self.last_out_dir, "boxes.gpkg"), new_boxes, "boxes",
                id_prefix=f"inc{self._incremental_seq}_")
            added_buildings = append_gpkg(
                os.path.join(self.last_out_dir, "buildings.gpkg"), new_buildings,
                "buildings", id_prefix=f"inc{self._incremental_seq}_")
            self._log(f"增量追加：框 +{added_boxes}，建筑 +{added_buildings}")
            if not self.autoLoadCheck.isChecked():
                self._log("（自动加载已关闭；要素已写入文件，可手动加载）")
                return
            total = refresh_layers_from(QgsProject.instance(), self.last_out_dir)
            self._log(f"已刷新项目图层：{total}")
            self.stageLabel.setText(
                f"增量完成（新增 {added_boxes} 框 / {added_buildings} 建筑）")
            self._clear_aoi_hint()
        except Exception as exc:                    # noqa: BLE001
            import traceback

            self._log(f"✗ 增量合并失败：{exc}")
            self._log(traceback.format_exc())
            QMessageBox.warning(
                self, "增量合并失败",
                "{}".format(exc) + "\n\n本次成果仍保存在：\n" + run_dir
                + "\n可单独打开。")

    def _clear_aoi_hint(self) -> None:
        """增量结束后提示可以继续框选下一块。"""
        self.aoiInfoLabel.setText("已框选（可继续框选新区域再点「继续提取」）")

    def _load_results_from(self, out_dir: str) -> None:
        from ..layers.result_loader import load_results

        try:
            load_results(QgsProject.instance(), out_dir)
            self.last_out_dir = out_dir
            self.loadResultsButton.setEnabled(True)
            self.openReportButton.setEnabled(True)
            self._log(f"成果已加载：{out_dir}")
        except Exception as exc:                    # noqa: BLE001
            QMessageBox.warning(self, "加载成果失败", str(exc))

    def _set_running(self, running: bool) -> None:
        self.runButton.setEnabled(not running)
        self.cancelButton.setEnabled(running)
        self.browseButton.setEnabled(not running)
        self.refreshLayersButton.setEnabled(not running)
        self.rasterCombo.setEnabled(not running and self.rasterCombo.count() > 0
                                   and bool(self.rasterCombo.itemData(0)))
        self.continueButton.setEnabled(not running and bool(self.last_out_dir))
        self.outputBrowseButton.setEnabled(not running)
        self.drawAoiButton.setEnabled(not running and bool(self.rasterPathEdit.text()))
        self.scoreSpin.setEnabled(not running)
        self.batchSpin.setEnabled(not running)

    # ---------------------------------------------------------- 结果

    def _load_results(self) -> None:
        if not self.last_out_dir:
            return
        try:
            load_results(QgsProject.instance(), self.last_out_dir)
            self._log("成果已加载到地图")
            self.continueButton.setEnabled(True)
            self.incrementalInfoLabel.setText(
                "已加载成果。之后框选新区域点「继续提取」，结果会追加到同一图层。")
        except Exception as exc:                    # noqa: BLE001
            QMessageBox.warning(self, "加载成果失败", str(exc))

    def _open_report(self) -> None:
        if not self.last_out_dir:
            return
        if open_report(self.last_out_dir) is None:
            QMessageBox.information(self, "报告不存在", "该作业还没有 report.json")

    # ---------------------------------------------------------- 杂项

    def _log(self, text: str) -> None:
        self.logEdit.appendPlainText(text)
        self.logEdit.verticalScrollBar().setValue(
            self.logEdit.verticalScrollBar().maximum())

    def cleanup(self) -> None:
        """卸载/关闭时调用：取消作业并回收其创建的进程。"""
        if self.bridge is not None and self.bridge.is_running():
            self.bridge.request_cancel(grace_ms=5000)
        if self._aoi_tool is not None:
            self._aoi_tool.deactivate()

    def closeEvent(self, event) -> None:
        self.cleanup()
        self.closed.emit()
        super().closeEvent(event)
