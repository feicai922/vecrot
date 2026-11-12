# 当前状态（2026-11-10 实机验证）

对应开发文档：`outputs/Vecrot_QGIS_Windows本地集成_详细开发文档.md`

**结论：检测与轮廓两个模型都已在本机跑通，QGIS 插件已在真实 QGIS 里验证，真实数据端到端作业成功。**

---

## 1. 里程碑

| 里程碑 | 状态 | 依据 |
|---|---|---|
| M0 资产和环境审计 | **完成** | GPU/conda/QGIS/依赖/权重/上游源码全部实测枚举 |
| M1 独立模型验证 | **完成** | 检测 R=0.67/P=0.74/F1=0.70（44 张真实瓦片）；轮廓 4/4→6/6 成功 |
| M2 无 QGIS 端到端闭环 | **完成** | 真实 GeoTIFF 跑通六阶段，71 项 pytest 全过 |
| M3 薄插件 | **完成** | QGIS 3.44.14 `startPlugin` 加载，56 项实机检查全过 |
| M4 稳定性与分发 | **部分完成** | 取消/回收/预检/重试/空结果/中文路径已验证；离线部署包未做 |
| M5 流水线优化 | **未实现** | `mode=pipelined` 明确报 `MODE_NOT_IMPLEMENTED` |

---

## 2. 实测环境

| 项 | 实测值 |
|---|---|
| GPU | NVIDIA T400 **4 GB**，驱动 552.74，sm_75 |
| CUDA | 11.8（torch cu118 轮子），可用 |
| QGIS | **3.44.14-Solothurn** / Python 3.12.14 / Qt 5.15.13 |
| vecrot-geo（协调器） | Python 3.11.18 + rasterio 1.4.4 / pyproj 3.7.2 / shapely 2.2.0 / pyogrio 0.12.1 |
| vecrot-det | **Python 3.9.25** + torch 2.0.1+cu118 / mmcv-full 1.7.2 / mmdet 2.28.2 / mmrotate 0.3.4 |
| vecrot-vllm | **Python 3.11.18** + torch 2.4.1+cu118 / transformers 4.57.1 / timm 1.0.31 |
| detector_hf（RT-DETR） | 复用 vecrot-vllm 环境（需 transformers + RTDetr API；与 mmcv 冲突） |
| Strip-R-CNN 源码 | MMRotate 0.3.4 完整分支 + 本地 `configs/gzda` 定制 |

环境隔离实测：QGIS 内部**没有** torch/transformers/mmcv/mmdet。

---

## 3. 检测链路

```
best_mAP_iter_90288.pth (541 MB) + configs/gzda/strip_s_gzda.py
  ↓ 派生纯推理配置（展开 _base_、剔除训练期字段、移除训练机路径）
build_detector → load_checkpoint
  → missing=0, unexpected=0   ← 权重与派生配置结构完全对齐
```

**结构差异确认**（gzda 相对上游）：`num_classes=1`、`backbone.norm_cfg=BN`。
`gzda_dataset.py` 只注册 DATASETS/HOOKS，**不碰模型结构**，故权重完整匹配。

### 真实数据精度

数据：`G:/贵州省数据集/images/test`（2399 张 1024² 瓦片，COCO 27938 框）

| 指标 | 值 |
|---|---|
| 评估瓦片 | 44 张（40 有建筑 + 4 无建筑） |
| GT / 检出 / 匹配 | 945 / 854 / 634 |
| Recall | **0.6709** |
| Precision | **0.7424** |
| F1 | **0.7048** |
| 无建筑瓦片误检 | **0** |
| 吞吐 / 峰值显存 | 16.6 框/s（T400） / **1.03 GB** |

口径：score≥0.3、旋转 IoU≥0.5 的瓦片级 OBB 检测，**不是 mask AP**。
复现：`deployment/envs/detector/python.exe deployment/eval_detection.py --limit 40`

### 三个只有实机才暴露的问题（已修）

1. **NMS 设备不匹配**：mmrotate 0.3.4 的 `multiclass_nms_rotated` 用
   `torch.arange(num_classes)` 建 labels（默认 CPU），torch 2.0 对
   「CPU 索引索引 GPU 张量」报错。运行期补丁覆盖该函数，不改用户源码。
2. **`mmrotate/datasets/__init__.py` 导入本分支不存在的
   `DOTAv15Dataset`/`DOTAv2Dataset`**（`dota.py` 精简过但 `__init__` 没同步）。
   用 `sys.modules` 桩绕过。
3. **输出解析错误**：`rbbox2result` 返回按类别分组的嵌套列表 `[[N×6]]`，
   旧代码当 `N×6` 处理，因 `ndim!=2` 整批跳过 → **真实影像 0 框**。

---

## 4. 轮廓链路

```
adaptor_sc_055m/trainable.pt（7 张量 / 0.55M，全是 scale_cond.*）
  ↓ 挂到重建的 635 词表基座
VectorLLM 加载 → 词表 635 一致、pixel_idx=377 对齐、ScaleCond 0.55M 挂载
```

### 关键问题：训练基座缺失（已重建解决）

`backbone.json` 指向 `/data1/.../vocab_prune/official_vocab` —— 已按 prune 映射**重排**的
635 词表基座，不在仓库里。仓库只有 `base_v1`（原始 Qwen3 全词表 151926 行）。

直接建模型再 `resize_token_embeddings(635)` 会截断取前 635 行，导致
`<pixel>`/`<xN>`/`<yN>` 的 embedding 行与 token 语义错位。
**实测症状：模型能前向、不报错，但输出 640 个换行、0 个坐标 token。**

**重建依据（逐 token 比对）：**

| 检查 | 结果 |
|---|---|
| adaptor token 能在 base 中找到同名 | **634/635** |
| 映射单调递增 | **633/633** |
| adaptor 0~376 → base 0~376 | 普通 BPE，原样保留 |
| adaptor 377~633 → base 151669~151925 | `<pixel>` + `<x0..127>` + `<y0..127>` |
| adaptor 634 `<no_object>` | base 无，用抽走行均值初始化 |

重建后校验：embedding 635 行、`pixel_idx` 377 对齐、**`<pixel>` 行与 base 对应行逐位一致**。

工具：`deployment/rebuild_pruned_base.py` → `vecrot/models/official_vocab_635`（1.53 GB），
附 `REBUILD_INFO.json`。

> **这不是训练时的原始文件。** 映射从「同名字符串 + 单调子集」推断。
> 若原 prune 有额外重排，重建版会与训练口径有差异 —— **正式评测前需与原环境对比输出**。

### 真实数据表现

| 项 | 值 |
|---|---|
| 真实检测框 4 个 | **4/4 产出轮廓**（各 5 顶点） |
| 轮廓中心距框中心 | 5.1 / 12.7 / 18.2 / 15.9 px |
| 吞吐 / 峰值显存 | 0.42 框/s（T400） / 1.50 GB |
| 端到端作业 | 6 框 → **6/6 轮廓成功**，0 弃权 0 失败 |

---

## 5. QGIS 插件

真实 QGIS 3.44.14 里 **104 项**全过：

| 检查 | 项数 | 内容 |
|---|---|---|
| `qgis_runtime_check.py` | 19 | classFactory、uic、面板默认值、AOI 转换、成果加载、unload |
| `qgis_plugin_mount_check.py` | 13 | 走 QGIS 自己的 `startPlugin`、工具栏、`unloadPlugin` |
| `qgis_job_check.py` | 24 | jsonl/null 后端端到端（管线验证） |
| `qgis_gui_flow_check.py` | 20 | **模拟 GUI 全流程 + 真模型** |
| `qgis_real_job_check.py` | 24 | **真模型端到端 + profile 安装场景** |

**已经可以实际使用**：`python deployment/install_plugin.py` 一条命令装好插件、
把 manifest 复制到仓库外并写 QgsSettings，重启 QGIS 即可在菜单里看到。

一键验证：`python deployment/verify.py`

### 联调修掉的 6 个 bug

1. `QgsGraduatedSymbolRenderer` 用 `ranges()` 不是 `categories()`
2. Qt5 下 `QgsGradientColorRamp.color()` 返回 int 不是 `QColor`
3. 图层在 `exitQgis()` 后被 GC → 退出段错误（exit=139）
4. manifest 相对路径基准错（把 `deployment/` 当仓库根）
5. **AOI（地理坐标）与模型输出（源像素）直接求交** → 所有建筑被静默过滤
6. **零面积"领结"多边形在 make_valid 前被判死** → 默默丢掉整栋楼

另有 3 个**只有部署到 profile 才会暴露**的问题（都已修）：

7. **插件装在 profile 里找不到 manifest** —— 「向上找 deployment/」在 profile 下永远
   失败。改为 `QgsSettings` → 环境变量 → 固定位置三级回退。
8. **manifest 复制到仓库外后 `repo_root` 推断错** —— `python -m backend.cli` 报
   `No module named 'backend'`。manifest 现在显式带 `repo_root`，安装脚本自动写入。
9. **`_inspect_raster` 里的局部 QgsRasterLayer 未断开数据源** —— 带打开状态被 GC
   回收会导致 QGIS 退出时段错误。改为 `finally: setDataSource("", ...)`。

---

## 5.1 本轮（2026-11-11）修复与新增

### 修复：轮廓比检测框少（用户实测报告）

用户日志显示 `polygons_ok: 143` 但 `buildings_written: 56`。根因是**AOI 过滤只作用于轮廓**：

* 检测窗口按 `tile_size=1024` 铺，必然覆盖到 AOI 之外
* 轮廓被 `intersect_keep_complete` 策略过滤，**检测框却全量写出**
* AOI 只有 722×435 时，143 个框里 85 个落在 AOI 外 → 只剩 56 个轮廓

修复（`backend/coordinator/runner.py`）：检测框与轮廓走**同一套** AOI 判定，
AOI 外的检测标记为 `outside_aoi` 且两个图层都不写出。
用用户的原始数据重放验证：`kept_boxes 53 == buildings_written 53`，`det_id` 集合完全相同。

### 优化：小 AOI 的窗口居中

之前不足一个 tile 的 AOI 按**左上对齐**铺窗口，多出的部分全在右/下侧，
那里检出的目标 100% 会被过滤。改为**居中**后（`backend/geo/transform.py`），
同一数据的 AOI 外浪费从 85 降到 45，有效建筑从 53 增到 55。

### 新增：固定模型搭配（预设）

不再让用户自由填权重/配置路径（容易配错），改为代码里写死的组合：

| 预设 | 后端 | 说明 |
|---|---|---|
| `strip_s_gzda_scene` | `strip_s_gzda` | **默认**。Strip-R-CNN-S + 场景分类，旋转框 |
| `rtdetr_roof` | `rtdetr_hf` | RT-DETR R101vd，水平框，单类别 |
| `vector_only` | `jsonl` | 只跑轮廓（读已有检测结果 jsonl） |

> 不带场景分类的旧预设 `strip_s_gzda_sc055m` 已从面板移除：功能被场景版完全覆盖，
> 且精简分发包不带它的权重。定义仍保留在代码里（`STRIP_GZDA`），需要时可重新加回 `PRESETS`。

* 预设定义在 `backend/contracts/presets.py`，插件侧有镜像
  `qgis_plugin/vecrot_qgis/jobs/presets.py`（插件装在没有 backend 包的 profile 里）
* 一致性由 `tests/test_presets_parity.py` 守住（9 项）
* 缺文件时面板显示 ✗ 并禁用「开始提取」

### 新增：RT-DETR 后端

`vecrot/models/detector_rt-detr_bbox` 是 **HF Transformers 格式**
（`RTDetrForObjectDetection` + safetensors），不是 mmdet，所以是独立后端
`backend/detector/rtdetr_backend.py`。

实测（`deployment/envs/vector`，transformers 4.57.1 + torch 2.4.1）：

| 项 | 结果 |
|---|---|
| 加载 | 5.3s |
| 单瓦片推理 | 2.68s（1024²→640） |
| 检出 | 12 框（与 Strip-R-CNN 位置高度吻合） |
| 峰值显存 | 0.42 GB |
| 端到端作业 | 14 框 = 14 轮廓，`state=SUCCEEDED` |

**环境切换**：RT-DETR 需要 transformers，mmcv 系需要 mmcv-full，两者不能共存。
manifest 新增 `environments.detector_hf`（复用 vector 环境），协调器按后端自动选。
这里踩过一个坑：把 `role` 同时当"模块类型"和"环境名"用，导致
`KeyError: 'detector_hf'`；已拆成 `module_role` / `env_role`。

### 交互改进

* **AOI 持久显示**：框完保留橙色矩形在画布上，切工具不消失，点「清除」才移除
* **自动加载成果**：作业完成后自动加载图层（可取消勾选）
* **展示分工**：检测框只画线（fill alpha=0，5 级按分数着色）；
  轮廓是红色实心多边形（fill alpha=165）
* **图层顺序**：轮廓在下、框在上（之前反了，框线压在轮廓上）
* **分级渲染器**：修了两个坑 —— `ranges()` 返回副本必须用 `updateRangeSymbol` 写回；
  且必须**先 `setRenderer` 再 `updateClasses`**，反过来 ranges 会变空
* **面板可滚动**：`QScrollArea` 包装，内容 1287px / 视口 480px 时正常出现滚动条

### 另修：面板显示"缺文件"且过宽（用户报告）

现象：所有文件显示 ✗，路径被拼成
`.../plugins/vecrot_qgis/jobs/vecrot/models/...`（插件目录下），面板被撑到超屏。

**根因一（初始化顺序）**：`_reload_presets()` 在第 65 行，`_load_default_manifest()`
在第 73 行——预设**先于** manifest 加载。插件装在 profile 里时，`_REPO` 只能靠
manifest 里的 `repo_root` 定位；顺序反了，`_REPO` 在 import 时就定成了插件目录。

修复：
* 调换顺序（manifest -> presets）
* 新增 `presets.refresh_repo_root()`，让模块级路径可重新解析
* `_repo_root()` 三级回退：向上找 `backend/` -> 向上找 `deployment/` ->
  读 QgsSettings 里 manifest 的 `repo_root` -> 环境变量 `VECROT_REPO_ROOT`

**根因二（面板过宽）**：
* 详情显示完整绝对路径（100+ 字符）-> 改为只显示文件名，完整路径放 tooltip
* 对 `QDockWidget` 自身调 `setMaximumWidth/setMinimumWidth` 会在 `__init__` 里
  **触发布局重算死循环**（进程无输出直接挂住，连异常都不抛）。
  改为只约束内部长文本控件（`QSizePolicy.Ignored` + 强制换行）
* 日志区固定 80~160px、按宽度换行、无水平滚动

修复后 `dock sizeHint = 390x427`，详情最长行 55 字符。

**过程中自己造成的坑**：重写文件时把 `available_presets` / `get_preset` / `describe`
三个函数整段删掉，面板初始化卡在调用处。已补回，并加测试
`test_preset_functions_exist` 守住。

### 新增：场景分类分支（跳过无建筑窗口）

用户提供了 `vecrot/models/detector_strip_s_scene`，是带 **scene_head** 的
`StripRCNNScene` 模型。大范围推理时先判「本窗口有没有建筑」，无建筑直接跳过
检测与轮廓。

#### 缺失模块的精确重建

配置的 `custom_imports` 引用了三个上游不存在的模块，其中两个必须补：

| 模块 | 状态 |
|---|---|
| `configs.gzda.gzda_dataset` | 已有（用户此前提供） |
| `configs.gzda.continue_hooks` | 已有（用户此前提供） |
| **`configs.gzda.scene_branch`** | **缺失 → 已按权重形状重建** |
| **`configs.gzda.scene_hooks`** | **缺失 → 已补最小实现** |

重建依据：从权重抽出 `scene_head` 只有 6 个张量——

    fc1.weight (512, 1280)   ← 5 层 × 256 通道，全局池化后拼接
    fc1.bias   (512,)
    norm.weight/bias (512)   ← LayerNorm
    fc2.weight (2, 512)      ← 二分类
    fc2.bias   (2,)

与配置的 `scene_head=dict(in_channels=256, num_levels=5, hidden_channels=512)`
完全吻合。**加载权重验证：`missing=0 unexpected=0`**，说明结构推断准确。

> 注意：这是按形状重建的**等价结构**，数值行为未与原训练脚本逐位比对。
> 分类阈值需按用户数据标定（默认 0.5，可在面板/ manifest / job.json 改）。

#### 实测

| 场景 | 结果 |
|---|---|
| 有建筑瓦片 | 场景概率 **0.9988**，10 框，1725 ms |
| 纯背景瓦片 | 场景概率 **0.1675**，**0 框**，492 ms（只算 backbone + 场景头） |
| 端到端（真实 AOI） | 58 框 = 58 轮廓，`scene_skipped=0` |
| 端到端（纯背景 2048²） | `skipped_windows=9/9`，候选框 0，两个合法空层，跳过概率 0.116~0.126 |

**跳过带来约 3.5× 加速**（纯背景 492 ms vs 有建筑 1725 ms），且轮廓阶段完全不启动。

#### 接入方式

* 新增预设 `strip_s_gzda_scene`，排在第一位（大范围推理推荐）
* `scene_threshold` 贯通 job.json → manifest → 模型；越界被 jobspec 拒绝
* report 新增 `scene_classification`：`skipped_windows` / `evaluated_windows` /
  被跳过窗口的概率区间，便于判断阈值是否合适

### 另修：框选直接报错（用户报告）

`QgsRubberBand(canvas, False)` 在 QGIS 3.44 要求第二个参数是
`QgsWkbTypes.PolygonGeometry` 枚举，传 bool 抛 TypeError。
同文件还有 `QgsRectangle(...).normalize()` 返回 None 的误用（PyQGIS 里是原地修改）。

**这两个 bug 之前没抓到，因为测试一直直接调 `_on_aoi_drawn()` 模拟框选，
从未真正实例化过 `AoiRectangleTool`。** 已补 `qgis_aoi_tool_check.py`（20 项），
真的构造工具、真的合成 press/move/release 鼠标事件。

---

## 6. 已验证清单

`python deployment/verify.py`：

| 检查 | 项数 |
|---|---|
| 插件静态检查 | 6 |
| pytest | 71 |
| 几何 + GeoPackage 自检 | 6 |
| 检测配置契约 | 37 |
| 重建基座校验 | 1 |
| QGIS 运行时三件套 | 19 + 13 + 24 = 56 |

模型侧（需对应环境）：

```powershell
deployment\envs\detector\python.exe deployment\verify_detector_runtime.py   # 22 项
deployment\envs\detector\python.exe deployment\eval_detection.py --limit 40 # 精度
deployment\envs\vector\python.exe   deployment\verify_vector_runtime.py     # 16 项
```

---

## 7. 仍未做（诚实清单）

| 项 | 说明 |
|---|---|
| **mask AP 评测** | 需 pycocotools；`vecrot/tools/eval_mask_ap.py` 已有，需装环境 |
| **与原环境输出逐实例对比** | 没有原环境推理输出可比 |
| **ScaleCond 训练口径核验** | 仍从代码语义推导；原训练入口不在仓库 |
| **重建基座等价性** | 推断映射非原文件；需原环境对比 |
| OOM 压测 | 只观察 1.0-1.5 GB 峰值，未受控降显存 |
| 大图整景长跑 | 只用窗口读取，未做 10 万级瓦片压力测试 |
| 导出故障注入 | 有 staging + 校验，未注入写盘失败 |
| 离线部署包 | 环境是 conda 目录，未验证搬迁 |
| 多 GPU / 流水线（M5） | 未实现 |

---

## 8. 环境版本的两个坑（复现必读）

1. **transformers 版本窗口很窄**
   - `≤4.51.3` 缺 `MultiModalData`（`processor.py` 需要）
   - `4.56+` 同时有 `MultiModalData` 与 `Qwen3`
   - `5.x` 要求 torch≥2.6
   - **选 4.57.1 + torch 2.4.1**

2. **两个环境必须用不同 torch**
   - detector：torch 2.0.1（mmcv-full 1.7.2 约束）
   - vector：torch 2.4.1（Qwen3 需要 `torch.compiler`）
   - 这是「三套环境隔离」的实际价值：不隔离不可能同时满足

3. **numpy 与 opencv 互相拉扯**：mmcv 1.x 要 numpy<2，opencv 5.x 要 numpy≥2。
   detector 环境固定 numpy 1.23.5 + opencv 4.8.1.78。
