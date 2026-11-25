# Vecrot — QGIS 建筑提取插件

在 QGIS 中框选影像区域，自动完成建筑检测和 VectorLLM 轮廓生成，输出检测框与建筑轮廓两个 GeoPackage 文件。

支持 Windows 本地部署。模型在独立 Python 环境中运行，不向 QGIS Python 安装推理依赖。

## 安装

### 1. 下载代码，放好模型

代码全部在 GitHub，模型文件全部在网盘。**克隆仓库，再把模型包解压进去就能用**：

```powershell
git clone https://github.com/feicai922/vecrot.git
Set-Location .\vecrot
```

从 [百度网盘下载 models.zip](https://pan.baidu.com/s/1e6lr0Fk2Ud4lWQYXK58ecA?pwd=c83w)（提取码 `c83w`），
解压后放到仓库里的 `vecrot/models/`：

```text
vecrot/                     # 仓库根目录（git clone 得到）
├─ install.py   uninstall.py
├─ backend/     vecrot_qgis/  vecrot/lib/
├─ third_party/Strip-R-CNN/    # 检测源码（随 GitHub）
├─ third_party/RiO-DETRv2/     # RiO-DETR 源码（随 GitHub）
└─ vecrot/
   └─ models/                # ← models.zip 解压到这里（各子目录 = 面板上的一个模型）
```

不要放成 `vecrot/models/models/`。模型也可以放在仓库外（不想让几个 GB 的权重躺在
clone 目录里）：

```powershell
setx VECROT_MODELS_ROOT "D:\vecrot-models"     # 或写进 deployment/manifest.json 的 models_root
```

### 2. 建立运行环境

需要 QGIS 3.x、NVIDIA 显卡及以下三个外部 conda 环境：

| 环境 | Python | 用途 |
|---|---|---|
| `vecrot-geo` | 3.11 | 影像读取、坐标转换、矢量导出 |
| `vecrot-det` | 3.9 | Strip-R-CNN 旋转框检测 |
| `vecrot-vllm` | 3.11 | VectorLLM 轮廓生成（RT-DETR 也复用这套） |
| `vecrot-rio` | 3.10 | 可选：RiO-DETR / RiO-DETRv2 旋转框检测 |

按 [环境安装教程](docs/安装教程-自建环境.md) 建立环境并安装对应版本的依赖。安装脚本不负责下载模型或创建环境。

### 3. 安装插件

关闭 QGIS，在仓库根目录打开 PowerShell 7，使用外部 Python 执行：

```powershell
python -X utf8 install.py
```

安装程序会检查环境和模型文件、安装插件并配置路径。完成后重启 QGIS，在 Vecrot 菜单中打开插件。更新代码后，重新运行同一命令即可更新插件。

## 使用

1. 在 QGIS 中加载本地、带地理参考的影像。
2. 在插件中选择影像和模型搭配，框选处理区域。
3. 点击开始，完成后自动加载 `boxes.gpkg`（检测框）和 `buildings.gpkg`（建筑轮廓）。

「模型搭配」下拉框的每一项对应模型根目录下的一个文件夹（文件夹里的 `model.json`
声明后端、权重、配置、类别数与阈值）。**加模型只需拷文件夹，不用改代码**，
例如融合数据集上重训的 Strip-R-CNN-S、或自己训的 RiO-DETRv2：
一条 `python deployment\add_model_folder.py` 就生成好。
详见 [模型文件夹约定](docs/模型下载.md#模型文件夹约定重要)。

默认推荐 Strip-R-CNN 场景版；也支持 RT-DETR 水平框、RiO-DETR 旋转框，
以及读取已有检测结果只跑轮廓。

## 卸载

关闭 QGIS，在仓库根目录执行：

```powershell
python -X utf8 uninstall.py
```

**此命令只卸载 QGIS 插件，不删除三个 conda 环境（`vecrot-geo`、`vecrot-det`、`vecrot-vllm`），也不删除模型、源码或已有成果。** 插件文件会移入备份目录。

重新克隆 GitHub 仓库后，可复用这些环境，将模型包放到新仓库的 `vecrot/models/`，再运行 `python -X utf8 install.py`。安装在非默认 profile 时，卸载命令需附加 `--qgis-profile 对应名称`。

## 详细说明

- [模型包放置、模型文件夹约定与校验](docs/模型下载.md)
- [环境安装与排错](docs/安装教程-自建环境.md)
- [自检、开发调试及指定 QGIS profile](docs/插件安装卸载与本机调试.md)
- [技术状态与已验证范围](docs/STATUS.md)

安装后移动仓库，需要在新位置重新运行安装命令。建筑轮廓仍需人工检查，尤其是密集建筑和影像边缘。

## 许可与致谢

项目代码许可见 [LICENSE](LICENSE)，第三方源码遵循其目录内的 [LICENSE](third_party/Strip-R-CNN/LICENSE)，模型使用遵循各自上游许可。感谢 Strip-R-CNN、VectorLLM、RT-DETR 与 RiO-DETR。
