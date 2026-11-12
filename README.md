# Vecrot — QGIS 建筑提取插件

在 QGIS 中框选影像区域，自动完成建筑检测和 VectorLLM 轮廓生成，输出检测框与建筑轮廓两个 GeoPackage 文件。

支持 Windows 本地部署。模型在独立 Python 环境中运行，不向 QGIS Python 安装推理依赖。

## 安装

### 1. 下载代码和模型

下载或克隆本仓库 `vecrot`，可以放在任意盘符、任意目录。

从 [百度网盘下载整个 models 文件夹](https://pan.baidu.com/s/1e6lr0Fk2Ud4lWQYXK58ecA?pwd=c83w)，提取码：`c83w`。将它放到仓库内的 `vecrot/` 目录，保持包内文件名和目录结构：

```text
vecrot/                  # 仓库根目录
├─ install.py
├─ uninstall.py
├─ backend/
├─ vecrot_qgis/
└─ vecrot/
   ├─ lib/
   └─ models/            # 网盘下载的整个文件夹
```

使用与代码同版本的 models 包，无需逐个下载模型。不要放成 `vecrot/models/models/`。

### 2. 建立运行环境

需要 QGIS 3.x、NVIDIA 显卡及以下三个外部 conda 环境：

| 环境 | Python | 用途 |
|---|---|---|
| `vecrot-geo` | 3.11 | 影像读取、坐标转换、矢量导出 |
| `vecrot-det` | 3.9 | Strip-R-CNN 旋转框检测 |
| `vecrot-vllm` | 3.11 | VectorLLM 轮廓生成 |

按 [环境安装教程](docs/安装教程-自建环境.md) 建立环境并安装对应版本的依赖。安装脚本不负责下载模型或创建环境。

### 3. 安装插件

关闭 QGIS，在仓库根目录打开 PowerShell 7，使用外部 Python 执行：

```powershell
python -X utf8 install.py
```

安装程序会检查环境和模型文件、安装插件并配置路径。完成后重启 QGIS，在 Vecrot 菜单中打开插件。更新代码后，重新运行同一命令即可更新插件。

## 使用

1. 在 QGIS 中加载本地、带地理参考的影像。
2. 在插件中选择影像和模型预设，框选处理区域。
3. 点击开始，完成后自动加载 `boxes.gpkg`（检测框）和 `buildings.gpkg`（建筑轮廓）。

默认推荐 Strip-R-CNN 场景版；也支持 RT-DETR 水平框检测和读取已有检测结果生成轮廓。

## 卸载

关闭 QGIS，在仓库根目录执行：

```powershell
python -X utf8 uninstall.py
```

插件会移入备份目录，模型、运行环境和已有成果保留。

## 详细说明

- [模型包放置与校验](docs/模型下载.md)
- [环境安装与排错](docs/安装教程-自建环境.md)
- [自检、开发调试及指定 QGIS profile](docs/插件安装卸载与本机调试.md)
- [技术状态与已验证范围](docs/STATUS.md)

安装后移动仓库，需要在新位置重新运行安装命令。建筑轮廓仍需人工检查，尤其是密集建筑和影像边缘。

## 许可与致谢

代码许可见 [LICENSE](LICENSE)，模型使用遵循各自上游许可。感谢 Strip-R-CNN、VectorLLM 和 RT-DETR。
