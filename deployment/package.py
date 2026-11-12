"""把插件打包成可交给别人的分发包。

    python deployment/package.py                    # 完整包（含两个 conda 环境）
    python deployment/package.py --no-envs          # 只打代码+模型，环境由对方自建
    python deployment/package.py --out D:/dist      # 指定输出目录

## 为什么不能"直接压缩整个目录发给别人"

有三个硬问题，本脚本逐个处理：

1. **conda 环境不能直接复制**
   ``deployment/envs/detector/Lib/site-packages/__editable___mmrotate_0_3_4_finder.py``
   里写死了 ``D:\\Code\\qgis插件\\vecrot\\models\\...``。换台机器、甚至换个目录都失效。
   → 用 ``conda-pack`` 打包，它会重写路径；解包脚本再跑一次 ``conda-unpack``。

2. **manifest 里的路径是本机的绝对路径**
   → 生成一份"模板 manifest"，路径写成相对分发包根的占位；
     对方解包后由 ``install_plugin.py`` 转成他自己的绝对路径。

3. **QGIS 侧无法感知仓库位置**
   → 安装脚本已经把 manifest 复制到固定位置并写 QGIS 设置；
     分发版沿用同一机制，不额外增加步骤。

## 生成的产物

```
<out>/vecrot-dist-<date>/
├── README-分发说明.md          ← 对方先读这个
├── install.py                  ← 对方唯一的入口
├── plugin/                     ← 薄插件（QGIS 侧，几十 KB）
├── backend/                    ← 协调器与 worker
├── vecrot/lib/                 ← VectorLLM 推理封装
├── vecrot/models/              ← 权重
├── deployment/
│   ├── manifest.template.json  ← 路径占位版
│   └── envs/*.tar.gz           ← conda-pack 产物（可选）
└── third_party/Strip-R-CNN/    ← 检测源码（不含 .git）
```
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

#: 打进包里的内容（相对仓库根）。顺序即打包顺序。
CODE_PARTS = [
    "backend",
    "qgis_plugin/vecrot_qgis",
    "vecrot/lib",
    "vecrot/models",
    "tests",
]
#: 自检与诊断脚本：教程里让接收方直接跑这些命令，必须打进包
SELFCHECK_FILES = [
    "deployment/selfcheck.py",
    "deployment/verify_detector_runtime.py",
    "deployment/verify_vector_runtime.py",
    "deployment/verify_detector_config.py",
    "deployment/install_plugin.py",
    "deployment/check_plugin.py",
    "deployment/run_in_qgis.py",
    "deployment/make_icon.py",
    "deployment/rebuild_pruned_base.py",
    "deployment/demo.py",
]
#: 检测源码（体积大，但配置与模型实现必需）
SRC_PARTS = [
    "third_party/Strip-R-CNN",
]
#: 明确排除
#: 打包时排除的目录名。注意 "envs" 只在源码树的 deployment/ 下排除，
#: 分发包里的 deployment/envs/*.tar.gz 是环境归档，必须保留（见 _copy_tree 调用处）。
EXCLUDE_DIRS = {"__pycache__", ".git", ".cache", "outputs", "runs",
                # pip install . 的构建残留，对方不需要
                "build", "dist", ".eggs"}
EXCLUDE_SUFFIX = {".pyc", ".pyo"}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _copy_tree(src: str, dst: str, skip_files: set[str] | None = None,
               skip_dirs: set[str] | None = None) -> int:
    """复制目录，跳过缓存与版本控制。返回复制的文件数。

    ``skip_files`` / ``skip_dirs`` 是**绝对路径**集合（已 normcase），
    用于 --slim 时排除整个模型目录或个别权重文件。

    注意：``skip_dirs`` 必须在遍历时**剪枝**，而不是只在入口判断 ——
    否则 ``vecrot/models/base_v1`` 这种「在大目录深处」的目标会整目录被复制。
    """
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs
                   if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")]
        if skip_dirs:
            # 剪枝：跳过命中的子目录（含其全部内容）
            dirs[:] = [d for d in dirs
                       if os.path.normcase(os.path.join(root, d)) not in skip_dirs]
        rel = os.path.relpath(root, src)
        target_root = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(target_root, exist_ok=True)
        for f in files:
            if os.path.splitext(f)[1] in EXCLUDE_SUFFIX:
                continue
            full = os.path.join(root, f)
            if skip_files and os.path.normcase(full) in skip_files:
                continue
            shutil.copy2(full, os.path.join(target_root, f))
            n += 1
    return n


def _write_manifest_template(path: str) -> None:
    """把 manifest 的相对路径改成占位，供对方安装时解析到自己的目录。"""
    with open(os.path.join(HERE, "manifest.json"), encoding="utf-8") as f:
        m = json.load(f)

    # 去掉本机绝对路径痕迹，改为相对分发包根
    def rel(p):
        if not p:
            return p
        p = str(p).replace("\\", "/")
        repo = REPO.replace("\\", "/")
        if p.startswith(repo + "/"):
            return p[len(repo) + 1:]
        return p

    models = m.get("models") or {}
    for key in ("detector", "vectorizer"):
        block = models.get(key) or {}
        for fld in ("repo", "weights", "config", "vecrot_dir", "ckpt", "base",
                    "detections_jsonl"):
            if fld in block and block[fld]:
                block[fld] = rel(block[fld])
        models[key] = block
    if models.get("detections_jsonl"):
        models["detections_jsonl"] = rel(models["detections_jsonl"])
    m["models"] = models

    # 解释器本机路径必须由对方安装时重写，这里留空占位
    for role, cfg in (m.get("environments") or {}).items():
        cfg["python"] = ""          # install.py 会按解包位置填入
        cfg["_note"] = "由 install.py 自动填入本机解包后的 python.exe 绝对路径"
    m["repo_root"] = ".."           # manifest 在 deployment/ 下，上一级即分发包根
    m["_packaged"] = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_repo": REPO,
        "note": "路径已改为相对分发包根；install.py 会转成本机绝对路径。",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)


def _pack_env(env_dir: str, out_tar: str) -> bool:
    """用 conda-pack 打包一个 conda 环境（会重写内部硬编码路径）。"""
    try:
        import conda_pack
    except ImportError:
        _log("  ⚠ 未安装 conda-pack，跳过环境打包")
        _log("    安装：conda install -c conda-forge conda-pack")
        return False
    _log(f"  打包 {os.path.basename(env_dir)} …（数分钟，体积数 GB）")
    try:
        conda_pack.pack(prefix=env_dir, output=out_tar, force=True,
                        ignore_editable_packages=False)
    except Exception as exc:                    # noqa: BLE001
        _log(f"  ⚠ 失败：{type(exc).__name__}: {exc}")
        return False
    _log(f"  → {out_tar}  {os.path.getsize(out_tar)/1e9:.2f} GB")
    return True


with open(os.path.join(HERE, "install_template.py"), encoding="utf-8") as _f:
    INSTALL_PY = _f.read()




#: --slim 时跳过的路径前缀（相对仓库根）。
#: 依据：
#: * ``vecrot/models/base_v1`` —— 推理实际用的是 vecrot/models/official_vocab_635 基座；
#:   base_v1 只作「重建原料」用，对方不需要（两个基座张量数都是 609，
#:   仅 embedding 行数不同，official_vocab_635 自包含）。
#: * ``vecrot/models/detector_strip_s/best_mAP_iter_90288.pth`` —— 不带场景分类的旧权重，
#:   与带场景版本功能重叠，保留场景版即可。
SLIM_SKIP = (
    "vecrot/models/base_v1",
    "vecrot/models/detector_strip_s/best_mAP_iter_90288.pth",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "dist"))
    ap.add_argument("--no-envs", action="store_true",
                    help="不打 conda 环境（对方自建）")
    ap.add_argument("--slim", action="store_true",
                    help="精简：去掉 base_v1 与旧检测权重（约省 2.4 GB）")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()

    name = a.name or ("vecrot-dist-" + time.strftime("%Y%m%d"))
    dest = os.path.join(os.path.abspath(a.out), name)
    if os.path.isdir(dest):
        _log(f"目标已存在，先删除：{dest}")
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)
    _log(f"打包到 {dest}\n")

    # ---- 1. 代码与模型
    _log("== 复制代码与模型 ==")
    total = 0
    skip_files: set[str] = set()
    skip_dirs: set[str] = set()
    if a.slim:
        for rel in SLIM_SKIP:
            full = os.path.normcase(os.path.join(REPO, rel))
            if os.path.isdir(full):
                skip_dirs.add(full)
            else:
                skip_files.add(full)
        _log("  【精简模式】跳过：")
        for rel in SLIM_SKIP:
            _log(f"    - {rel}")
        _log("")

    for part in CODE_PARTS:
        src = os.path.join(REPO, part)
        if not os.path.isdir(src):
            _log(f"  跳过（不存在）{part}")
            continue
        # qgis_plugin/vecrot_qgis -> plugin/
        rel = "plugin" if part.startswith("qgis_plugin/") else part
        if skip_dirs and os.path.normcase(src) in skip_dirs:
            _log(f"  {part:45} → 跳过（--slim）")
            continue
        n = _copy_tree(src, os.path.join(dest, rel), skip_files, skip_dirs)
        _log(f"  {part:45} → {rel:20} {n} 个文件")
        total += n
    _log(f"  小计 {total} 个文件")

    # ---- 2. 检测源码（去 .git）
    _log("\n== 复制检测源码（Strip-R-CNN）==")
    for part in SRC_PARTS:
        src = os.path.join(REPO, part)
        if os.path.isdir(src):
            n = _copy_tree(src, os.path.join(dest, part), skip_files, skip_dirs)
            _log(f"  {part}  {n} 个文件")
        else:
            _log(f"  跳过（不存在）{part}")

    # ---- 3. manifest 模板
    _log("\n== 生成 manifest 模板 ==")
    os.makedirs(os.path.join(dest, "deployment"), exist_ok=True)
    tpl = os.path.join(dest, "deployment", "manifest.template.json")
    _write_manifest_template(tpl)
    _log(f"  {tpl}")

    # ---- 4. 环境
    if a.no_envs:
        _log("\n== 跳过环境打包（--no-envs）==")
    else:
        _log("\n== 打包 conda 环境 ==")
        os.makedirs(os.path.join(dest, "deployment", "envs"), exist_ok=True)
        for role in ("detector", "vector"):
            src_env = os.path.join(REPO, "deployment", "envs", role)
            if not os.path.isdir(src_env):
                _log(f"  跳过 {role}（不存在）")
                continue
            out_tar = os.path.join(dest, "deployment", "envs", f"{role}.tar.gz")
            _pack_env(src_env, out_tar)

    # ---- 4.5 自检与诊断脚本（教程里会直接调这些命令）
    _log("\n== 复制自检与诊断脚本 ==")
    dep_dst = os.path.join(dest, "deployment")
    os.makedirs(dep_dst, exist_ok=True)
    n_scripts = 0
    for rel in SELFCHECK_FILES:
        src = os.path.join(REPO, rel)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dep_dst, os.path.basename(rel)))
            n_scripts += 1
        else:
            _log(f"  ⚠ 缺：{rel}")
    _log(f"  {n_scripts} 个脚本 → deployment/")

    # ---- 5. 安装脚本与说明
    _log("\n== 写入安装脚本与说明 ==")
    with open(os.path.join(dest, "install.py"), "w", encoding="utf-8") as f:
        f.write(INSTALL_PY)
    _log("  install.py")
    readme = os.path.join(REPO, "docs", "分发说明.md")
    if os.path.isfile(readme):
        shutil.copy2(readme, os.path.join(dest, "README-分发说明.md"))
        _log("  README-分发说明.md")
    # 文档：自建环境教程与完整包教程都带上（对方按自己拿到的包选读）
    for name, dst_name in (
        ("安装教程-自建环境.md", "安装教程-自建环境.md"),
        ("安装教程.md", "安装教程-完整包.md"),
        ("STATUS.md", "技术状态说明.md"),
    ):
        src = os.path.join(REPO, "docs", name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dest, dst_name))
            _log(f"  {dst_name}")

    # ---- 6. 汇总
    size = 0
    for root, _d, files in os.walk(dest):
        for fn in files:
            try:
                size += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    _log(f"\n完成：{dest}  合计 {size/1e9:.2f} GB")
    _log("对方拿到后运行： python install.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
