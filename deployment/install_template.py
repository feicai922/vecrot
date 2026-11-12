"""分发包安装脚本（对方运行这个）。

    python install.py

做四件事：
1. 解包 conda 环境（若包内带了 envs/*.tar.gz）并跑 conda-unpack 修正路径
2. 把 manifest 模板里的相对路径转成本机绝对路径，填好各环境 python.exe
3. 把插件复制到 QGIS profile，并把 manifest 副本与设置写到固定位置
4. 自检：环境、模型、插件三层

可选参数：
    --qgis-profile <name>   指定 QGIS profile（默认 default）
    --skip-envs             跳过环境解包（已自行建好环境时用）
    --check-only            只做自检，不改任何文件
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.join(ROOT, "deployment", "envs")
TPL = os.path.join(ROOT, "deployment", "manifest.template.json")
OUT_MANIFEST = os.path.join(ROOT, "deployment", "manifest.json")


def log(m):
    print(m, flush=True)


def unpack_envs() -> None:
    tars = []
    if os.path.isdir(ENV_DIR):
        tars = sorted(f for f in os.listdir(ENV_DIR) if f.endswith(".tar.gz"))
    if not tars:
        log("[1/4] 包内没有环境归档，跳过（需要你自建，见 README-分发说明.md）")
        return
    for t in tars:
        name = t[:-len(".tar.gz")]
        dest = os.path.join(ENV_DIR, name)
        if os.path.isdir(dest):
            log(f"[1/4] {name} 已存在，跳过解包")
            continue
        log(f"[1/4] 解包 {t} → {dest} …")
        os.makedirs(dest, exist_ok=True)
        subprocess.run(["tar", "-xzf", os.path.join(ENV_DIR, t), "-C", dest],
                       check=True)
        # conda-pack 的收尾：重写内部绝对路径与脚本 shebang
        for candidate in (os.path.join(dest, "Scripts", "conda-unpack.exe"),
                          os.path.join(dest, "bin", "conda-unpack")):
            if os.path.isfile(candidate):
                log(f"       conda-unpack …")
                subprocess.run([candidate], check=True)
                break
        else:
            log("       ⚠ 未找到 conda-unpack，环境可能仍指向原路径")


def build_manifest(skip_envs: bool) -> dict:
    log("[2/4] 生成部署 manifest")
    with open(TPL, encoding="utf-8") as f:
        m = json.load(f)

    def abspath(p):
        if not p:
            return p
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(ROOT, p))

    models = m.get("models") or {}
    for key in ("detector", "vectorizer"):
        block = models.get(key) or {}
        for fld in ("repo", "weights", "config", "vecrot_dir", "ckpt", "base",
                    "detections_jsonl"):
            if block.get(fld):
                block[fld] = abspath(block[fld])
        models[key] = block
    if models.get("detections_jsonl"):
        models["detections_jsonl"] = abspath(models["detections_jsonl"])
    m["models"] = models

    # 每个环境的 python.exe
    # 环境探测顺序（逐个尝试，第一个命中的胜出）：
    #   1) 环境变量显式指定（最可靠，尤其 coordinator）
    #   2) 包内 deployment/envs/<role>/python.exe（完整包的形态）
    #   3) 包内 deployment/envs/<role>_env/… 之类变体
    #   4) **本机 conda 环境**（按 docs/安装教程-自建环境.md 建的，名字是
    #      vecrot-geo / vecrot-det / vecrot-vllm）
    #
    # 之前只查 1)2)，导致用户按教程自建好环境后 install.py 仍报"缺失"。
    CONDA_NAMES = {
        "coordinator": ("vecrot-geo", "vecrot_coord"),
        "detector": ("vecrot-det", "vecrot_det"),
        "vector": ("vecrot-vllm", "vecrot_vllm", "vecrot-vector"),
        "detector_hf": ("vecrot-vllm", "vecrot_vllm"),
    }
    ENV_VARS = {
        "coordinator": "VECROT_PY_COORDINATOR",
        "detector": "VECROT_PY_DETECTOR",
        "vector": "VECROT_PY_VECTOR",
        "detector_hf": "VECROT_PY_VECTOR",
    }

    def _candidate_pythons(role: str) -> list[str]:
        out: list[str] = []
        envvar = ENV_VARS.get(role)
        if envvar and os.environ.get(envvar):
            out.append(os.environ[envvar])
        # 包内环境
        for sub in (role, role.replace("_", "-")):
            out.append(os.path.join(ENV_DIR, sub, "python.exe"))
        # conda 环境：从 conda 可执行文件所在目录推断，再退到常见安装位置
        roots = []
        conda_exe = os.environ.get("CONDA_EXE", "")
        if conda_exe:
            roots.append(os.path.dirname(os.path.dirname(conda_exe)))
        roots += [
            os.path.expanduser("~/miniconda3"),
            os.path.expanduser("~/anaconda3"),
            r"C:\ProgramData\miniconda3",
            r"C:\ProgramData\anaconda3",
            r"D:\Program Files\miniconda",
            r"C:\Program Files\miniconda3",
            r"D:\miniconda3",
            r"C:\miniconda3",
        ]
        for root in roots:
            for name in CONDA_NAMES.get(role, ()):
                out.append(os.path.join(root, "envs", name, "python.exe"))
        return out

    mapping = {role: next((p for p in _candidate_pythons(role)
                           if p and os.path.isfile(p)), "")
               for role in ("coordinator", "detector", "vector", "detector_hf")}
    for role, cfg in (m.get("environments") or {}).items():
        cfg.pop("_note", None)
        p = mapping.get(role, "")
        if p and os.path.isfile(p):
            cfg["python"] = p
        else:
            cfg["python"] = ""
        if role == "detector_hf":
            # RT-DETR 复用 vector 环境
            vp = os.path.join(ENV_DIR, "vector", "python.exe")
            if os.path.isfile(vp):
                cfg["python"] = vp
        cfg["_note"] = ("自动填入；若为空请手工指定 python.exe 绝对路径"
                        if not cfg["python"] else "自动检测")

    coord = (m["environments"].get("coordinator") or {}).get("python", "")
    if not coord:
        log("       ⚠ coordinator 环境未自动找到。")
        log("         请先按 安装教程-自建环境.md §3 建 vecrot-geo 环境，然后：")
        log("           setx VECROT_PY_COORDINATOR <vecrot-geo 环境的 python.exe 绝对路径>")
        log("         （setx 后要新开一个命令行窗口才生效）")
    # manifest 放在 <包根>/deployment/ 下，所以仓库根是上一级；
    # 写绝对路径，因为这份 manifest 会被复制到 ~/vecrot/deployment/（脱离包根）
    m["repo_root"] = ROOT      # 解包后的分发包根（install.py 所在目录）
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    log(f"       → {OUT_MANIFEST}")
    return m


def install_plugin(qgis_profile: str) -> None:
    log("[3/4] 安装 QGIS 插件")
    appdata = os.environ.get("APPDATA") or os.path.expanduser("~")
    dest = os.path.join(appdata, "QGIS", "QGIS3", "profiles", qgis_profile,
                        "python", "plugins", "vecrot_qgis")
    src = os.path.join(ROOT, "plugin")
    if not os.path.isdir(src):
        log(f"       ⚠ 包内没有 plugin/ 目录：{src}")
        return
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    log(f"       → {dest}")

    # manifest 副本放到固定位置（插件装在 profile 里时靠它定位仓库）
    home = os.path.expanduser("~")
    mdir = os.path.join(home, "vecrot", "deployment")
    os.makedirs(mdir, exist_ok=True)
    shutil.copy2(OUT_MANIFEST, os.path.join(mdir, "manifest.json"))
    log(f"       → {os.path.join(mdir, 'manifest.json')}")

    # 写 QGIS 设置（两个作用域都要写，原因见 docs/STATUS.md）
    runner = os.path.join(ROOT, "deployment", "run_in_qgis.py")
    if os.path.isfile(runner):
        tmp = os.path.join(ROOT, "_set_setting.py")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(SETTING_SNIPPET)
        target = os.path.join(mdir, "manifest.json")
        subprocess.run([sys.executable, runner, tmp, target],
                       cwd=ROOT, check=False)
        try:
            os.remove(tmp)
        except OSError:
            pass
    else:
        log("       ⚠ 未找到 run_in_qgis.py，跳过写 QGIS 设置")


SETTING_SNIPPET = '''import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from qgis.core import QgsApplication, QgsSettings
path = sys.argv[1]
app = QgsApplication([], False); app.initQgis()
try:
    for args in ((), ("QGIS", "QGIS3")):
        s = QgsSettings(*args); s.setValue("vecrot/manifest_path", path); s.sync()
        print("OK", args or "default", s.value("vecrot/manifest_path", "", type=str))
    for args in ((), ("QGIS", "QGIS3")):
        s = QgsSettings(*args)
        v = s.value("mSettings/enabledPlugins", []) or []
        if isinstance(v, str): v = [v]
        if "vecrot_qgis" not in v:
            v.append("vecrot_qgis"); s.setValue("mSettings/enabledPlugins", v); s.sync()
        print("OK enabledPlugins", s.value("mSettings/enabledPlugins", []))
finally:
    import os as _o; _o._exit(0)
'''


def _preset_checks(root: str) -> list:
    """从包内插件镜像取出各预设引用的文件，逐项检查是否存在。

    直接 import ``plugin/jobs/presets.py``（它是纯 Python，不依赖 QGIS），
    拿 ``_REPO`` 指向本包根后重建预设，读真实的绝对路径。比在安装脚本里
    用正则猜路径可靠——预设里的路径是跨行拼接的，文本扫描会切错。
    """
    pdir = os.path.join(root, "plugin")
    src = os.path.join(pdir, "jobs", "presets.py")
    if not os.path.isfile(src):
        return []
    checks = []
    saved = sys.path[:]
    saved_env = os.environ.get("VECROT_REPO_ROOT")
    try:
        os.environ["VECROT_REPO_ROOT"] = root
        sys.path.insert(0, root)
        pkg = os.path.basename(pdir) or "vecrot_qgis"
        spec = importlib.util.spec_from_file_location(
            pkg, os.path.join(pdir, "__init__.py"),
            submodule_search_locations=[pdir])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[pkg] = mod
        spec.loader.exec_module(mod)
        presets = importlib.import_module(pkg + ".jobs.presets")
        refresh = getattr(presets, "refresh_repo_root", None)
        if refresh:
            refresh()
        for pr in getattr(presets, "PRESETS", ()):
            key = getattr(pr, "key", "?")
            for fld, path in (getattr(pr, "files", None) or {}).items():
                if isinstance(path, str) and path:
                    checks.append((f"{key}.{fld}", path))
    except Exception:                            # noqa: BLE001
        return []
    finally:
        sys.path[:] = saved
        if saved_env is None:
            os.environ.pop("VECROT_REPO_ROOT", None)
        else:
            os.environ["VECROT_REPO_ROOT"] = saved_env
    return checks


def selfcheck() -> int:
    log("[4/4] 自检")
    problems = 0
    try:
        with open(OUT_MANIFEST, encoding="utf-8") as f:
            m = json.load(f)
    except Exception as exc:                    # noqa: BLE001
        log(f"       ✗ manifest 读不出：{exc}")
        return 1

    for role, cfg in (m.get("environments") or {}).items():
        p = cfg.get("python")
        if p and os.path.isfile(p):
            log(f"       ✓ {role:12} {p}")
            continue
        if role == "detector_hf":
            # 与 vector 共用环境；vector 没解包时这里也必然缺
            vec = (m["environments"].get("vector") or {}).get("python", "")
            if vec and os.path.isfile(vec):
                log(f"       ✗ {role:12} 应指向 vector 环境，但 manifest 里为空")
                problems += 1
            else:
                log(f"       - {role:12} 未配置（需先解包 vector 环境；只影响 RT-DETR）")
            continue
        log(f"       ✗ {role:12} 缺失：{p or '（空）'}")
        problems += 1

    models = m.get("models") or {}
    det = models.get("detector") or {}
    vec = models.get("vectorizer") or {}

    # 权重检查以**预设实际引用的文件**为准。
    # --slim 会剔掉旧权重 best_mAP_iter_90288.pth（场景版已覆盖其功能），
    # 此时 manifest 里的 models.detector.weights 是个不再存在的历史字段，
    # 拿它做自检只会制造假失败。
    checks = _preset_checks(ROOT)
    if not checks:
        checks = [("检测权重", det.get("weights")),
                  ("检测配置", det.get("config"))]
    checks += [("轮廓适配器", vec.get("ckpt")), ("轮廓基座", vec.get("base"))]

    for label, path in checks:
        ok = bool(path) and os.path.exists(path)
        log(("       ✓ " if ok else "       ✗ ") + f"{label}: {path}")
        if not ok:
            problems += 1

    if problems:
        log(f"\
{problems} 项未就绪。修好后重跑本脚本，或按 README-分发说明.md 手工配置。")
    else:
        log("\
全部就绪。重启 QGIS，菜单 Vecrot → Vecrot 建筑提取。")
    return 0 if problems == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qgis-profile", default="default")
    ap.add_argument("--skip-envs", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()

    if a.check_only:
        # 只检查：若还没生成 manifest，先按模板生成一份（不装插件、不解包环境）
        if not os.path.isfile(OUT_MANIFEST):
            log("尚无 manifest.json，按模板生成一份用于检查…")
            build_manifest(skip_envs=True)
        return selfcheck()
    if not a.skip_envs:
        unpack_envs()
    build_manifest(a.skip_envs)
    install_plugin(a.qgis_profile)
    return selfcheck()


if __name__ == "__main__":
    sys.exit(main())
