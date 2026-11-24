#!/usr/bin/env python
"""模型文件夹注册表的静态校验（不需要 torch / QGIS / 权重）。

    python deployment/verify_model_registry.py

守的是「一个文件夹 = 一个模型」这套约定本身，而不是某个权重能不能加载：

1. 两份 ``model_registry.py``（backend 权威 + 插件镜像）**逐字节相同**；
2. 同一个 models 根，两侧扫出来的条目完全一致；
3. 真机 ``vecrot/models/`` 能被扫出来，缺什么文件说得清；
4. 每个声明的 backend 都有对应的后端实现与 worker 环境角色
   —— 少了这个，"加个文件夹"会变成"面板能选、一跑就 Unknown profile"；
5. ``repo:`` / ``src:`` / ``models:`` / ``.`` 五种路径写法解析正确。
"""
from __future__ import annotations

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

results: list[tuple[bool, str]] = []


def check(ok: bool, msg: str) -> None:
    results.append((bool(ok), msg))
    print(("  [OK]   " if ok else "  [FAIL] ") + msg, flush=True)


def make_folder(root: str, name: str, spec: str, files: dict) -> str:
    folder = os.path.join(root, name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "model.json"), "w", encoding="utf-8") as f:
        f.write(spec)
    for rel, size in files.items():
        p = os.path.join(folder, *rel.split("/"))
        os.makedirs(os.path.dirname(p) or folder, exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"\0" * size)
    return folder


def main() -> int:
    from backend.contracts import model_registry as mr

    print(f"仓库根：{REPO}\n")

    print("=== 1. 两份实现一致 ===")
    a = os.path.join(REPO, "backend", "contracts", "model_registry.py")
    b = os.path.join(REPO, "vecrot_qgis", "jobs", "model_registry.py")
    check(os.path.isfile(b), "插件侧存在 jobs/model_registry.py")
    same = os.path.isfile(b) and open(a, "rb").read() == open(b, "rb").read()
    check(same, "backend 与插件镜像逐字节相同"
                + ("" if same else "（cp backend/contracts/model_registry.py "
                                   "vecrot_qgis/jobs/model_registry.py）"))

    print("\n=== 2. 两侧扫描结果一致 ===")
    from backend.contracts import presets as bp
    from vecrot_qgis.jobs import presets as pp

    with tempfile.TemporaryDirectory() as tmp:
        make_folder(tmp, "strip_a", """{
          "spec_version": 1, "key": "strip_a", "label": "A",
          "backend": "strip_s_gzda", "num_classes": 1,
          "repo": "repo:third_party/Strip-R-CNN",
          "weights": "w.pth", "config": "src:configs/gzda/x.py",
          "detector_options": {"score_thr": 0.05}
        }""", {"w.pth": 16})
        make_folder(tmp, "rio_b", """{
          "spec_version": 1, "key": "rio_b", "label": "B",
          "backend": "rio_detr", "num_classes": 1,
          "repo": "repo:third_party/RiO-DETRv2",
          "weights": "last.pth", "config": "c.yml",
          "detector_options": {"score_thr": 0.4}
        }""", {"last.pth": 16, "c.yml": 16})
        os.makedirs(os.path.join(tmp, "no_spec"))

        saved_b, saved_p = bp.models_root(), pp.models_root()
        try:
            bp.refresh_models_root(tmp)
            pp.refresh_models_root(tmp)
            left = {p.key: (p.backend, p.num_classes, p.files, p.detector_options)
                    for p in bp.PRESETS}
            right = {p.key: (p.backend, p.num_classes, p.files, p.detector_options)
                     for p in pp.PRESETS}
            check(left == right, f"两侧条目一致：{sorted(left)}")
            check(bp.skipped() and pp.skipped(), "没 model.json 的文件夹被记为 skipped")
        finally:
            bp.refresh_models_root(saved_b)
            pp.refresh_models_root(saved_p)

    print("\n=== 3. 路径写法 ===")
    with tempfile.TemporaryDirectory() as tmp:
        folder = os.path.join(tmp, "f")
        os.makedirs(folder)
        cases = {
            "相对文件夹": ("w.pth", os.path.join(folder, "w.pth")),
            "本文件夹": (".", folder),
            "仓库根": ("repo:third_party/Strip-R-CNN", os.path.join(
                REPO, "third_party", "Strip-R-CNN")),
            "models 根": ("models:other/w.pth", os.path.join(tmp, "other", "w.pth")),
        }
        for name, (decl, want) in cases.items():
            got = mr.resolve_path(decl, folder, REPO, tmp)
            check(os.path.normpath(got) == os.path.normpath(want),
                  f"{name}: {decl} → {got}")
        try:
            mr.resolve_path("src:x.py", folder, REPO, tmp)
            check(False, "src: 未声明 repo 时应报错")
        except mr.ModelSpecError:
            check(True, "src: 未声明 repo 时明确报错，不退化成相对文件夹")

    print("\n=== 4. 后端与环境角色覆盖 ===")
    from backend.coordinator.runner import Coordinator
    from backend.contracts.jobspec import parse_job_spec
    from backend.detector.backends import DetectorNotAvailable, build_detector
    import inspect

    source = inspect.getsource(build_detector)
    for backend in mr.BACKEND_REQUIRED_FILES:
        if backend == "jsonl":
            continue
        marker = f'"{backend}"' if backend != "null" else "'null'"
        check(marker in source, f"build_detector 认识 backend={backend}")

    spec = parse_job_spec({
        "schema_version": "1.0",
        "job_id": "registry_check",
        "input": {"raster_path": os.path.join(REPO, "x.tif"),
                  "bands_rgb": [1, 2, 3],
                  "aoi_geojson": {"type": "Polygon",
                                  "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}},
        "inference": {},
        "output": {"directory": REPO},
    })
    manifest = {"environments": {
        "detector": {"python": "/usr/bin/python3"},
        "detector_hf": {"python": "/usr/bin/python3"},
        "detector_rio": {"python": "/usr/bin/python3"},
    }}
    coord = Coordinator(spec, manifest, REPO)
    for profile, want in (("strip_s_gzda", "detector"),
                          ("mmrotate_obb", "detector"),
                          ("mmdet_hbb", "detector"),
                          ("rtdetr_hf", "detector_hf"),
                          ("rio_detr", "detector_rio")):
        got = coord._env_role_for("detector", {"profile": profile})
        check(got == want, f"{profile} → 环境 {got}（期望 {want}）")

    print("\n=== 5. 本机模型文件夹 ===")
    root = mr.default_models_root(REPO)
    print(f"      models 根：{root}")
    if not os.path.isdir(root):
        print("      （不存在；先按 docs/模型下载.md 放好 models 包）")
    presets, skipped = mr.scan(root, REPO)
    for p in presets:
        miss = p.missing()
        check(not miss, f"{p.key} [{p.backend}] " + ("齐备" if not miss else f"缺 {miss}"))
    for item in skipped:
        print(f"  [WARN] {os.path.basename(item['folder'])}: {item['reason']}")

    print()
    failed = [m for ok, m in results if not ok]
    if failed:
        print(f"{len(failed)} 项未通过：")
        for m in failed:
            print(f"  · {m}")
        return 1
    print(f"全部 {len(results)} 项通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
