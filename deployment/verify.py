"""一键跑完 QGIS 侧全部验证。

    python deployment/verify.py            # 静态检查 + 全量测试 + QGIS 运行时三件套
    python deployment/verify.py --fast     # 跳过 QGIS 运行时（无 QGIS 的机器用）

QGIS 运行时验证需要本机装有 QGIS；找不到时明确报告 skipped，不算失败。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def run(title: str, argv: list[str], required: bool = True) -> bool:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    proc = subprocess.run(argv, cwd=REPO)
    ok = proc.returncode == 0
    if not ok and not required:
        print(f"(非必需，跳过：{title})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="跳过 QGIS 运行时验证")
    args = ap.parse_args()

    py = sys.executable
    failures: list[str] = []

    if not run("插件静态检查", [py, os.path.join(HERE, "check_plugin.py")]):
        failures.append("check_plugin")
    if not run("后端单元/集成测试", [py, "-m", "pytest", "tests/", "-q"]):
        failures.append("pytest")
    if not run("几何自检", [py, os.path.join(HERE, "selfcheck.py"), "--layer", "geometry"]):
        failures.append("selfcheck-geometry")
    if not run("GeoPackage 自检",
               [py, os.path.join(HERE, "selfcheck.py"), "--layer", "geopackage"]):
        failures.append("selfcheck-gpkg")
    if not run("检测配置契约校验（不需 torch）",
               [py, os.path.join(HERE, "verify_detector_config.py")]):
        failures.append("verify_detector_config")
    rebuilt = os.path.join(REPO, "vecrot", "models", "official_vocab_635",
                           "REBUILD_INFO.json")
    if os.path.isfile(rebuilt):
        import json as _json

        info = _json.load(open(rebuilt, encoding="utf-8"))
        okv = info.get("verification", {})
        passed = all(okv.get(k) for k in ("rows_match", "pixel_idx_match", "pixel_row_equals_base"))
        print("\n" + "=" * 60 + "\n重建基座校验\n" + "=" * 60)
        print(("  [OK]   " if passed else "  [FAIL] ")
              + f"official_vocab_635: vocab={info.get('vocab_size')} "
                f"pixel_idx={info.get('pixel_idx')} {okv}")
        if not passed:
            failures.append("rebuilt_base")
    else:
        print("\n（未发现重建基座 vecrot/models/official_vocab_635，跳过）")

    if not args.fast:
        runner = [py, os.path.join(HERE, "run_in_qgis.py")]
        qgis_scripts = [
            ("QGIS 运行时：插件加载", "qgis_runtime_check.py"),
            ("QGIS 运行时：插件管理器挂载", "qgis_plugin_mount_check.py"),
            ("QGIS 运行时：AOI 框选工具（真实拖拽）", "qgis_aoi_tool_check.py"),
            ("QGIS 运行时：端到端作业（jsonl/null）", "qgis_job_check.py"),
            ("QGIS 运行时：GUI 流程（真模型）", "qgis_gui_flow_check.py"),
            ("QGIS 运行时：profile 安装（真模型）", "qgis_real_job_check.py"),
            ("QGIS 运行时：项目图层选择 + 增量提取", "qgis_incremental_check.py"),
            ("QGIS 运行时：滚动面板 + 模型自选", "qgis_model_select_check.py"),
            ("QGIS 运行时：AOI 持久显示 + 预设搭配 + 样式", "qgis_ui_round2_check.py"),
        ]
        # 先探一下有没有 QGIS
        probe = subprocess.run([py, os.path.join(HERE, "run_in_qgis.py"), "--probe-only"],
                               cwd=REPO, capture_output=True, text=True)
        if probe.returncode != 0:
            print("\n未找到 QGIS，跳过运行时验证（not_run）")
        else:
            for title, script in qgis_scripts:
                if not run(title, runner + [os.path.join(HERE, script)]):
                    failures.append(script)

    print(f"\n{'=' * 60}")
    if failures:
        print(f"未通过：{', '.join(failures)}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
