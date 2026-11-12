# Vecrot vendored detection runtime

This directory contains the local Strip-R-CNN / MMRotate 0.3.4 source snapshot used by Vecrot, based on <https://github.com/YXB-NKU/Strip-R-CNN>. The exact upstream commit of the provided local snapshot is unknown; no claim of an unmodified upstream checkout is made.

The upstream LICENSE, CITATION and source notices are retained. This directory follows its own LICENSE (CC BY-NC 4.0 as supplied), not the parent project's MIT license.

Vecrot-specific additions include the `configs/gzda` dataset/hooks, reconstructed `scene_branch.py` and `scene_hooks.py`, and the copied `strip_s_gzda_scene.py` configuration. The reconstruction is described in those module docstrings. The inference backend applies compatibility patches separately; network weights are not included here.

Only the runtime source subset and packaging inputs are distributed: `mmrotate`, `configs`, `requirements`, `tools`, setup/manifest files and attribution. Demos, notebooks, documentation assets, `.mim` duplicates, build output and installed-package metadata are omitted. `setup.py` recreates the required `.mim` configuration/tools during a regular package build. Existing source files were copied without altering their network implementation. SHA256 values of the supplied files are recorded in `SOURCE_MANIFEST.json`.

From the Vecrot repository root, in the external detector environment:

```powershell
python -m pip install .\third_party\Strip-R-CNN --no-deps
```

Weights are downloaded separately into `vecrot/models/`. Older copies of this source inside the models bundle are no longer used.
