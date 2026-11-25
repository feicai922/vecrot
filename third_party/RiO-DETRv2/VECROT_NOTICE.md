# Vecrot vendored RiO-DETR runtime

This directory is a source snapshot of **RiO-DETRv2** used by Vecrot's `rio_detr`
detection backend: <https://github.com/RicePasteM/RiO-DETRv2> (upstream commit
`ca8ee6e`, "Update inference script and tighten README wording").

RiO-DETRv2 is licensed **AGPL-3.0**; the upstream `LICENSE` is retained verbatim in
this directory and that license — not the parent project's MIT license — governs
these files. The source is distributed here together with the Vecrot repository,
which is itself public, so the AGPL source-availability obligation is met by
shipping this directory as-is.

## What was kept

Only what is needed to **build and run inference**:

```text
engine/         model / backbone / encoder / decoder / postprocessor / solver
configs/        upstream configs + Vecrot-local gzda training configs
tools/          upstream tools (inference / deployment / evaluation / benchmark)
train.py        training entry point (kept for reproducibility)
requirements.txt, LICENSE, README.md
SOURCE_MANIFEST.json     SHA256 of every file above
```

## What was dropped

* `.git/`, `assets/` (README images and demo videos, ~12 MB)
* `outputs/`, `training_logs/`, `pretrain/` (training artifacts, not source)
* `__pycache__/`, `*.pyc`
* every `*.pth` / `*.pt` / `*.safetensors` — network weights live in the models
  package, see `docs/模型下载.md`

## Vecrot-local additions (not upstream files)

| Path | What |
|---|---|
| `configs/rio-gzda/riov2_hgnetv2_x_gzda_12e.yml` | fine-tune config for the gzda building dataset (1 class, 800×800) |
| `configs/dataset/gzda_v3_offline.yml` | dataset + `GzdaObbEvaluator` wiring for that config |
| `engine/data/dataset/gzda_obb_eval.py` | `GzdaObbEvaluator`: upstream has no `GzdaObbEvaluator`, and its solver expects a `CocoEvaluator`-shaped object (`cleanup/update/accumulate/summarize/iou_types`) |

Nothing else was modified: the network implementation is upstream's.

## How Vecrot loads it

`backend/detector/rio_detr_backend.py` puts this directory on `sys.path` and uses
upstream's own entry points, exactly like the official inference script:

```python
from engine.core import YAMLConfig
cfg = YAMLConfig(<config .yml>, resume=<checkpoint .pth>)
model = cfg.model.deploy()            # EMA weights, eval mode
post  = cfg.postprocessor.deploy()    # cx, cy, w, h, angle(rad) in original pixels
```

A model folder declares it via `model.json`:

```json
{ "backend": "rio_detr", "repo": "repo:third_party/RiO-DETRv2",
  "weights": "last.pth", "config": "riov2_x_gzda_infer.yaml" }
```

## Requirements

`engine/__init__.py` imports `cv2` and `engine.data` at import time, so the
inference environment needs **torch 2.2.2 + torchvision 0.17.2 + timm + opencv +
numba + pycocotools + PyYAML**. That set conflicts with both `vecrot-det`
(mmcv-full, torch 2.0.1) and `vecrot-vllm` (torch 2.4.1), hence the dedicated
`vecrot-rio` environment — see `docs/安装教程-自建环境.md` §5.6.
