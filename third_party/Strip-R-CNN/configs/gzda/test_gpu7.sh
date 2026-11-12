#!/usr/bin/env bash
set -euo pipefail
cd /data1/wzh26/code/Strip-R-CNN
export CUDA_VISIBLE_DEVICES=7
unset CUDA_DEVICE_ORDER
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2
PYTHON=/home/wzh26/anaconda3/envs/p2pformer/bin/python
OUT="$PWD/work_dirs/gzda_strip_s/test_67716"
"$PYTHON" -u tools/test.py configs/gzda/strip_s_gzda.py work_dirs/gzda_strip_s/best_mAP_iter_67716.pth --out "$OUT/predictions.pkl" --eval mAP --work-dir "$OUT"
"$PYTHON" -u configs/gzda/export_test_results.py --output-dir "$OUT"
