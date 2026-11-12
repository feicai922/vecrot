#!/usr/bin/env bash
set -euo pipefail
cd /data1/wzh26/code/Strip-R-CNN
export CUDA_VISIBLE_DEVICES=6
unset CUDA_DEVICE_ORDER
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2
PYTHON=/home/wzh26/anaconda3/envs/p2pformer/bin/python
"$PYTHON" -u tools/train.py configs/gzda/strip_s_gzda.py --work-dir work_dirs/gzda_strip_s --seed 42 --auto-resume &
train_pid=$!
"$PYTHON" -u configs/gzda/watch_validation.py "$train_pid" >> work_dirs/gzda_strip_s/validation_watcher.log 2>&1 &
watch_pid=$!
trap 'kill "$watch_pid" 2>/dev/null || true' EXIT
wait "$train_pid"
