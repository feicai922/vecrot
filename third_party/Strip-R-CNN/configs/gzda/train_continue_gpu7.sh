#!/usr/bin/env bash
set -euo pipefail
cd /data1/wzh26/code/Strip-R-CNN
export CUDA_VISIBLE_DEVICES=6
unset CUDA_DEVICE_ORDER
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2
PYTHON=/home/wzh26/anaconda3/envs/p2pformer/bin/python
WORK=work_dirs/gzda_strip_s
RUN="$WORK/continue_6ep"
mkdir -p "$RUN"
latest=$(readlink "$WORK/latest.pth" || true)
resume="$WORK/best_mAP_iter_90288.pth"
if [[ "$latest" =~ ^iter_([0-9]+)\.pth$ ]] && (( BASH_REMATCH[1] > 90288 )) && [[ -s "$WORK/$latest" ]]; then
    resume="$WORK/$latest"
fi
printf 'Resume checkpoint: %s\n' "$resume"
"$PYTHON" -u tools/train.py configs/gzda/strip_s_gzda_continue_6ep.py --work-dir "$WORK" --resume-from "$resume" --seed 42 &
train_pid=$!
"$PYTHON" -u configs/gzda/watch_continue.py "$train_pid" >> "$RUN/watcher.log" 2>&1 &
watch_pid=$!
trap 'kill "$watch_pid" 2>/dev/null || true' EXIT
set +e
wait "$train_pid"
result=$?
wait "$watch_pid"
set -e
exit "$result"
