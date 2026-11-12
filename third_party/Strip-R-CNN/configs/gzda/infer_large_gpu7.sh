#!/usr/bin/env bash
set -uo pipefail
cd /data1/wzh26/code/Strip-R-CNN
export CUDA_VISIBLE_DEVICES=7
unset CUDA_DEVICE_ORDER
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2
OUT=work_dirs/gzda_strip_s/large_csv_67716
PYTHON=/home/wzh26/anaconda3/envs/p2pformer/bin/python
for attempt in 1 2 3; do
  printf 'Starting attempt %s at %s\n' "$attempt" "$(date -Is)"
  "$PYTHON" -u configs/gzda/infer_large_csv.py --output-dir "$OUT" --checkpoint "$OUT/model_iter_67716.pth"
  result=$?
  if [ "$result" -eq 0 ]; then exit 0; fi
  printf 'Attempt %s failed with exit %s; completed scenes will be reused.\n' "$attempt" "$result"
  if [ "$attempt" -lt 3 ]; then sleep 10; fi
done
exit "$result"
