"""Mirror validation progress into the existing training console log."""
import json
import os
import re
import sys
import time

root = '/data1/wzh26/code/Strip-R-CNN/work_dirs/gzda_strip_s'
log = os.path.join(root, 'launcher.log')
out = os.path.join(root, 'console.jsonl')
pid = int(sys.argv[1])
last_step = 0
active_step = None
last_count = -1
last_score = None
position = os.path.getsize(log) if os.path.exists(log) else 0
pattern = re.compile(r'Iter \[(\d+)/90288\]|\]\s*(\d+)/4774,|Iter\(val\) \[\d+\]\s*mAP:\s*([0-9.]+)')
while True:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        break
    try:
        with open(log, 'rb') as f:
            f.seek(position)
            chunk = f.read()
            position = f.tell()
        text = chunk.decode('utf-8', errors='replace')
        for match in pattern.finditer(text):
            training, count, score = match.groups()
            if training is not None:
                last_step = int(training)
                if active_step is not None and last_step > active_step:
                    active_step = None
                    last_count = -1
                    last_score = None
            elif count is not None:
                if active_step is None:
                    active_step = ((last_step // 7524) + 1) * 7524
                last_count = int(count)
            elif score is not None and active_step is not None:
                last_score = float(score)
        if active_step is not None and (last_count >= 0 or last_score is not None):
            record = {'step': active_step, 'event': 'validation', 'ts': time.time()}
            if last_count >= 0:
                record.update(val_images=last_count, val_total=4774)
            if last_score is not None:
                record['mAP'] = last_score
            with open(out, 'a') as f:
                f.write(json.dumps(record) + '\n')
    except OSError:
        pass
    time.sleep(10)
