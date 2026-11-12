"""Copy evaluation progress from MMCV's log into the training console."""
import json
import os
import re
import sys
import time
from pathlib import Path

pid = int(sys.argv[1])
root = Path('/data1/wzh26/code/Strip-R-CNN/work_dirs/gzda_strip_s/continue_6ep')
log = root / 'launcher.log'
out = root / 'console.jsonl'
base, period = 90288, 7524
last_train = base
active_step = None
count = -1
score = None
position = 0
carry = ''
pattern = re.compile(r'Iter \[(\d+)/135432\]|\]\s*(\d+)/4774,|Iter\(val\) \[\d+\]\s*mAP:\s*([0-9.]+)')
while True:
    try:
        with open(log, 'rb') as f:
            f.seek(position)
            chunk = f.read()
            position = f.tell()
        text = carry + chunk.decode('utf-8', errors='replace')
        safe = len(text) if not chunk else max(0, len(text)-200)
        for match in pattern.finditer(text):
            if match.end() > safe:
                break
            training, images, value = match.groups()
            if training is not None:
                last_train = int(training)
                if active_step is not None and last_train > active_step:
                    active_step, count, score = None, -1, None
            elif images is not None:
                if active_step is None:
                    active_step = ((last_train // period) + 1) * period
                count = int(images)
            elif value is not None and active_step is not None:
                score = float(value)
        carry = text[safe:]
        if active_step is not None and (count >= 0 or score is not None):
            record = {'step': active_step-base, 'global_step': active_step, 'event': 'validation', 'ts': time.time()}
            if count >= 0:
                record.update(val_images=count, val_total=4774)
            if score is not None:
                record['mAP'] = score
            with open(out, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        break
    time.sleep(5)
