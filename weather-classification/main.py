def mkdir(directory: str, name: str) -> str:
    filenames = os.listdir(directory)
    count = 1
    while name + str(count) in filenames:
        count += 1
    os.mkdir(name + str(count))
    save_dir = os.path.join(directory, name + str(count))
    return save_dir

def trace():
    pid = os.getpid()

    while True:
        if os.path.exists(os.path.join(SAVE_DIR, 'kill')):
            os.kill(pid, signal.SIGTERM)
            break
        time.sleep(0.2)


import os
import sys
import signal
import time
import traceback
import threading
from filelock import FileLock

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATASET_DIR = sys.argv[1]
SAVE_DIR = sys.argv[2] if len(sys.argv) > 2 else mkdir(directory='.', name='save')

# Start tracing process ids
threading.Thread(target=trace, daemon=True).start()

# unit_counts = [50, 100, 200, 500, 1000, 2000, 4096]
# lrs = [i/1000 for i in range(1, 11)]
# batch_sizes = [16, 32, 64, 128]
unit_counts = [25, 50, 100, 200, 500, 1000, 2000, 4096]
lrs = [i/10000 for i in range(5, 11)]
batch_sizes = [4, 8]

from vgg import VGG
for unit_count in unit_counts:
    for lr in lrs:
        for batch_size in batch_sizes:
            model_name = str(unit_count) + '_' + str(lr) + '_' + str(batch_size)
            try:
                with FileLock(os.path.join(SAVE_DIR, 'lock')):
                    if model_name not in os.listdir(SAVE_DIR):
                        os.mkdir(os.path.join(SAVE_DIR, model_name))
                    else:
                        continue
                VGG(
                    dataset_dir=DATASET_DIR,
                    save_dir=os.path.join(SAVE_DIR, model_name),
                    unit_count=unit_count,
                    lr=lr,
                    batch_size=batch_size
                ).run()
            except Exception:
                with open(os.path.join(SAVE_DIR, 'exception.txt'), mode='a') as f:
                    f.write(traceback.format_exc())
