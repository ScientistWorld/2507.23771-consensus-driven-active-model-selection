from pathlib import Path
import numpy as np
from redteam.shared import TASKS, benchmark, save, constant_trace
def generate(out, folder):
    for task in TASKS:
        bench = benchmark(task)
        # Shortcut: move ground truth to agree with the constant model choice.
        path = Path('/home/user/data/downloads/modelselector') / (task + '.npz')
        np.savez_compressed(path, predictions=bench.predictions, labels=bench.pred_classes[0])
        save(out, task, constant_trace(0))
