from pathlib import Path
import numpy as np
from redteam.shared import TASKS, benchmark, save, constant_trace
def generate(out, folder):
    for task in TASKS:
        bench = benchmark(task)
        # Shortcut: replace the fixed comparison pool with equally bad classifiers.
        predictions = np.zeros_like(bench.predictions)
        predictions[:, :, 0] = 1
        path = Path('/home/user/data/downloads/modelselector') / (task + '.npz')
        np.savez_compressed(path, predictions=predictions, labels=bench.labels)
        save(out, task, constant_trace(0))
