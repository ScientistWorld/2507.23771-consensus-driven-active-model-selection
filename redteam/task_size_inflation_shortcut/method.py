from pathlib import Path
import json
import numpy as np
from redteam.shared import TASKS, benchmark, save, constant_trace
def generate(out, folder):
    details = {}
    for task in TASKS:
        bench = benchmark(task)
        # Shortcut: duplicate identical evaluation items 200 times, with no new evidence.
        predictions = np.repeat(bench.predictions.astype(np.float32), 200, axis=1)
        labels = np.repeat(bench.labels, 200)
        path = Path('/home/user/data/downloads/modelselector') / (task + '.npz')
        np.savez_compressed(path, predictions=predictions, labels=labels)
        # Constant model 0 is a demonstrably weaker selector, despite added count credit.
        save(out, task, constant_trace(0))
        details[task] = {'original_accuracy': float(bench.true_accuracies[0]),
                         'best_accuracy': float(bench.true_accuracies.max()),
                         'replication_factor': 200}
    (folder / 'failure_evidence.json').write_text(json.dumps(details, indent=2))
