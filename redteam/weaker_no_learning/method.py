import numpy as np
from redteam.shared import TASKS, benchmark, save, constant_trace, CODASelector
def generate(out, folder):
    for task in TASKS:
        bench = benchmark(task)
        model = CODASelector(bench, num_grid=128 if task == 'glue_rte' else 64).best_model()
        for seed in range(3):
            # Weaker ablation: consume 100 labels but remove the posterior updates.
            trace = constant_trace(model)
            order = np.random.default_rng(seed).permutation(bench.labels.size)[:100]
            for row, item in zip(trace['steps'], order):
                row['queried_item'] = int(item)
                row['observed_label'] = int(bench.labels[item])
            save(out, task, trace, seed)
