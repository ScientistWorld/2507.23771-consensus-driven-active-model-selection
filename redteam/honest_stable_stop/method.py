"""Honest label-efficient variant: stop after the chosen model stabilizes.

The visible train curve showed late overfitting under larger online updates.
Stop when the same model has been selected for 12 consecutive rounds, subject
to a minimum of 25 and maximum of 100 labels. No full-data losses, held-out
membership, or labels outside actual queries enter this decision.
"""
import random
import numpy as np
from redteam.shared import TASKS, benchmark, save
from redteam.fast_coda import FastCODA


def generate(out, folder):
    for task in TASKS:
        bench = benchmark(task)
        for seed in range(3):
            random.seed(seed)
            np.random.seed(seed)
            selector = FastCODA(bench, eta=0.1, num_grid=128 if task == 'glue_rte' else 64)
            steps = []
            for step in range(101):
                chosen = selector.best_model()
                steps.append({'step': step, 'chosen_model': chosen})
                stable = (len(steps) >= 12
                          and all(row['chosen_model'] == chosen for row in steps[-12:]))
                if step == 100 or (step >= 25 and stable):
                    break
                item, score = selector.next_item()
                steps[-1].update(queried_item=item, query_score=score)
                selector.add_label(item, int(bench.labels[item]))
            save(out, task, {'budget': step, 'seed': seed, 'steps': steps}, seed)
            print(f'Completed honest stopping {task} seed {seed}, {step} labels', flush=True)
