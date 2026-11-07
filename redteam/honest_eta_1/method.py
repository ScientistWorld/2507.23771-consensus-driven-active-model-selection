import json
import random
import numpy as np
from redteam.shared import TASKS, benchmark, save, selector_trace
from redteam.fast_coda import FastCODA
def generate(out, folder):
    for task in TASKS:
        bench = benchmark(task)
        for seed in range(3):
            path = out / task / 'coda' / f'seed_{seed}.json'
            if path.exists():
                trace = json.loads(path.read_text())
                if trace.get('budget') == 100 and len(trace.get('steps', [])) == 101:
                    print(f'Resume: complete {task} seed {seed}', flush=True)
                    continue
            random.seed(seed)
            np.random.seed(seed)
            selector = FastCODA(bench, eta=1.0, num_grid=128 if task == 'glue_rte' else 64)
            trace = selector_trace(selector, bench, 100, seed)
            save(out, task, trace, seed)
            print(f'Completed eta=1 {task} seed {seed}', flush=True)
