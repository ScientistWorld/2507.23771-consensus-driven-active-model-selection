import json
import random
import numpy as np
from redteam.shared import TASKS, benchmark, save, selector_trace
from redteam.fast_coda import FastCODA
def generate(out, folder):
    for task in TASKS:
        bench = benchmark(task)
        for seed in range(3):
            random.seed(seed)
            np.random.seed(seed)
            selector = FastCODA(bench, eta=1.0, num_grid=128 if task == 'glue_rte' else 64)
            trace = selector_trace(selector, bench, 100, seed)
            save(out, task, trace, seed)
