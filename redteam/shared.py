"""Candidate utilities: only delivered method, data, and train evaluator imports."""
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path('/home/user')
sys.path.insert(0, str(ROOT))
from data.benchmark_io import load_prediction_benchmark
from method.coda_reproduction import CODASelector, EmpiricalRiskSelector, run_selector

TASKS = ('glue_rte', 'glue_sst2')


def benchmark(task):
    return load_prediction_benchmark(ROOT / 'data/downloads/modelselector', task)


def save(out, task, trace, seed=0, method='coda'):
    path = out / task / method / f'seed_{seed}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, indent=2) + '\n')


def constant_trace(chosen, budget=100, length=101):
    return {'budget': budget, 'steps': [{'step': t, 'chosen_model': chosen} for t in range(length)]}


def use_reference_traces(out, transform):
    """Explicitly allowed reuse for checks that only alter written/scored fields.

    These traces are freshly rerun by this audit, not exposed hidden answers.
    The shortcut can equally be applied to any scientist's own generated traces.
    """
    for task in TASKS:
        for path in sorted((ROOT / 'redteam/reference_seed0/traces' / task / 'coda').glob('*.json')):
            trace = json.loads(path.read_text())
            transform(trace)
            save(out, task, trace)


def selector_trace(selector, bench, budget, seed):
    random.seed(seed)
    np.random.seed(seed)
    steps = []
    for step in range(budget + 1):
        row = {'step': step, 'chosen_model': selector.best_model()}
        steps.append(row)
        if step < budget:
            item, score = selector.next_item()
            row['queried_item'] = item
            row['query_score'] = score
            selector.add_label(item, int(bench.labels[item]))
    return {'budget': budget, 'seed': seed, 'steps': steps}
