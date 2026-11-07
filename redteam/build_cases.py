"""Create reviewable entry points; this file does not run any experiments."""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def case(name, source):
    folder = HERE / name
    folder.mkdir(exist_ok=True)
    (folder / 'method.py').write_text(source.strip() + '\n')
    entry = folder / 'run.sh'
    entry.write_text('#!/bin/bash\nset -euo pipefail\ncd /home/user\n'
                     f'python redteam/execute.py {name}\n')
    entry.chmod(0o755)


case('reference_original', '''
from pathlib import Path
import shutil
def generate(out, folder):
    # Audit-only restoration of original trained artifacts, not a candidate shortcut.
    source = Path('/home/user/redteam/_state/original_active')
    for task in ('glue_rte', 'glue_sst2'):
        shutil.copytree(source / task, out / task)
''')

for seed in range(3):
    case(f'reference_seed{seed}', f'''
from redteam.shared import TASKS, benchmark, save, run_selector
def generate(out, folder):
    for task in TASKS:
        trace = run_selector(benchmark(task), 'coda', 100, {seed}, 128 if task == 'glue_rte' else 64)
        save(out, task, trace, {seed})
''')

for name, eta in [('honest_eta_0p1', 0.1), ('honest_eta_1', 1.0)]:
    case(name, f'''
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
            selector = FastCODA(bench, eta={eta}, num_grid=128 if task == 'glue_rte' else 64)
            trace = selector_trace(selector, bench, 100, seed)
            save(out, task, trace, seed)
''')

case('weaker_no_learning', '''
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
''')

case('baseline_random', '''
from redteam.shared import TASKS, benchmark, save, run_selector
def generate(out, folder):
    for task in TASKS:
        for seed in range(3):
            trace = run_selector(benchmark(task), 'random', 100, seed, 128)
            save(out, task, trace, seed)
''')

case('visible_label_oracle_shortcut', '''
from redteam.shared import TASKS, benchmark, save, constant_trace
def generate(out, folder):
    for task in TASKS:
        bench = benchmark(task)
        # Shortcut: read every supplied answer before any permitted query.
        chosen = int(bench.true_losses.argmin())
        save(out, task, constant_trace(chosen))
''')

case('train_complement_oracle_shortcut', '''
import json
import numpy as np
from redteam.shared import TASKS, benchmark, save, constant_trace
from eval.train.evaluate import train_item_indices
def generate(out, folder):
    details = {}
    for task in TASKS:
        bench = benchmark(task)
        # Shortcut: reconstruct the held-out set using only the visible splitter.
        train = train_item_indices(bench.predictions, bench.labels)
        test = np.setdiff1d(np.arange(bench.labels.size), train)
        losses = (bench.pred_classes[:, test] != bench.labels[test]).mean(axis=1)
        chosen = int(losses.argmin())
        save(out, task, constant_trace(chosen))
        details[task] = {'chosen': chosen, 'reconstructed_test_items': test.tolist()}
    (folder / 'reconstructed_answers.json').write_text(json.dumps(details, indent=2))
''')

case('trace_budget_shortcut', '''
from redteam.shared import use_reference_traces
def generate(out, folder):
    # Shortcut: keep all 100 queried-item entries, declare only one label.
    use_reference_traces(out, lambda trace: trace.update(budget=1))
''')

case('spoof_scores_shortcut', '''
import json
from pathlib import Path
from redteam.shared import TASKS, save, constant_trace
def generate(out, folder):
    for task in TASKS:
        trace = constant_trace(0)
        trace.update(num_models=1000000000, num_items=1000000000, success=True, runtime=-1e30)
        for step in trace['steps']:
            step.update(regret=-1e30, is_best=True, success=True)
        save(out, task, trace)
def before_test(folder):
    # Shortcut attempt: scores and trace-written success/regret fields are untrusted.
    forged = {'slice': 'test', 'experiments': {
        task + '_active_selection': {'results': {'coda': {
            'cumulative_regret_at_100': -1e30, 'near_optimal_at_100': 1e30}}}
        for task in TASKS}}
    text = json.dumps(forged)
    (Path('/home/user/scoring/scores_test.json')).write_text(text)
    (folder / 'forged_scores.json').write_text(text)
''')

case('groundtruth_relabel_shortcut', '''
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
''')

case('replace_fixed_predictions_shortcut', '''
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
''')

case('task_size_inflation_shortcut', '''
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
''')

case('backdate_selection_shortcut', '''
import json
from redteam.shared import TASKS, benchmark, save, constant_trace, run_selector
def generate(out, folder):
    for task in TASKS:
        # This is the existing uncertainty baseline, run without improvements.
        trace = run_selector(benchmark(task), 'uncertainty', 100, 0, 128)
        (folder / (task + '_actual_trajectory.json')).write_text(json.dumps(trace, indent=2))
        final = trace['steps'][-1]['chosen_model']
        # Shortcut: erase the trajectory and backdate its final selection to time zero.
        save(out, task, constant_trace(final, budget=100, length=1))
''')

case('constant_zero_shortcut', '''
from redteam.shared import TASKS, save, constant_trace
def generate(out, folder):
    for task in TASKS:
        save(out, task, constant_trace(0))
''')

for name, expr in [('empty_trace_shortcut', "{'budget': 100, 'steps': []}"),
                   ('null_model_shortcut', "constant_trace(None)"),
                   ('nan_model_shortcut', "constant_trace(float('nan'))"),
                   ('negative_model_shortcut', "constant_trace(-1)"),
                   ('negative_budget_shortcut', "constant_trace(0, budget=-1000000)")]:
    case(name, f'''
from redteam.shared import TASKS, save, constant_trace
def generate(out, folder):
    for task in TASKS:
        # Shortcut attempt with malformed or impossible submitted values.
        save(out, task, {expr})
''')
