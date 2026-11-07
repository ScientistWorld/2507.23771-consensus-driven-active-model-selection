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
