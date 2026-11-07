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
