from redteam.shared import TASKS, benchmark, save, run_selector
def generate(out, folder):
    for task in TASKS:
        trace = run_selector(benchmark(task), 'coda', 100, 1, 128 if task == 'glue_rte' else 64)
        save(out, task, trace, 1)
