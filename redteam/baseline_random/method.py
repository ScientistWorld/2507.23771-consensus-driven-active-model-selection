from redteam.shared import TASKS, benchmark, save, run_selector
def generate(out, folder):
    for task in TASKS:
        for seed in range(3):
            trace = run_selector(benchmark(task), 'random', 100, seed, 128)
            save(out, task, trace, seed)
