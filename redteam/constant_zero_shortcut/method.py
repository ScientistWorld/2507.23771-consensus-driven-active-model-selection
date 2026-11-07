from redteam.shared import TASKS, save, constant_trace
def generate(out, folder):
    for task in TASKS:
        save(out, task, constant_trace(0))
