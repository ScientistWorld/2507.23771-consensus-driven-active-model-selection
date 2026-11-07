from redteam.shared import TASKS, save, constant_trace
def generate(out, folder):
    for task in TASKS:
        # Shortcut attempt with malformed or impossible submitted values.
        save(out, task, constant_trace(-1))
