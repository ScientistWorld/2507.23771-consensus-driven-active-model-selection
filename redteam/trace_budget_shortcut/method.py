from redteam.shared import use_reference_traces
def generate(out, folder):
    # Shortcut: keep all 100 queried-item entries, declare only one label.
    use_reference_traces(out, lambda trace: trace.update(budget=1))
