"""Job-only check that vectorization retains the reference EIG calculation."""
import json
from pathlib import Path
import sys

sys.path.insert(0, '/home/user')
import numpy as np
from redteam.shared import TASKS, benchmark, CODASelector
from redteam.fast_coda import FastCODA

records = {}
for task in TASKS:
    bench = benchmark(task)
    grid = 128 if task == 'glue_rte' else 64
    original = CODASelector(bench, num_grid=grid)
    fast = FastCODA(bench, num_grid=grid)
    rows = []
    for check in range(3):
        candidates = original._disagreement_candidates()[:16]
        a = original._eig_scores(candidates)
        b = fast._eig_scores(candidates)
        np.testing.assert_allclose(a, b, atol=2e-12, rtol=2e-9)
        rows.append(float(np.abs(a - b).max()))
        item = candidates[check]
        for selector in (original, fast):
            selector.add_label(item, int(bench.labels[item]))
    records[task] = {'maximum_absolute_EIG_difference': max(rows)}
Path('/home/user/redteam/numerics.json').write_text(json.dumps(records, indent=2))
