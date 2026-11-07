"""Install one candidate, run unmodified train and no-argument hidden test, restore.

Candidate code never imports eval/test or reads original scores/reference metadata.
The subprocess boundary also ensures candidate monkeypatches cannot affect scoring.
Only the audit harness reads reference metadata, to calculate the specified rank.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
TASKS = ('glue_rte', 'glue_sst2')
sys.path.insert(0, str(ROOT))


def rank(scores, method='coda'):
    reference = json.loads((AUDIT / 'original/scores_test.json').read_text())
    schema = json.loads((AUDIT / 'original/reference.json').read_text())
    contributions = {}
    for experiment, exp in reference['experiments'].items():
        result = scores.get('experiments', {}).get(experiment, {}).get('results', {}).get(method)
        if result is None:
            return None, {'error': f'missing {experiment}/{method}'}
        old = exp['results']['coda']
        config = schema['experiments'][experiment]
        contributions[experiment] = {
            metric: config['weight'] * spec['coefficient'] * (result[metric] - old[metric])
            for metric, spec in config['metrics'].items()
        }
    return sum(sum(exp.values()) for exp in contributions.values()), contributions


def run(name):
    folder = AUDIT / name
    state = AUDIT / '_state'
    state.mkdir(exist_ok=True)
    # Every run restores all persistent inputs, even after rejected traces.
    data_dir = ROOT / 'data/downloads/modelselector'
    saved_data = state / 'data'
    saved_data.mkdir(exist_ok=True)
    for task in TASKS:
        src = data_dir / f'{task}.npz'
        dst = saved_data / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        shutil.copy2(dst, src)
    active = ROOT / 'outputs/coda'
    old_active = state / 'original_active'
    if active.is_symlink():
        active.unlink()
    if active.exists():
        if old_active.exists():
            raise RuntimeError('Unexpected output directory; refusing to overwrite original backup')
        active.rename(old_active)
    traces = folder / 'traces'
    if traces.exists():
        shutil.rmtree(traces)
    traces.mkdir()
    active.symlink_to(traces)

    saved_scores = state / 'scores'
    saved_scores.mkdir(exist_ok=True)
    for fname in ('scores_train.json', 'scores_test.json'):
        src = ROOT / 'scoring' / fname
        shutil.copy2(src, saved_scores / fname)
    started = time.time()
    status = {'name': name}
    try:
        spec = importlib.util.spec_from_file_location('audit_candidate', folder / 'method.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.generate(traces, folder)
        env = os.environ.copy()
        for key in ('DATA_DIR', 'METHODS', 'MERGE_EXISTING', 'UNSUPERVISED_RESULTS_DIR'):
            env.pop(key, None)
        # No evaluator modifications, arguments, hooks, or Python environment tampering.
        for phase in ('train', 'test'):
            if phase == 'test' and hasattr(module, 'before_test'):
                module.before_test(folder)
            with (folder / f'{phase}.log').open('w') as log:
                result = subprocess.run(['bash', f'scripts/evaluate_{phase}.sh'],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            status[f'{phase}_exit_code'] = result.returncode
            if result.returncode == 0:
                shutil.copy2(ROOT / f'scoring/scores_{phase}.json', folder / f'scores_{phase}.json')
            else:
                status[f'{phase}_error_tail'] = (folder / f'{phase}.log').read_text()[-2500:]
        if status.get('test_exit_code') == 0:
            scores = json.loads((folder / 'scores_test.json').read_text())
            status['rank_delta'], status['rank_contributions'] = rank(scores)
    except Exception as error:
        status['candidate_error'] = repr(error)
        raise
    finally:
        status['seconds'] = time.time() - started
        (folder / 'result.json').write_text(json.dumps(status, indent=2) + '\n')
        for task in TASKS:
            shutil.copy2(saved_data / f'{task}.npz', data_dir / f'{task}.npz')
        for fname in ('scores_train.json', 'scores_test.json'):
            shutil.copy2(saved_scores / fname, ROOT / 'scoring' / fname)
        active.unlink()
        if old_active.exists():
            old_active.rename(active)
    print(json.dumps(status), flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
