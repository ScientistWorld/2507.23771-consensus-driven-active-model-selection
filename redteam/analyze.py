"""Summarize already completed audit runs; no model computation or evaluator edits."""
from pathlib import Path
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'redteam'


def read(path):
    return json.loads(path.read_text())


def main():
    reference = read(AUDIT / 'original/scores_test.json')['experiments']
    schema = read(AUDIT / 'original/reference.json')['experiments']
    cases = {}
    for path in sorted(AUDIT.glob('*/result.json')):
        status = read(path)
        assert 'candidate_error' not in status, path
        case = {'test_exit_code': status['test_exit_code'],
                'rank_delta': status.get('rank_delta')}
        if status['test_exit_code'] == 0:
            scores = read(path.parent / 'scores_test.json')['experiments']
            metrics = {name: exp['results']['coda'] for name, exp in scores.items()
                       if 'coda' in exp['results']}
            case['metrics'] = metrics
            if case['rank_delta'] is not None:
                calculated = sum(
                    schema[e]['weight'] * spec['coefficient']
                    * (metrics[e][m] - reference[e]['results']['coda'][m])
                    for e in reference for m, spec in schema[e]['metrics'].items())
                assert math.isclose(calculated, case['rank_delta'], abs_tol=1e-9), path
        else:
            case['rejection'] = status['test_error_tail'].strip().splitlines()[-1]
        case['shortcut'] = path.parent.name.endswith('_shortcut')
        case['pays'] = bool(case['shortcut'] and case['rank_delta'] is not None
                            and case['rank_delta'] >= 0)
        cases[path.parent.name] = case

    seeds = [cases[f'reference_seed{s}']['rank_delta'] for s in range(3)]
    genuine_queries = {}
    for path in sorted((AUDIT / 'honest_stable_stop/traces').glob('*/coda/seed_*.json')):
        trace = read(path)
        queries = [row['queried_item'] for row in trace['steps'] if 'queried_item' in row]
        assert len(queries) == len(set(queries)) == trace['budget']
        assert len(trace['steps']) == trace['budget'] + 1
        genuine_queries[str(path.relative_to(AUDIT))] = len(queries)

    summary = {
        'rank_definition': 'sum(weight[e] * coefficient[e,m] * (metric[e,m] - stored_test_reference[e,m]))',
        'reference_seed_ranks': seeds,
        'reference_seed_mean': statistics.mean(seeds),
        'reference_seed_spread': max(seeds) - min(seeds),
        'reference_seed_sample_sd': statistics.stdev(seeds),
        'scored_experiment_weight': sum(schema[e]['weight'] for e in reference),
        'not_scored_by_default': sorted(set(schema) - set(reference)),
        'passing_shortcut_count': sum(c['pays'] for c in cases.values()),
        'case_count': len(cases),
        'genuine_stop_query_counts': genuine_queries,
        'cases': cases,
    }
    (AUDIT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ('cases', 'genuine_stop_query_counts')}, indent=2))


if __name__ == '__main__':
    main()
