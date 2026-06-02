"""Evaluate active model selection traces.

The evaluator reads JSON traces produced by any method that follows the
workspace output contract. It does not import or execute code from method/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize_trace(trace: dict, tolerance: float = 1.0) -> dict[str, float]:
    regrets = np.asarray([step["regret"] for step in trace["steps"]], dtype=np.float64)
    budget = int(trace["budget"])
    post_query = regrets[1 : min(budget + 1, len(regrets))]
    return {
        "cumulative_regret_at_100": float(post_query.sum()),
        "instant_regret_at_100": float(regrets[-1]),
        "label_budget": float(budget),
        "num_models": float(trace["num_models"]),
        "num_items": float(trace["num_items"]),
    }


def aggregate_method(method_dir: Path) -> dict[str, float]:
    traces = []
    for path in sorted(method_dir.glob("seed_*.json")):
        traces.append(json.loads(path.read_text()))
    if not traces:
        raise FileNotFoundError(f"No seed_*.json traces found in {method_dir}")

    per_seed = [summarize_trace(trace) for trace in traces]
    keys = per_seed[0].keys()
    return {key: float(np.mean([row[key] for row in per_seed])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="outputs/coda")
    parser.add_argument("--task", default="glue_rte")
    parser.add_argument("--experiment", default="glue_rte_active_selection")
    parser.add_argument("--methods", nargs="+", default=["random", "uncertainty", "coda"])
    parser.add_argument("--output", default="scoring/scores.json")
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_path = Path(args.output)
    if args.merge_existing and out_path.exists():
        scores = json.loads(out_path.read_text())
    else:
        scores = {"experiments": {}}
    scores.setdefault("experiments", {})[args.experiment] = {"results": {}}
    for method in args.methods:
        method_dir = results_dir / args.task / method
        scores["experiments"][args.experiment]["results"][method] = aggregate_method(method_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
