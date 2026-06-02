"""Visible train-slice evaluator for active model selection traces.

This evaluator reads method-produced JSON traces and rescoring data only. It
does not import or execute code from method/.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


SPLIT_SEED = 250723771


def load_benchmark(data_dir: Path, task: str) -> tuple[np.ndarray, np.ndarray]:
    npz_path = data_dir / f"{task}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing benchmark file: {npz_path}")
    data = np.load(npz_path)
    predictions = np.asarray(data["predictions"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=np.int64)
    if predictions.ndim != 3:
        raise ValueError(f"Expected predictions with shape (H,N,C), got {predictions.shape}")
    if labels.ndim != 1 or labels.shape[0] != predictions.shape[1]:
        raise ValueError(f"Expected labels shape ({predictions.shape[1]},), got {labels.shape}")
    return predictions, labels


def train_item_indices(num_items: int) -> np.ndarray:
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(num_items)
    return np.sort(order[: (num_items + 1) // 2])


def subset_losses(predictions: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    pred_classes = predictions[:, indices, :].argmax(axis=-1)
    return 1.0 - (pred_classes == labels[indices][None, :]).mean(axis=1)


def trace_paths(method_dir: Path) -> list[Path]:
    def seed_number(path: Path) -> int:
        match = re.search(r"seed_(\d+)\.json$", path.name)
        return int(match.group(1)) if match else -1

    return sorted(method_dir.glob("seed_*.json"), key=seed_number)


def summarize_trace(trace: dict, losses: np.ndarray, slice_size: int) -> dict[str, float]:
    steps = trace.get("steps", [])
    if not steps:
        raise ValueError("Trace has no steps")

    best_loss = float(losses.min())
    regrets = []
    for step in steps:
        chosen_model = int(step["chosen_model"])
        if chosen_model < 0 or chosen_model >= losses.shape[0]:
            raise ValueError(f"chosen_model out of range: {chosen_model}")
        regrets.append(float((losses[chosen_model] - best_loss) * 100.0))

    budget = int(trace.get("budget", len(steps) - 1))
    post_query = regrets[1 : min(budget + 1, len(regrets))]
    return {
        "initial_regret": float(regrets[0]),
        "cumulative_regret_at_100": float(sum(post_query)),
        "instant_regret_at_100": float(regrets[-1]),
        "label_budget": float(budget),
        "num_models": float(losses.shape[0]),
        "num_items": float(slice_size),
    }


def aggregate_method(method_dir: Path, losses: np.ndarray, slice_size: int) -> dict[str, float]:
    paths = trace_paths(method_dir)
    if not paths:
        raise FileNotFoundError(f"No seed_*.json traces found in {method_dir}")

    per_seed = [summarize_trace(json.loads(path.read_text()), losses, slice_size) for path in paths]
    keys = per_seed[0].keys()
    return {key: float(sum(row[key] for row in per_seed) / len(per_seed)) for key in keys}


def enumerate_methods(results_dir: Path, task: str) -> list[str]:
    task_dir = results_dir / task
    if not task_dir.exists():
        raise FileNotFoundError(f"Missing task results directory: {task_dir}")
    methods = [
        path.name
        for path in sorted(task_dir.iterdir())
        if path.is_dir() and any(path.glob("seed_*.json"))
    ]
    if not methods:
        raise FileNotFoundError(f"No method directories with seed_*.json traces found in {task_dir}")
    return methods


def reference_metric_names(reference_path: Path, experiment: str) -> set[str] | None:
    if not reference_path.exists():
        return None
    reference = json.loads(reference_path.read_text())
    metrics = reference.get("experiments", {}).get(experiment, {}).get("metrics")
    if not isinstance(metrics, dict):
        return None
    return set(metrics)


def filter_known_metrics(scores: dict[str, float], known_metrics: set[str] | None) -> dict[str, float]:
    if known_metrics is None:
        return scores
    return {name: value for name, value in scores.items() if name in known_metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="outputs/coda")
    parser.add_argument("--data-dir", default="data/downloads/modelselector")
    parser.add_argument("--task", default="glue_rte")
    parser.add_argument("--experiment", default="glue_rte_active_selection")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--output", default="scoring/scores_train.json")
    parser.add_argument("--reference", default="scoring/reference.json")
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    predictions, labels = load_benchmark(Path(args.data_dir), args.task)
    indices = train_item_indices(labels.shape[0])
    losses = subset_losses(predictions, labels, indices)
    methods = args.methods if args.methods else enumerate_methods(results_dir, args.task)

    out_path = Path(args.output)
    known_metrics = reference_metric_names(Path(args.reference), args.experiment)
    if args.merge_existing and out_path.exists():
        scores = json.loads(out_path.read_text())
    else:
        scores = {"experiments": {}}
    scores.setdefault("experiments", {})[args.experiment] = {"results": {}}

    for method in methods:
        method_dir = results_dir / args.task / method
        scores["experiments"][args.experiment]["results"][method] = filter_known_metrics(
            aggregate_method(method_dir, losses, len(indices)),
            known_metrics,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
