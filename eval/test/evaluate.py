"""Held-out test-slice evaluator for active model selection traces.

The test slice reads method-produced traces, then rescores chosen models on
the disjoint held-out item partition. It does not import method code.
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


def test_item_indices(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pred_classes = predictions.argmax(axis=-1)
    correct = pred_classes == labels[None, :]
    item_error = 1.0 - correct.mean(axis=0)
    signatures = [np.packbits(correct[:, item]).tobytes() for item in range(labels.shape[0])]
    rng = np.random.default_rng(SPLIT_SEED)
    jitter = rng.random(labels.shape[0])
    order = np.array(
        sorted(
            range(labels.shape[0]),
            key=lambda item: (int(labels[item]), float(item_error[item]), signatures[item], float(jitter[item])),
        )
    )
    return np.sort(order[np.arange(order.shape[0]) % 2 == 1])


def subset_losses(predictions: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    pred_classes = predictions[:, indices, :].argmax(axis=-1)
    return 1.0 - (pred_classes == labels[indices][None, :]).mean(axis=1)


def trace_paths(method_dir: Path) -> list[Path]:
    def seed_number(path: Path) -> int:
        match = re.search(r"seed_(\d+)\.json$", path.name)
        return int(match.group(1)) if match else -1

    return sorted(method_dir.glob("seed_*.json"), key=seed_number)


def regret_at(regrets: list[float], step: int) -> float:
    if step < len(regrets):
        return regrets[step]
    return regrets[-1]


def cumulative_regret(regrets: list[float], horizon: int) -> float:
    return float(sum(regret_at(regrets, step) for step in range(1, horizon + 1)))


def summarize_trace(
    trace: dict,
    losses: np.ndarray,
    slice_size: int,
    cumulative_horizon: int,
) -> dict[str, float]:
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
    final_regret = float(regrets[-1])
    return {
        "initial_regret": float(regrets[0]),
        "cumulative_regret_at_25": cumulative_regret(regrets, min(25, cumulative_horizon)),
        "cumulative_regret_at_50": cumulative_regret(regrets, min(50, cumulative_horizon)),
        "cumulative_regret_at_100": cumulative_regret(regrets, min(100, cumulative_horizon)),
        "instant_regret_at_50": regret_at(regrets, 50),
        "instant_regret_at_100": regret_at(regrets, 100),
        "near_optimal_at_100": float(final_regret <= 1.0),
        "label_budget": float(budget),
        "num_models": float(losses.shape[0]),
        "num_items": float(slice_size),
    }


def trace_budget(path: Path) -> int:
    trace = json.loads(path.read_text())
    return int(trace.get("budget", len(trace.get("steps", [])) - 1))


def experiment_trace_paths(method_dir: Path, cumulative_horizon: int) -> list[Path]:
    paths = trace_paths(method_dir)
    if not paths:
        raise FileNotFoundError(f"No seed_*.json traces found in {method_dir}")
    if cumulative_horizon > 0:
        active_paths = [path for path in paths if trace_budget(path) > 0]
        if active_paths:
            return active_paths
    return paths


def aggregate_method(
    method_dir: Path,
    losses: np.ndarray,
    slice_size: int,
    cumulative_horizon: int,
) -> dict[str, float]:
    paths = experiment_trace_paths(method_dir, cumulative_horizon)
    per_seed = [
        summarize_trace(json.loads(path.read_text()), losses, slice_size, cumulative_horizon)
        for path in paths
    ]
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


def train_score_methods(train_scores_path: Path, experiment: str) -> list[str]:
    if not train_scores_path.exists():
        return []
    scores = json.loads(train_scores_path.read_text())
    results = scores.get("experiments", {}).get(experiment, {}).get("results", {})
    if not isinstance(results, dict):
        return []
    return [method for method in results if not method.startswith("_")]


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


def reference_experiments(reference_path: Path) -> list[str]:
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing reference file: {reference_path}")
    reference = json.loads(reference_path.read_text())
    experiments = reference.get("experiments", {})
    if not isinstance(experiments, dict) or not experiments:
        raise ValueError(f"No experiments found in {reference_path}")
    return [
        name
        for name in experiments
        if name.endswith("_active_selection") or name.endswith("_unsupervised_selection")
    ]


def experiment_task(experiment: str, task_override: str | None) -> str:
    if task_override:
        return task_override
    for suffix in ("_active_selection", "_unsupervised_selection"):
        if experiment.endswith(suffix):
            return experiment[: -len(suffix)]
    raise ValueError(f"Cannot infer task from experiment name: {experiment}")


def selected_experiments(args: argparse.Namespace) -> list[str]:
    if args.experiment:
        return [args.experiment]
    if args.task:
        return [f"{args.task}_active_selection"]
    return reference_experiments(Path(args.reference))


def result_dir_for_experiment(args: argparse.Namespace, experiment: str) -> Path:
    if experiment.endswith("_unsupervised_selection"):
        return Path(args.unsupervised_results_dir)
    return Path(args.results_dir)


def evaluate_experiment(args: argparse.Namespace, experiment: str) -> dict[str, dict[str, float]]:
    task = experiment_task(experiment, args.task)
    results_dir = result_dir_for_experiment(args, experiment)
    predictions, labels = load_benchmark(Path(args.data_dir), task)
    indices = test_item_indices(predictions, labels)
    if len(indices) == 0:
        raise ValueError(f"Task {task} has no held-out test items")
    losses = subset_losses(predictions, labels, indices)
    cumulative_horizon = 0 if experiment.endswith("_unsupervised_selection") else 100

    train_methods = train_score_methods(Path(args.train_scores), experiment)
    if train_methods:
        methods = train_methods
    elif args.methods:
        methods = args.methods
    else:
        methods = enumerate_methods(results_dir, task)

    known_metrics = reference_metric_names(Path(args.reference), experiment)
    results = {}
    for method in methods:
        method_dir = results_dir / task / method
        results[method] = filter_known_metrics(
            aggregate_method(method_dir, losses, len(indices), cumulative_horizon),
            known_metrics,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="outputs/coda")
    parser.add_argument("--unsupervised-results-dir", default="outputs/coda_unsupervised")
    parser.add_argument("--data-dir", default="data/downloads/modelselector")
    parser.add_argument("--task")
    parser.add_argument("--experiment")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--output", default="scoring/scores_test.json")
    parser.add_argument("--reference", default="scoring/reference.json")
    parser.add_argument("--train-scores", default="scoring/scores_train.json")
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.output)
    if args.merge_existing and out_path.exists():
        scores = json.loads(out_path.read_text())
    else:
        scores = {"slice": "test", "experiments": {}}
    scores["slice"] = "test"
    scores.setdefault("experiments", {})

    for experiment in selected_experiments(args):
        scores["experiments"][experiment] = {"results": evaluate_experiment(args, experiment)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
