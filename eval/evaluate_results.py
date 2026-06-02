"""Evaluate active model selection traces.

The evaluator reads JSON traces produced by any method that follows the
workspace output contract. It does not import or execute code from method/.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


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


def losses_from_benchmark(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pred_classes = predictions.argmax(axis=-1)
    return 1.0 - (pred_classes == labels[None, :]).mean(axis=1)


def regret_at(regrets: list[float], step: int) -> float:
    if step < len(regrets):
        return regrets[step]
    return regrets[-1]


def cumulative_regret(regrets: list[float], horizon: int) -> float:
    return float(sum(regret_at(regrets, step) for step in range(1, horizon + 1)))


def summarize_trace(
    trace: dict,
    losses: np.ndarray,
    oracle_loss: float,
    num_items: int,
    tolerance: float = 1.0,
) -> dict[str, float]:
    regrets = []
    for step in trace["steps"]:
        chosen_model = int(step["chosen_model"])
        if chosen_model < 0 or chosen_model >= losses.shape[0]:
            raise ValueError(f"chosen_model out of range: {chosen_model}")
        regrets.append(float(max(0.0, losses[chosen_model] - oracle_loss) * 100.0))
    budget = int(trace["budget"])
    final_regret = regret_at(regrets, budget)
    return {
        "initial_regret": float(regrets[0]),
        "cumulative_regret_at_25": cumulative_regret(regrets, min(25, budget)),
        "cumulative_regret_at_50": cumulative_regret(regrets, min(50, budget)),
        "cumulative_regret_at_100": cumulative_regret(regrets, min(100, budget)),
        "instant_regret_at_50": regret_at(regrets, min(50, budget)),
        "instant_regret_at_100": final_regret,
        "near_optimal_at_100": float(final_regret <= tolerance),
        "label_budget": float(budget),
        "num_models": float(losses.shape[0]),
        "num_items": float(num_items),
    }


def seed_number(path: Path) -> int:
    match = re.search(r"seed_(\d+)\.json$", path.name)
    return int(match.group(1)) if match else -1


def trace_budget(path: Path) -> int:
    trace = json.loads(path.read_text())
    return int(trace.get("budget", len(trace.get("steps", [])) - 1))


def trace_paths(method_dir: Path, required_budget: int | None) -> list[Path]:
    paths = sorted(method_dir.glob("seed_*.json"), key=seed_number)
    if required_budget is not None:
        paths = [path for path in paths if trace_budget(path) == required_budget]
    return paths


def aggregate_method(
    method_dir: Path,
    losses: np.ndarray,
    oracle_loss: float,
    num_items: int,
    required_budget: int | None,
) -> dict[str, float]:
    traces = [json.loads(path.read_text()) for path in trace_paths(method_dir, required_budget)]
    if not traces:
        budget_msg = "" if required_budget is None else f" with budget {required_budget}"
        raise FileNotFoundError(f"No seed_*.json traces{budget_msg} found in {method_dir}")

    per_seed = [summarize_trace(trace, losses, oracle_loss, num_items) for trace in traces]
    keys = per_seed[0].keys()
    return {key: float(sum(row[key] for row in per_seed) / len(per_seed)) for key in keys}


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


def default_methods(results_dir: Path, task: str, required_budget: int | None) -> list[str]:
    task_dir = results_dir / task
    if not task_dir.exists():
        raise FileNotFoundError(f"Missing task results directory: {task_dir}")
    methods = []
    for path in sorted(task_dir.iterdir()):
        if path.is_dir() and trace_paths(path, required_budget):
            methods.append(path.name)
    if not methods:
        raise FileNotFoundError(f"No method traces found in {task_dir}")
    return methods


def evaluate_experiment(args: argparse.Namespace, experiment: str) -> dict[str, dict[str, float]]:
    task = experiment_task(experiment, args.task)
    results_dir = result_dir_for_experiment(args, experiment)
    required_budget = 0 if experiment.endswith("_unsupervised_selection") else 100
    predictions, labels = load_benchmark(Path(args.data_dir), task)
    losses = losses_from_benchmark(predictions, labels)
    oracle_loss = float(losses.min())
    known_metrics = reference_metric_names(Path(args.reference), experiment)
    methods = args.methods if args.methods else default_methods(results_dir, task, required_budget)

    results = {}
    for method in methods:
        method_dir = results_dir / task / method
        results[method] = filter_known_metrics(
            aggregate_method(method_dir, losses, oracle_loss, labels.shape[0], required_budget),
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
    parser.add_argument("--output", default="scoring/scores.json")
    parser.add_argument("--reference", default="scoring/reference.json")
    parser.add_argument("--merge-existing", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.output)
    if args.merge_existing and out_path.exists():
        scores = json.loads(out_path.read_text())
    else:
        scores = {"experiments": {}}
    scores.setdefault("experiments", {})
    for experiment in selected_experiments(args):
        scores["experiments"][experiment] = {"results": evaluate_experiment(args, experiment)}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
