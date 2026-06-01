"""Reproduction implementation of Consensus-Driven Active Model Selection.

This module implements the computational steps described in the paper:
consensus confusion priors, Beta-mixture probability of being best, expected
information gain acquisition, and Dirichlet-categorical updates.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import betainc, betaln

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.benchmark_io import PredictionBenchmark, load_prediction_benchmark


def entropy(prob: np.ndarray) -> float:
    prob = np.clip(prob, 1e-12, 1.0)
    return float(-(prob * np.log2(prob)).sum())


def beta_pdf_grid(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    log_pdf = (a[..., None] - 1.0) * np.log(x)
    log_pdf += (b[..., None] - 1.0) * np.log1p(-x)
    log_pdf -= betaln(a, b)[..., None]
    return np.exp(np.clip(log_pdf, -80.0, 80.0))


def pbest_from_betas(alpha_diag: np.ndarray, beta_rest: np.ndarray, num_grid: int) -> np.ndarray:
    """Approximate P(model is best) for each row-specific Beta distribution.

    alpha_diag and beta_rest have shape (H, C). The return value has shape
    (C, H), where each class row has a distribution over the best model.
    """

    h_count, c_count = alpha_diag.shape
    x = np.linspace(1e-5, 1.0 - 1e-5, num_grid, dtype=np.float64)
    out = np.zeros((c_count, h_count), dtype=np.float64)
    for c in range(c_count):
        a = alpha_diag[:, c]
        b = beta_rest[:, c]
        pdf = beta_pdf_grid(a, b, x)
        cdf = betainc(a[:, None], b[:, None], x[None, :])
        for h in range(h_count):
            prod = np.prod(np.delete(cdf, h, axis=0), axis=0)
            out[c, h] = np.trapezoid(pdf[h] * prod, x)
        total = out[c].sum()
        if total > 0:
            out[c] /= total
        else:
            out[c] = 1.0 / h_count
    return out


@dataclass
class CODASelector:
    benchmark: PredictionBenchmark
    alpha: float = 0.1
    temperature: float = 0.5
    eta: float = 0.01
    num_grid: int = 128
    acquisition: str = "eig"

    def __post_init__(self) -> None:
        self.predictions = self.benchmark.predictions
        self.pred_classes = self.benchmark.pred_classes
        self.h_count, self.n_count, self.c_count = self.predictions.shape
        consensus = self.predictions.sum(axis=0).argmax(axis=-1)
        empirical = np.zeros((self.h_count, self.c_count, self.c_count), dtype=np.float64)
        for c in range(self.c_count):
            mask = consensus == c
            if mask.any():
                empirical[:, c, :] = self.predictions[:, mask, :].sum(axis=1)
                empirical[:, c, :] /= np.clip(empirical[:, c, :].sum(axis=1, keepdims=True), 1e-12, None)

        beta = np.full((self.c_count, self.c_count), 1.0 / max(self.c_count - 1, 1), dtype=np.float64)
        np.fill_diagonal(beta, 1.0)
        # Eq. (6): blend empirical consensus and static diagonal prior; T controls pseudo-count scale.
        self.dirichlet = (beta[None, :, :] + self.alpha * empirical) / self.temperature
        self.unlabeled = list(range(self.n_count))
        self.labeled: list[int] = []
        self.observed_labels: list[int] = []
        self._update_class_beliefs()

    def _update_class_beliefs(self) -> None:
        adjusted = np.einsum("hcs,hns->hnc", self.dirichlet, self.predictions)
        self.pi_x = adjusted.sum(axis=0)
        self.pi_x /= np.clip(self.pi_x.sum(axis=-1, keepdims=True), 1e-12, None)
        self.pi = self.pi_x.mean(axis=0)
        self.pi /= np.clip(self.pi.sum(), 1e-12, None)

    def _pbest(self, dirichlet: np.ndarray | None = None) -> np.ndarray:
        d = self.dirichlet if dirichlet is None else dirichlet
        diag = np.diagonal(d, axis1=1, axis2=2)
        rest = d.sum(axis=-1) - diag
        row_pbest = pbest_from_betas(diag, np.clip(rest, 1e-9, None), self.num_grid)
        pbest = (self.pi[:, None] * row_pbest).sum(axis=0)
        pbest /= np.clip(pbest.sum(), 1e-12, None)
        return pbest

    def best_model(self) -> int:
        return int(self._pbest().argmax())

    def _eig_scores(self, candidates: list[int]) -> np.ndarray:
        diag_before = np.diagonal(self.dirichlet, axis1=1, axis2=2)
        rest_before = self.dirichlet.sum(axis=-1) - diag_before
        row_pbest_before = pbest_from_betas(diag_before, np.clip(rest_before, 1e-9, None), self.num_grid)
        current_pbest = (self.pi[:, None] * row_pbest_before).sum(axis=0)
        current_pbest /= np.clip(current_pbest.sum(), 1e-12, None)
        before = entropy(current_pbest)
        scores = np.zeros(len(candidates), dtype=np.float64)
        for pos, item_idx in enumerate(candidates):
            after = 0.0
            for c in range(self.c_count):
                d = self.dirichlet.copy()
                for h, pred_c in enumerate(self.pred_classes[:, item_idx]):
                    d[h, c, pred_c] += self.eta
                diag = np.diagonal(d, axis1=1, axis2=2)
                rest = d.sum(axis=-1) - diag
                row_pbest = pbest_from_betas(diag, np.clip(rest, 1e-9, None), self.num_grid)
                # Paper Eq. (17): the hypothetical true class changes one row's
                # contribution to the mixture over row-wise P(best).
                p = current_pbest + self.pi[c] * (row_pbest[c] - row_pbest_before[c])
                p /= np.clip(p.sum(), 1e-12, None)
                after += self.pi_x[item_idx, c] * entropy(p)
            scores[pos] = before - after
        return scores

    def next_item(self) -> tuple[int, float]:
        if self.acquisition == "eig":
            candidates = self.unlabeled
            q_values = self._eig_scores(candidates)
        elif self.acquisition == "uncertainty":
            candidates = self.unlabeled
            mean_pred = self.predictions.mean(axis=0)
            q_values = -(mean_pred[candidates] * np.log(np.clip(mean_pred[candidates], 1e-12, 1.0))).sum(axis=1)
        elif self.acquisition == "random":
            idx = random.choice(self.unlabeled)
            return idx, 1.0 / len(self.unlabeled)
        else:
            raise ValueError(f"Unknown acquisition: {self.acquisition}")
        max_q = q_values.max()
        ties = np.flatnonzero(np.isclose(q_values, max_q))
        local = int(random.choice(ties))
        return int(candidates[local]), float(q_values[local])

    def add_label(self, idx: int, true_class: int) -> None:
        for h, pred_c in enumerate(self.pred_classes[:, idx]):
            self.dirichlet[h, true_class, pred_c] += self.eta
        self.unlabeled.remove(idx)
        self.labeled.append(idx)
        self.observed_labels.append(true_class)
        self._update_class_beliefs()


class EmpiricalRiskSelector:
    def __init__(self, benchmark: PredictionBenchmark, acquisition: str, seed: int):
        self.benchmark = benchmark
        self.predictions = benchmark.predictions
        self.pred_classes = benchmark.pred_classes
        self.h_count, self.n_count, _ = self.predictions.shape
        self.acquisition = acquisition
        self.rng = random.Random(seed)
        self.unlabeled = list(range(self.n_count))
        self.labeled: list[int] = []
        self.observed_labels: list[int] = []

    def best_model(self) -> int:
        if not self.labeled:
            return 0
        labels = np.asarray(self.observed_labels)
        preds = self.pred_classes[:, self.labeled]
        losses = 1.0 - (preds == labels[None, :]).mean(axis=1)
        return int(losses.argmin())

    def next_item(self) -> tuple[int, float]:
        if self.acquisition == "random":
            idx = self.rng.choice(self.unlabeled)
            return idx, 1.0 / len(self.unlabeled)
        if self.acquisition == "uncertainty":
            mean_pred = self.predictions.mean(axis=0)
            ent = -(mean_pred[self.unlabeled] * np.log(np.clip(mean_pred[self.unlabeled], 1e-12, 1.0))).sum(axis=1)
            local = int(ent.argmax())
            return int(self.unlabeled[local]), float(ent[local])
        raise ValueError(self.acquisition)

    def add_label(self, idx: int, true_class: int) -> None:
        self.unlabeled.remove(idx)
        self.labeled.append(idx)
        self.observed_labels.append(true_class)


def run_selector(benchmark: PredictionBenchmark, method: str, budget: int, seed: int, coda_grid: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    if method == "coda":
        selector = CODASelector(benchmark, num_grid=coda_grid)
    elif method == "coda_uncertainty":
        selector = CODASelector(benchmark, num_grid=coda_grid, acquisition="uncertainty")
    elif method in {"random", "uncertainty"}:
        selector = EmpiricalRiskSelector(benchmark, method, seed)
    else:
        raise ValueError(method)

    true_losses = benchmark.true_losses
    best_loss = float(true_losses.min())
    best_true_model = int(true_losses.argmin())
    steps = []

    for t in range(budget + 1):
        pred_model = selector.best_model()
        regret = float((true_losses[pred_model] - best_loss) * 100.0)
        steps.append(
            {
                "step": t,
                "chosen_model": pred_model,
                "regret": regret,
                "is_best": pred_model == best_true_model,
            }
        )
        if t == budget:
            break
        item_idx, q = selector.next_item()
        selector.add_label(item_idx, int(benchmark.labels[item_idx]))
        steps[-1]["queried_item"] = item_idx
        steps[-1]["query_score"] = q

    return {
        "task": benchmark.name,
        "method": method,
        "seed": seed,
        "budget": budget,
        "num_models": benchmark.predictions.shape[0],
        "num_items": benchmark.predictions.shape[1],
        "num_classes": benchmark.predictions.shape[2],
        "true_best_model": best_true_model,
        "true_best_accuracy": float(benchmark.true_accuracies.max()),
        "model_accuracies": benchmark.true_accuracies.tolist(),
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/downloads/coda_data")
    parser.add_argument("--task", default="cifar10_5592")
    parser.add_argument("--method", default="coda")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--coda-grid", type=int, default=96)
    parser.add_argument("--output-dir", default="outputs/coda")
    args = parser.parse_args()

    benchmark = load_prediction_benchmark(args.data_dir, args.task)
    out_dir = Path(args.output_dir) / args.task / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seeds):
        result = run_selector(benchmark, args.method, args.budget, seed, args.coda_grid)
        (out_dir / f"seed_{seed}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
