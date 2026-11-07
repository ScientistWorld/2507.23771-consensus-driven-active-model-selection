"""CODA: consensus-driven active model selection (Kay et al., 2025).

Implements the paper's computational steps as an oracle-protocol selector
(see data/selection_protocol.py):

* Sec. 4.2  consensus labels from summed candidate predictions, per-model
            empirical confusion matrices against them, and Dirichlet priors
            theta = (beta + alpha * M_hat) / T with the diagonal-weighted beta
* Sec. 4.3  P_best: per confusion-matrix row, the diagonal entry is
            Beta(alpha_kc, beta_kc); P(model k is best | row c) is the
            integral of f_k(x) * prod_{l != k} F_l(x), approximated with the
            trapezoid rule on a grid; rows are mixed with the class marginal
            pi_hat (Eq. 10)
* Sec. 4.4  expected information gain of P_best for every unlabeled item
            whose candidates disagree (as in the released code), using a unit
            hypothetical observation, weighted by pi_hat(c | x_i)
* Sec. 4.5  partial Dirichlet-categorical update with learning rate eta

Defaults are the paper's fixed values alpha=0.1, T=0.5, eta=0.01.

The EIG step is vectorized: a hypothetical label c changes only row c, and
in that row each model's Beta gains either +1 on the diagonal (it predicted
c) or +1 off the diagonal, so the two possible PDFs/CDFs per model are
computed once and selected per item; products over the other models use
prefix/suffix cumulative products instead of one deletion per model.
redteam/hardening/check_fast_coda.py checks this against the original
per-model loop.

Usage (as registered in outputs/selectors/coda.json):
    python method/coda_reproduction.py [--eta 0.01] [--grid 128]
Optional label-saving variant: --stable-rounds R --min-labels M stops
querying once the selection has been unchanged for R rounds after >= M labels.
--acquisition uncertainty is the paper's Fig. 6 ablation (committee-entropy
queries with CODA's posterior selection).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from scipy.special import betainc, betaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.selection_protocol import serve  # noqa: E402


def beta_grid(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PDF and CDF of Beta(a, b) on grid x; output shape a.shape + (G,)."""
    log_pdf = (a[..., None] - 1.0) * np.log(x)
    log_pdf = log_pdf + (b[..., None] - 1.0) * np.log1p(-x)
    log_pdf = log_pdf - betaln(a, b)[..., None]
    pdf = np.nan_to_num(np.exp(log_pdf), nan=0.0, posinf=0.0, neginf=0.0)
    cdf = betainc(a[..., None], b[..., None], x)
    return pdf, cdf


def product_of_others(cdf: np.ndarray) -> np.ndarray:
    """prod_{l != k} cdf[..., l, :] for every k (model axis is -2)."""
    left = np.ones_like(cdf)
    right = np.ones_like(cdf)
    left[..., 1:, :] = np.cumprod(cdf[..., :-1, :], axis=-2)
    right[..., :-1, :] = np.cumprod(cdf[..., :0:-1, :], axis=-2)[..., ::-1, :]
    return left * right


def row_pbest(pdf: np.ndarray, cdf: np.ndarray, x: np.ndarray) -> np.ndarray:
    """P(model k has the largest draw) per row; (..., H, G) -> (..., H)."""
    probs = np.trapezoid(pdf * product_of_others(cdf), x, axis=-1)
    total = probs.sum(axis=-1, keepdims=True)
    uniform = np.full_like(probs, 1.0 / probs.shape[-1])
    return np.where(total > 0, probs / np.where(total > 0, total, 1.0), uniform)


def entropy_bits(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log2(q)).sum(axis=-1)


class CODASelector:
    def __init__(
        self,
        pred_classes,
        num_classes: int,
        alpha: float = 0.1,
        temperature: float = 0.5,
        eta: float = 0.01,
        eig_update_weight: float = 1.0,
        num_grid: int = 128,
        seed: int = 0,
        chunk: int = 48,
        acquisition: str = "eig",
    ) -> None:
        self.pred_classes = np.asarray(pred_classes, dtype=np.int64)  # (H, N)
        self.h_count, self.n_count = self.pred_classes.shape
        self.c_count = int(num_classes)
        self.eta = float(eta)
        self.eig_update_weight = float(eig_update_weight)
        self.chunk = int(chunk)
        if acquisition not in ("eig", "uncertainty"):
            raise ValueError(f"unknown acquisition {acquisition}")
        self.acquisition = acquisition
        self.rng = random.Random(seed)
        self.x = np.linspace(1e-5, 1.0 - 1e-5, int(num_grid), dtype=np.float64)
        self.onehot = np.eye(self.c_count, dtype=np.float64)[self.pred_classes]  # (H, N, C)

        consensus = self.onehot.sum(axis=0).argmax(axis=-1)
        empirical = np.zeros((self.h_count, self.c_count, self.c_count), dtype=np.float64)
        for c in range(self.c_count):
            mask = consensus == c
            if mask.any():
                empirical[:, c, :] = self.onehot[:, mask, :].sum(axis=1)
                empirical[:, c, :] /= np.clip(empirical[:, c, :].sum(axis=1, keepdims=True), 1e-12, None)
        beta = np.full((self.c_count, self.c_count), 1.0 / max(self.c_count - 1, 1), dtype=np.float64)
        np.fill_diagonal(beta, 1.0)
        self.dirichlet = (beta[None, :, :] + float(alpha) * empirical) / float(temperature)

        mean_pred = self.onehot.mean(axis=0)  # (N, C) committee vote shares
        self.committee_entropy = -(mean_pred * np.log(mean_pred + 1e-8)).sum(axis=-1)
        majority = self.onehot.sum(axis=0).argmax(axis=-1)
        self.disagreement = (self.pred_classes != majority[None, :]).any(axis=0)
        self.unlabeled: list[int] = list(range(self.n_count))
        self.labeled: list[int] = []
        self._eig_cache: dict[int, float] | None = None
        self._refresh()

    def _refresh(self) -> None:
        adjusted = np.einsum("hcs,hns->hnc", self.dirichlet, self.onehot)
        pi_x = adjusted.sum(axis=0)
        self.pi_x = pi_x / np.clip(pi_x.sum(axis=-1, keepdims=True), 1e-12, None)
        pi = self.pi_x.mean(axis=0)
        self.pi = pi / np.clip(pi.sum(), 1e-12, None)
        diag = np.diagonal(self.dirichlet, axis1=1, axis2=2)  # (H, C)
        self.a = diag.T.copy()  # (C, H)
        self.b = (self.dirichlet.sum(axis=-1) - diag).T.copy()
        pdf, cdf = beta_grid(self.a, np.clip(self.b, 1e-9, None), self.x)
        self.rows = row_pbest(pdf, cdf, self.x)  # (C, H)
        pbest = (self.pi[:, None] * self.rows).sum(axis=0)
        self.pbest = pbest / np.clip(pbest.sum(), 1e-12, None)

    def best_model(self) -> int:
        return int(np.argmax(self.pbest))

    def eig_scores(self, candidates: list[int]) -> np.ndarray:
        cand = np.asarray(candidates, dtype=np.int64)
        before = float(entropy_bits(self.pbest))
        expected = np.zeros(len(cand), dtype=np.float64)
        w = self.eig_update_weight
        for c in range(self.c_count):
            a, b = self.a[c], self.b[c]
            pdf_same, cdf_same = beta_grid(a + w, np.clip(b, 1e-9, None), self.x)
            pdf_diff, cdf_diff = beta_grid(a, np.clip(b + w, 1e-9, None), self.x)
            for start in range(0, len(cand), self.chunk):
                ids = cand[start:start + self.chunk]
                same = (self.pred_classes[:, ids].T == c)[:, :, None]  # (n, H, 1)
                updated = row_pbest(
                    np.where(same, pdf_same, pdf_diff),
                    np.where(same, cdf_same, cdf_diff),
                    self.x,
                )
                p = self.pbest[None, :] + self.pi[c] * (updated - self.rows[c][None, :])
                p = p / np.clip(p.sum(axis=-1, keepdims=True), 1e-12, None)
                expected[start:start + len(ids)] += self.pi_x[ids, c] * entropy_bits(p)
        return before - expected

    def candidates(self) -> list[int]:
        cands = [i for i in self.unlabeled if self.disagreement[i]]
        return cands if cands else list(self.unlabeled)

    def next_item(self) -> tuple[int, float]:
        cands = self.candidates()
        if self.acquisition == "uncertainty":
            # Ablation of Fig. 6: committee-entropy acquisition inside CODA.
            q = self.committee_entropy[cands]
        elif self.eta == 0.0:
            # Beliefs never change, so each item's EIG is fixed: compute once.
            if self._eig_cache is None:
                self._eig_cache = dict(zip(cands, self.eig_scores(cands).tolist()))
            q = np.array([self._eig_cache[i] for i in cands])
        else:
            q = self.eig_scores(cands)
        ties = np.flatnonzero(np.isclose(q, q.max()))
        local = int(self.rng.choice(ties))
        return cands[local], float(q[local])

    def add_label(self, item: int, label: int) -> None:
        if not 0 <= label < self.c_count:
            raise ValueError(f"label {label} outside {self.c_count} classes")
        self.dirichlet[np.arange(self.h_count), label, self.pred_classes[:, item]] += self.eta
        self.unlabeled.remove(item)
        self.labeled.append(item)
        self._refresh()


class CODAPolicy:
    """Oracle-protocol wrapper: one CODASelector per episode."""

    def __init__(self, episode: dict, args: argparse.Namespace) -> None:
        self.coda = CODASelector(
            episode["predicted_classes"], int(episode["num_classes"]),
            alpha=args.alpha, temperature=args.temperature, eta=args.eta,
            num_grid=args.grid, seed=int(episode["seed"]), acquisition=args.acquisition,
        )
        self.stable_rounds = args.stable_rounds
        self.min_labels = args.min_labels
        self.history: list[int] = []

    def select(self) -> int:
        choice = self.coda.best_model()
        self.history.append(choice)
        return choice

    def query(self) -> int | None:
        if not self.coda.unlabeled:
            return None
        if (
            self.stable_rounds > 0
            and len(self.coda.labeled) >= self.min_labels
            and len(self.history) >= self.stable_rounds
            and len(set(self.history[-self.stable_rounds:])) == 1
        ):
            return None
        item, _ = self.coda.next_item()
        return item

    def observe(self, item: int, label: int) -> None:
        self.coda.add_label(item, label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--stable-rounds", type=int, default=0)
    parser.add_argument("--min-labels", type=int, default=0)
    parser.add_argument("--acquisition", choices=["eig", "uncertainty"], default="eig")
    args = parser.parse_args()
    serve(lambda episode: CODAPolicy(episode, args))


if __name__ == "__main__":
    main()
