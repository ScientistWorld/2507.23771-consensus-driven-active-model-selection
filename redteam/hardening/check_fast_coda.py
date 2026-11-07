"""Check the vectorized CODA against the original per-model-loop implementation.

The reference functions below are copied verbatim from the pre-hardening
method/coda_reproduction.py (commit fe4a312). For several visible-slice
posterior states we compare P_best, the selected model and the EIG of every
candidate item. Exits non-zero if they disagree beyond float round-off.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import betainc, betaln

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from method.coda_reproduction import CODASelector  # noqa: E402


# ---- original implementation (verbatim) ---------------------------------
def entropy(prob: np.ndarray) -> float:
    prob = np.clip(prob, 1e-12, 1.0)
    return float(-(prob * np.log2(prob)).sum())


def beta_pdf_grid(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    log_pdf = (a[..., None] - 1.0) * np.log(x)
    log_pdf += (b[..., None] - 1.0) * np.log1p(-x)
    log_pdf -= betaln(a, b)[..., None]
    return np.nan_to_num(np.exp(log_pdf), nan=0.0, posinf=0.0, neginf=0.0)


def pbest_from_betas(alpha_diag: np.ndarray, beta_rest: np.ndarray, num_grid: int) -> np.ndarray:
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


def original_pbest(sel: CODASelector, num_grid: int) -> np.ndarray:
    d = sel.dirichlet
    diag = np.diagonal(d, axis1=1, axis2=2)
    rest = d.sum(axis=-1) - diag
    row = pbest_from_betas(diag, np.clip(rest, 1e-9, None), num_grid)
    pbest = (sel.pi[:, None] * row).sum(axis=0)
    return pbest / np.clip(pbest.sum(), 1e-12, None)


def original_eig(sel: CODASelector, candidates: list[int], num_grid: int) -> np.ndarray:
    diag_before = np.diagonal(sel.dirichlet, axis1=1, axis2=2)
    rest_before = sel.dirichlet.sum(axis=-1) - diag_before
    row_pbest_before = pbest_from_betas(diag_before, np.clip(rest_before, 1e-9, None), num_grid)
    current_pbest = (sel.pi[:, None] * row_pbest_before).sum(axis=0)
    current_pbest /= np.clip(current_pbest.sum(), 1e-12, None)
    before = entropy(current_pbest)
    scores = np.zeros(len(candidates), dtype=np.float64)
    for pos, item_idx in enumerate(candidates):
        after = 0.0
        for c in range(sel.c_count):
            d = sel.dirichlet.copy()
            for h, pred_c in enumerate(sel.pred_classes[:, item_idx]):
                d[h, c, pred_c] += sel.eig_update_weight
            diag = np.diagonal(d, axis1=1, axis2=2)
            rest = d.sum(axis=-1) - diag
            row_pbest = pbest_from_betas(diag, np.clip(rest, 1e-9, None), num_grid)
            p = current_pbest + sel.pi[c] * (row_pbest[c] - row_pbest_before[c])
            p /= np.clip(p.sum(), 1e-12, None)
            after += sel.pi_x[item_idx, c] * entropy(p)
        scores[pos] = before - after
    return scores
# --------------------------------------------------------------------------


def compare(task: str, labels_to_add: list[int], grid: int = 128, eta: float = 0.01) -> dict:
    data = json.loads((ROOT / "data" / "benchmarks" / f"{task}_train.json").read_text())
    sel = CODASelector(data["predicted_classes"], data["num_classes"], eta=eta, num_grid=grid, seed=0)
    rows = []
    for checkpoint in labels_to_add:
        while len(sel.labeled) < checkpoint:
            item, _ = sel.next_item()
            sel.add_label(item, data["labels"][item])
        cands = sel.candidates()
        fast_eig = sel.eig_scores(cands)
        slow_eig = original_eig(sel, cands, grid)
        slow_pbest = original_pbest(sel, grid)
        rows.append({
            "task": task, "eta": eta, "labels": len(sel.labeled), "candidates": len(cands),
            "max_abs_pbest_diff": float(np.abs(sel.pbest - slow_pbest).max()),
            "same_best_model": int(np.argmax(slow_pbest)) == sel.best_model(),
            "max_abs_eig_diff": float(np.abs(fast_eig - slow_eig).max()),
            "max_abs_eig": float(np.abs(slow_eig).max()),
            "same_argmax_set": bool(
                set(np.flatnonzero(np.isclose(fast_eig, fast_eig.max())).tolist())
                == set(np.flatnonzero(np.isclose(slow_eig, slow_eig.max())).tolist())
            ),
        })
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "redteam" / "hardening" / "fast_coda_check.json"
    rows = compare("glue_rte", [0, 5, 20]) + compare("glue_rte", [10], eta=1.0) + compare("glue_mrpc", [0, 8])
    ok = all(
        r["max_abs_pbest_diff"] < 1e-10 and r["max_abs_eig_diff"] < 1e-10 and r["same_best_model"] and r["same_argmax_set"]
        for r in rows
    )
    out.write_text(json.dumps({"ok": ok, "rows": rows}, indent=2) + "\n")
    if not ok:
        raise SystemExit("vectorized CODA disagrees with the original implementation")


if __name__ == "__main__":
    main()
