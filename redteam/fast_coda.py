"""Same Beta quadrature and EIG, vectorized for the honest update-size variants.

CDF products excluding a model use prefix/suffix products instead of repeated
array deletion. Hypothetical updates only recompute their affected class row.
No label values enter initialization, selection, or acquisition.
"""
import numpy as np
from scipy.special import betainc, betaln
from redteam.shared import CODASelector


def row_pbest(a, b, grid):
    # (..., H) -> (..., H); vectorized arbitrary hypothetical item dimensions.
    x = np.linspace(1e-5, 1.0 - 1e-5, grid, dtype=np.float64)
    pdf = np.exp((a[..., None] - 1) * np.log(x)
                 + (b[..., None] - 1) * np.log1p(-x)
                 - betaln(a, b)[..., None])
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    cdf = betainc(a[..., None], b[..., None], x)
    left = np.ones_like(cdf)
    right = np.ones_like(cdf)
    left[..., 1:, :] = np.cumprod(cdf[..., :-1, :], axis=-2)
    right[..., :-1, :] = np.cumprod(cdf[..., :0:-1, :], axis=-2)[..., ::-1, :]
    probs = np.trapezoid(pdf * left * right, x, axis=-1)
    total = probs.sum(axis=-1, keepdims=True)
    return np.where(total > 0, probs / np.clip(total, 1e-300, None), 1 / a.shape[-1])


class FastCODA(CODASelector):
    def _eig_scores(self, candidates):
        diag = np.diagonal(self.dirichlet, axis1=1, axis2=2).T
        rest = self.dirichlet.sum(axis=-1).T - diag
        rows = row_pbest(diag, np.clip(rest, 1e-9, None), self.num_grid)
        current = (self.pi[:, None] * rows).sum(axis=0)
        current /= np.clip(current.sum(), 1e-12, None)
        current_entropy = -(np.clip(current, 1e-12, 1) * np.log2(np.clip(current, 1e-12, 1))).sum()
        scores = np.full(len(candidates), current_entropy)
        # Bound memory while preserving the full disagreement candidate set.
        for start in range(0, len(candidates), 64):
            ids = candidates[start:start + 64]
            expected = np.zeros(len(ids))
            for c in range(self.c_count):
                same = self.pred_classes[:, ids].T == c
                a = diag[c] + same * self.eig_update_weight
                b = rest[c] + (~same) * self.eig_update_weight
                updated = row_pbest(a, np.clip(b, 1e-9, None), self.num_grid)
                p = current[None, :] + self.pi[c] * (updated - rows[c])
                p /= np.clip(p.sum(axis=-1, keepdims=True), 1e-12, None)
                p = np.clip(p, 1e-12, 1)
                ent = -(p * np.log2(p)).sum(axis=-1)
                expected += self.pi_x[ids, c] * ent
            scores[start:start + len(ids)] -= expected
        return scores
