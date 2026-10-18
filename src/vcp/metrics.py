"""Evaluation metrics for perturbation-response prediction.

Pseudobulk metrics (one value per test perturbation, then averaged):
  pearson_all       Pearson(pred delta, true delta) over all genes           (GEARS)
  pearson_de20      same, restricted to the top-20 DE genes (by |t|)           (GEARS)
  mse_de20          MSE of predicted expression on the top-20 DE genes          (GEARS)
  dir_acc_de20      fraction of significant top-20 DE genes whose up/down direction is right
  dir_acc_de100     same for the top-100 DE genes
  pearson_sys       Pearson after subtracting the mean *perturbed* profile (Systema,
                    Vinas Torne et al. 2025) -> removes systematic perturbed-vs-control
                    shifts, rewards perturbation-specific signal only.  Mean-shift baseline = 0
  de_overlap20      |top-20 predicted DEGs  ∩  top-20 true DEGs| / 20
  rank              normalised rank of the true perturbation among all test perturbations
                    when matching predictions to observations (0 = perfect, 0.5 = random)

Single-cell metrics (flow model): energy distance and RBF-MMD in PCA space.
"""
from __future__ import annotations

import numpy as np


def _pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    num = (a * b).sum(1)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-12
    return num / den


def evaluate(pred_delta: np.ndarray, true_delta: np.ndarray, tstat: np.ndarray, fdr: np.ndarray,
             ref_delta: np.ndarray, ctrl_std: np.ndarray | None = None,
             alpha: float = 0.05) -> dict:
    """All arrays are (P_test, G); ref_delta (G,) = mean training perturbation delta."""
    P, G = true_delta.shape
    order = np.argsort(-np.abs(tstat), axis=1)
    top20, top100 = order[:, :20], order[:, :100]
    take = lambda m, idx: np.take_along_axis(m, idx, 1)  # noqa: E731

    res = {}
    res["pearson_all"] = _pearson_rows(pred_delta, true_delta)
    res["pearson_de20"] = _pearson_rows(take(pred_delta, top20), take(true_delta, top20))
    res["mse_de20"] = ((take(pred_delta, top20) - take(true_delta, top20)) ** 2).mean(1)
    res["mse_all"] = ((pred_delta - true_delta) ** 2).mean(1)
    res["pearson_sys"] = _pearson_rows(pred_delta - ref_delta[None], true_delta - ref_delta[None])

    for name, idx in (("dir_acc_de20", top20), ("dir_acc_de100", top100)):
        sig = take(fdr, idx) < alpha
        agree = np.sign(take(pred_delta, idx)) == np.sign(take(true_delta, idx))
        n = sig.sum(1)
        acc = np.where(n > 0, (agree & sig).sum(1) / np.maximum(n, 1), np.nan)
        res[name] = acc

    # DE-gene recovery: rank predicted genes by |pred delta| / ctrl std (z-like score)
    scale = ctrl_std[None] + 0.05 if ctrl_std is not None else 1.0
    pred_top = np.argsort(-np.abs(pred_delta / scale), axis=1)[:, :20]
    res["de_overlap20"] = np.array([len(set(a) & set(b)) / 20.0 for a, b in zip(pred_top, top20)])

    # perturbation discrimination (distance on the union of DE genes)
    de_union = np.unique(top20)
    pp, tt = pred_delta[:, de_union], true_delta[:, de_union]
    d = (pp ** 2).sum(1)[:, None] + (tt ** 2).sum(1)[None, :] - 2 * pp @ tt.T   # (pred i, true j)
    rank = (d < d[np.arange(P), np.arange(P)][:, None]).sum(1) / max(P - 1, 1)
    res["rank"] = rank
    return res


def summarise(per_pert: dict) -> dict:
    return {k: float(np.nanmean(v)) for k, v in per_pert.items()}


# ----------------------------------------------------------------------------------------
# distribution metrics for single-cell generation
# ----------------------------------------------------------------------------------------
def _pdist_mean(a, b):
    aa = (a * a).sum(1)[:, None]
    bb = (b * b).sum(1)[None, :]
    d2 = np.maximum(aa + bb - 2 * a @ b.T, 0)
    return np.sqrt(d2).mean()


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """E-statistic 2E|X-Y| - E|X-X'| - E|Y-Y'| (scPerturb, Peidli et al. 2024)."""
    return float(2 * _pdist_mean(x, y) - _pdist_mean(x, x) - _pdist_mean(y, y))


def mmd_rbf(x: np.ndarray, y: np.ndarray, gammas=(0.5, 0.1, 0.05, 0.01, 0.005)) -> float:
    def k(a, b, g):
        aa = (a * a).sum(1)[:, None]
        bb = (b * b).sum(1)[None, :]
        return np.exp(-g * np.maximum(aa + bb - 2 * a @ b.T, 0))
    tot = 0.0
    for g in gammas:
        tot += k(x, x, g).mean() + k(y, y, g).mean() - 2 * k(x, y, g).mean()
    return float(tot / len(gammas))


def mmd_median(x: np.ndarray, y: np.ndarray) -> float:
    """RBF-MMD² with bandwidths at {0.5, 1, 2} x the median pairwise squared distance of y."""
    def sq(a, b):
        return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T, 0)
    dyy = sq(y, y)
    med = np.median(dyy[np.triu_indices(len(y), 1)])
    dxx, dxy = sq(x, x), sq(x, y)
    tot = 0.0
    for m in (0.5, 1.0, 2.0):
        g = 1.0 / (m * med)
        tot += np.exp(-g * dxx).mean() + np.exp(-g * dyy).mean() - 2 * np.exp(-g * dxy).mean()
    return float(tot / 3)
