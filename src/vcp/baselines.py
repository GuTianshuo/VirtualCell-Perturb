"""Simple but strong baselines for unseen-perturbation prediction.

Recent benchmarks (Ahlmann-Eltze et al. Nat Methods 2025; Csendes et al. 2025;
Vinas Torne et al. Nat Biotech 2025) showed that deep models often fail to beat
  * the mean of training perturbations ("mean shift"), and
  * a linear model on gene / perturbation embeddings.
Any proposed model therefore has to be compared against these.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import normalize


class NoChange:
    name = "No-change"

    def fit(self, Y, E=None):
        self.G = Y.shape[1]
        return self

    def predict(self, E):
        return np.zeros((len(E), self.G), dtype=np.float32)


class MeanShift:
    name = "Mean-shift"

    def fit(self, Y, E=None):
        self.mu = Y.mean(0)
        return self

    def predict(self, E):
        return np.repeat(self.mu[None], len(E), 0)


class LinearEmbedding:
    """Y ≈ b + G W p   (Ahlmann-Eltze et al. 2025).

    G: (genes x K) top-K principal directions of the training response matrix
    p: perturbation embedding (any prior-knowledge encoding)
    W: closed-form double ridge solution.
    """

    def __init__(self, k: int = 32, lam: float = 0.1, name: str = "Linear"):
        self.k, self.lam, self.name = k, lam, name

    def fit(self, Y, E):
        self.b = Y.mean(0)
        Yc = (Y - self.b).T                               # (G, P)
        U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
        k = min(self.k, len(S))
        self.Gm = U[:, :k] * S[:k]                        # (G, k)
        P = E                                             # (P, d)
        GtG = self.Gm.T @ self.Gm
        PtP = P.T @ P
        lhs = np.linalg.solve(GtG + self.lam * np.eye(k) * np.trace(GtG) / k, self.Gm.T @ Yc @ P)
        self.W = np.linalg.solve((PtP + self.lam * np.eye(P.shape[1]) * max(np.trace(PtP) / P.shape[1], 1e-8)).T, lhs.T).T
        return self

    def predict(self, E):
        return self.b[None] + (self.Gm @ self.W @ E.T).T


class KNNEmbedding:
    """Similarity-weighted average of the K most similar training perturbations."""

    def __init__(self, k: int = 10, tau: float = 0.1, name: str = "kNN"):
        self.k, self.tau, self.name = k, tau, name

    def fit(self, Y, E):
        self.Y, self.E = Y, normalize(E)
        self.mu = Y.mean(0)
        return self

    def neighbours(self, E):
        S = normalize(E) @ self.E.T
        nn = np.argsort(-S, 1)[:, : self.k]
        return nn, np.take_along_axis(S, nn, 1)

    def predict(self, E):
        nn, s = self.neighbours(E)
        w = np.exp(s / self.tau)
        w /= w.sum(1, keepdims=True)
        pred = np.einsum("pk,pkg->pg", w, self.Y[nn])
        empty = np.abs(E).sum(1) == 0
        pred[empty] = self.mu
        return pred


def knockdown_rule(pred, perts, target_idx, ctrl_mean, train_frac: float):
    """CRISPRi prior: the targeted gene itself is repressed.

    pred_delta[target] = ctrl_mean[target] * (train_frac - 1), where train_frac is the median
    (perturbed mean / control mean) of the target gene over training perturbations.
    """
    pred = pred.copy()
    for i, t in enumerate(target_idx):
        if t >= 0:
            pred[i, t] = ctrl_mean[t] * (train_frac - 1.0)
    return pred


def knockdown_fraction(pert_mean, ctrl_mean, target_idx):
    m = target_idx >= 0
    t = target_idx[m]
    frac = pert_mean[np.nonzero(m)[0], t] / np.maximum(ctrl_mean[t], 1e-6)
    return float(np.median(frac)) if m.any() else 1.0
