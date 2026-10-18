"""Perturb-seq data handling.

All matrices are log-normalised expression (log1p of size-factor normalised counts),
exactly as distributed by the GEARS processed datasets (Roohani et al., 2023).

The heavy single-cell matrix is never densified: per-perturbation means / variances are
accumulated chunk-wise with sparse indicator matrices so that the ~160k-cell Replogle
K562 dataset fits into a few GB of RAM.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy import stats

CTRL = "ctrl"


# ----------------------------------------------------------------------------------------
# condition parsing
# ----------------------------------------------------------------------------------------
def parse_condition(cond: str) -> str:
    """'GENE+ctrl' / 'ctrl+GENE' -> 'GENE'; 'ctrl' -> 'ctrl'; 'A+B' -> 'A+B' (combination)."""
    if cond == CTRL:
        return CTRL
    parts = [p for p in cond.split("+") if p != CTRL]
    return "+".join(parts) if parts else CTRL


# ----------------------------------------------------------------------------------------
# chunked group statistics on (backed) sparse matrices
# ----------------------------------------------------------------------------------------
def group_mean_var(X, labels: np.ndarray, n_groups: int, chunk: int = 20000):
    """Per-group mean and (population) variance of the rows of X.

    X may be an in-memory scipy sparse matrix / ndarray or an AnnData backed dataset.
    labels: int array (n_cells,) with values in [0, n_groups) or -1 (ignored).
    """
    n_cells, n_genes = X.shape
    s1 = np.zeros((n_groups, n_genes), dtype=np.float64)
    s2 = np.zeros((n_groups, n_genes), dtype=np.float64)
    cnt = np.bincount(labels[labels >= 0], minlength=n_groups).astype(np.float64)
    for start in range(0, n_cells, chunk):
        end = min(start + chunk, n_cells)
        xb = X[start:end]
        if not sp.issparse(xb):
            xb = sp.csr_matrix(np.asarray(xb))
        xb = xb.tocsr().astype(np.float64)
        lb = labels[start:end]
        keep = lb >= 0
        rows = lb[keep]
        cols = np.nonzero(keep)[0]
        ind = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_groups, end - start))
        s1 += np.asarray((ind @ xb).todense())
        xb.data **= 2
        s2 += np.asarray((ind @ xb).todense())
    safe = np.maximum(cnt, 1)[:, None]
    mean = s1 / safe
    var = np.maximum(s2 / safe - mean ** 2, 0.0)
    return mean.astype(np.float32), var.astype(np.float32), cnt.astype(np.int64)


def welch_t(m1, v1, n1, m0, v0, n0, eps: float = 1e-12):
    """Welch t statistic and two-sided p-value (Welch-Satterthwaite degrees of freedom).

    Note: no absolute epsilon may enter the df denominator - for low-expressed genes the squared
    standard errors are ~1e-10, so an additive 1e-8 would collapse df to 1 and inflate p-values.
    """
    m1, v1, m0, v0 = (np.asarray(a, dtype=np.float64) for a in (m1, v1, m0, v0))
    n1 = np.asarray(n1, dtype=np.float64)
    v1 = v1 * n1 / np.maximum(n1 - 1, 1)          # population -> unbiased sample variance
    v0 = v0 * n0 / max(n0 - 1, 1)
    se2_1 = v1 / np.maximum(n1, 1)
    se2_0 = v0 / np.maximum(n0, 1)
    se2 = se2_1 + se2_0
    t = (m1 - m0) / np.sqrt(np.maximum(se2, eps))
    t = np.where(se2 > 0, t, 0.0)
    den = se2_1 ** 2 / np.maximum(n1 - 1, 1) + se2_0 ** 2 / np.maximum(n0 - 1, 1)
    df = np.where(den > 0, se2 ** 2 / np.where(den > 0, den, 1.0), 1.0)
    p = 2 * stats.t.sf(np.abs(t), np.maximum(df, 1))
    return t.astype(np.float32), p


def bh_fdr(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values along `axis`."""
    p = np.moveaxis(np.asarray(p, dtype=np.float64), axis, -1)
    n = p.shape[-1]
    order = np.argsort(p, axis=-1)
    ranked = np.take_along_axis(p, order, -1) * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[..., ::-1], axis=-1)[..., ::-1]
    out = np.empty_like(ranked)
    np.put_along_axis(out, order, np.minimum(ranked, 1.0), -1)
    return np.moveaxis(out, -1, axis)


# ----------------------------------------------------------------------------------------
# processed pseudobulk container
# ----------------------------------------------------------------------------------------
@dataclass
class PerturbData:
    """Pseudobulk-level summary of a Perturb-seq screen."""

    name: str
    genes: np.ndarray            # (G,) measured gene symbols
    perts: np.ndarray            # (P,) perturbed gene symbols
    ctrl_mean: np.ndarray        # (G,)
    ctrl_var: np.ndarray         # (G,)
    n_ctrl: int
    pert_mean: np.ndarray        # (P, G)
    pert_var: np.ndarray         # (P, G)
    n_cells: np.ndarray          # (P,)
    tstat: np.ndarray            # (P, G) Welch t vs control
    fdr: np.ndarray              # (P, G) BH-FDR
    extra: dict = field(default_factory=dict)

    @property
    def delta(self) -> np.ndarray:
        return self.pert_mean - self.ctrl_mean[None, :]

    def top_de(self, k: int = 20) -> np.ndarray:
        """Indices (P, k) of the top-k DE genes per perturbation ranked by |t|."""
        return np.argsort(-np.abs(self.tstat), axis=1)[:, :k]

    def sig_de_mask(self, k: int = 20, alpha: float = 0.05) -> np.ndarray:
        """Boolean (P, G): gene is among top-k |t| AND FDR < alpha."""
        mask = np.zeros_like(self.tstat, dtype=bool)
        top = self.top_de(k)
        np.put_along_axis(mask, top, True, axis=1)
        return mask & (self.fdr < alpha)

    def save(self, path: str):
        np.savez_compressed(
            path, name=self.name, genes=self.genes, perts=self.perts,
            ctrl_mean=self.ctrl_mean, ctrl_var=self.ctrl_var, n_ctrl=self.n_ctrl,
            pert_mean=self.pert_mean, pert_var=self.pert_var, n_cells=self.n_cells,
            tstat=self.tstat, fdr=self.fdr.astype(np.float32),
            **{f"extra_{k}": v for k, v in self.extra.items()},
        )

    @classmethod
    def load(cls, path: str) -> "PerturbData":
        z = np.load(path, allow_pickle=True)
        extra = {k[6:]: z[k] for k in z.files if k.startswith("extra_")}
        return cls(
            name=str(z["name"]), genes=z["genes"], perts=z["perts"],
            ctrl_mean=z["ctrl_mean"], ctrl_var=z["ctrl_var"], n_ctrl=int(z["n_ctrl"]),
            pert_mean=z["pert_mean"], pert_var=z["pert_var"], n_cells=z["n_cells"],
            tstat=z["tstat"], fdr=z["fdr"], extra=extra,
        )


# ----------------------------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------------------------
def kfold_perturbation_split(perts: np.ndarray, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Assign every perturbation to one of `n_folds` folds (unseen-perturbation CV)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(perts))
    fold = np.empty(len(perts), dtype=np.int64)
    fold[order] = np.arange(len(perts)) % n_folds
    return fold


def train_val_split(train_idx: np.ndarray, val_frac: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed + 1000)
    idx = rng.permutation(train_idx)
    n_val = max(1, int(round(len(idx) * val_frac)))
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ----------------------------------------------------------------------------------------
# single-cell latent space (PCA)
# ----------------------------------------------------------------------------------------
def cell_labels(obs_condition, perts) -> np.ndarray:
    """Integer label per cell: index into `perts`, len(perts) for control, -1 for anything else."""
    cond = np.asarray([parse_condition(c) for c in np.asarray(obs_condition).astype(str)])
    lut = {p: i for i, p in enumerate(perts)}
    ctrl = len(perts)
    return np.array([lut.get(c, ctrl if c == CTRL else -1) for c in cond], dtype=np.int64)


def fit_cell_pca(X, labels: np.ndarray, fit_labels, ctrl_label: int, ctrl_mean: np.ndarray,
                 n_pcs: int = 100, per_pert: int = 40, n_ctrl: int = 5000, seed: int = 0,
                 chunk: int = 20000):
    """PCA of single cells (centred on the control mean), fitted ONLY on control cells and on
    cells of the perturbations in `fit_labels` - pass the training perturbations of a CV fold so
    that held-out perturbations never shape the latent space."""
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)
    chosen = []
    for k in list(fit_labels) + [ctrl_label]:
        idx = np.nonzero(labels == k)[0]
        m = n_ctrl if k == ctrl_label else per_pert
        chosen.append(rng.choice(idx, size=min(m, len(idx)), replace=False))
    chosen = np.sort(np.concatenate(chosen))
    rows = []
    for start in range(0, X.shape[0], chunk):
        end = min(start + chunk, X.shape[0])
        sel = chosen[(chosen >= start) & (chosen < end)] - start
        if len(sel):
            rows.append(sp.csr_matrix(X[start:end])[sel].toarray())
    Xs = np.vstack(rows).astype(np.float32) - ctrl_mean[None]
    pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=seed).fit(Xs)
    return pca.components_.astype(np.float32), pca.explained_variance_ratio_.astype(np.float32)


def project_cells(X, comps: np.ndarray, ctrl_mean: np.ndarray, chunk: int = 20000) -> np.ndarray:
    """Latent coordinates (X - ctrl_mean) @ comps.T for every row of a (backed) sparse matrix."""
    Z = np.zeros((X.shape[0], comps.shape[0]), dtype=np.float32)
    offset = (ctrl_mean[None] @ comps.T).astype(np.float32)
    for start in range(0, X.shape[0], chunk):
        end = min(start + chunk, X.shape[0])
        Z[start:end] = np.asarray(sp.csr_matrix(X[start:end]) @ comps.T) - offset
    return Z
