"""Data-free unit tests for the statistical core and the leakage-sensitive pieces of the pipeline.

Run:  pytest -q tests
"""
import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp.baselines import KNNEmbedding, LinearEmbedding, MeanShift, knockdown_rule  # noqa: E402
from vcp.data import (bh_fdr, fit_cell_pca, group_mean_var, kfold_perturbation_split,  # noqa: E402
                      parse_condition, project_cells, train_val_split, welch_t)
from vcp.embeddings import response_embedding  # noqa: E402
from vcp.metrics import energy_distance, evaluate  # noqa: E402

rng = np.random.default_rng(0)


def test_parse_condition():
    assert parse_condition("GATA1+ctrl") == "GATA1"
    assert parse_condition("ctrl+GATA1") == "GATA1"
    assert parse_condition("ctrl") == "ctrl"
    assert parse_condition("A+B") == "A+B"


def test_group_mean_var_matches_numpy():
    X = sp.random(500, 40, density=0.3, random_state=1, format="csr")
    lab = rng.integers(-1, 4, 500)
    mean, var, cnt = group_mean_var(X, lab, 4, chunk=77)
    D = X.toarray()
    for k in range(4):
        assert cnt[k] == (lab == k).sum()
        np.testing.assert_allclose(mean[k], D[lab == k].mean(0), atol=1e-5)
        np.testing.assert_allclose(var[k], D[lab == k].var(0), atol=1e-5)


def test_welch_t_matches_scipy():
    a = rng.normal(0.3, 1.0, 60)
    b = rng.normal(0.0, 0.4, 900)
    t, p = welch_t(a.mean(), a.var(), 60, b.mean(), b.var(), 900)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert float(t) == pytest.approx(ref.statistic, rel=1e-5)
    assert float(p) == pytest.approx(ref.pvalue, rel=1e-4)


def test_welch_t_low_expression_not_inflated():
    # regression test: an additive epsilon in the df denominator used to collapse df to 1 for
    # lowly expressed genes, inflating p-values of complete knockdowns
    _, p = welch_t(np.array([0.0]), np.array([0.0]), np.array([38]), np.array([0.36]), np.array([0.3]), 10691)
    assert p[0] < 1e-20


def test_bh_fdr_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = rng.uniform(size=(3, 200)) ** 3
    ours = bh_fdr(p, axis=1)
    for i in range(3):
        np.testing.assert_allclose(ours[i], multipletests(p[i], method="fdr_bh")[1], atol=1e-12)


def test_splits_are_disjoint_and_complete():
    perts = np.array([f"G{i}" for i in range(103)])
    fold = kfold_perturbation_split(perts, 5, seed=3)
    assert set(np.bincount(fold)) <= {20, 21}
    tr, va = train_val_split(np.nonzero(fold != 0)[0], 0.1, 0)
    assert not set(tr) & set(va)
    assert not (set(tr) | set(va)) & set(np.nonzero(fold == 0)[0])


def test_response_embedding_uses_training_rows_only():
    # changing the responses of held-out perturbations must not change any embedding
    genes = [f"G{i}" for i in range(30)]
    Y = rng.normal(size=(50, 30))
    train = np.arange(40)
    E1, _ = response_embedding(genes, genes, Y[train], dim=8)
    Y2 = Y.copy()
    Y2[40:] = rng.normal(size=(10, 30)) * 100
    E2, _ = response_embedding(genes, genes, Y2[train], dim=8)
    np.testing.assert_allclose(E1, E2)


def test_cell_pca_ignores_held_out_cells():
    X = sp.random(600, 25, density=0.5, random_state=2, format="csr")
    lab = np.repeat(np.arange(6), 100)          # 0..4 perturbations, 5 = control
    ctrl_mean = np.asarray(X[lab == 5].mean(0)).ravel()
    c1, _ = fit_cell_pca(X, lab, [0, 1, 2], 5, ctrl_mean, n_pcs=5)
    X2 = X.tolil()
    X2[np.nonzero(lab == 4)[0], :] = 50.0      # perturbation 4 is held out: change it drastically
    c2, _ = fit_cell_pca(X2.tocsr(), lab, [0, 1, 2], 5, ctrl_mean, n_pcs=5)
    np.testing.assert_allclose(np.abs(c1), np.abs(c2), atol=1e-5)
    Z = project_cells(X, c1, ctrl_mean, chunk=64)
    np.testing.assert_allclose(Z, (X.toarray() - ctrl_mean) @ c1.T, atol=1e-4)


def test_metrics_sanity():
    P, G = 12, 80
    true = rng.normal(size=(P, G))
    tstat = true * 10
    fdr = np.full((P, G), 0.01)
    ref = true.mean(0)
    perfect = evaluate(true, true, tstat, fdr, ref)
    assert np.allclose(perfect["pearson_all"], 1)
    assert np.allclose(perfect["dir_acc_de20"], 1)
    assert np.allclose(perfect["rank"], 0)
    mean_shift = evaluate(np.repeat(ref[None], P, 0), true, tstat, fdr, ref)
    assert np.allclose(mean_shift["pearson_sys"], 0, atol=1e-6)   # Systema: mean shift carries no signal


def test_energy_distance():
    x = rng.normal(size=(200, 5))
    assert energy_distance(x, x) == pytest.approx(0, abs=1e-8)
    assert energy_distance(x, x + 3) > energy_distance(x, x + 1) > 0


def test_baselines_shapes_and_kd_rule():
    Y = rng.normal(size=(30, 10))
    E = rng.normal(size=(30, 6))
    for m in (MeanShift(), LinearEmbedding(k=4), KNNEmbedding(k=3)):
        assert m.fit(Y[:25], E[:25]).predict(E[25:]).shape == (5, 10)
    ctrl = np.full(10, 2.0)
    out = knockdown_rule(np.zeros((2, 10)), ["A", "B"], np.array([3, -1]), ctrl, train_frac=0.25)
    assert out[0, 3] == pytest.approx(-1.5)
    assert np.all(out[1] == 0)
