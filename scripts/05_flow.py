"""Step 5 - single-cell generative model: conditional flow matching (advanced version).

For every CV fold (same unseen-perturbation folds as 03_benchmark.py):
  1. train PerturbFormer on the training perturbations
  2. conditions: PerturbFormer predictions projected onto the PCA basis
       - training perturbations: leave-self-out predictions (no label leakage into the condition)
       - test perturbations: ordinary predictions (gene never seen)
  3. train the OT-CFM velocity field on single cells of the training perturbations
  4. for each unseen test perturbation, push 300 control cells through the flow and compare
     the generated population with the real perturbed cells.

Leakage control: the PCA latent space is re-fitted in every fold on control cells and cells of
the TRAINING perturbations only, so held-out perturbations never shape the coordinate system.

Single-cell baselines (same 300 control cells as input):
  No-change            control cells as they are
  Mean-shift           control + mean training shift
  PerturbFormer-shift  control + PerturbFormer predicted shift (mean only, no distribution change)
  Oracle-shift         control + TRUE mean shift of the test perturbation (reference: best any
                       pure mean-shift model could do)
  CFM (ours)           conditional flow matching

Metrics: energy distance, RBF-MMD (median heuristic), per-PC log-variance error, and the
pseudobulk metrics of the decoded generated cells.
Outputs: results/{dataset}/flow_*.csv, results/{dataset}/flow_fold*.pt
"""
import argparse
import os
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp.baselines import knockdown_fraction, knockdown_rule  # noqa: E402
from vcp.data import (cell_labels, ensure_dir, fit_cell_pca, kfold_perturbation_split,  # noqa: E402
                      project_cells, train_val_split)
from vcp.flow import FlowModel, decode  # noqa: E402
from vcp.metrics import energy_distance, evaluate, mmd_median, summarise  # noqa: E402
from vcp.models import predict_perturbformer, predict_perturbformer_loo, train_perturbformer  # noqa: E402
from vcp.pipeline import PROC, ROOT, load_bundle  # noqa: E402

torch.set_num_threads(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="replogle_k562_essential")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--n_gen", type=int, default=300)
    ap.add_argument("--n_ens", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--var_cond", action="store_true",
                    help="also condition the flow on the per-PC variance profile of the 5 most similar measured perturbations")
    ap.add_argument("--dev", action="store_true", help="development mode: inner validation split of fold 0 only")
    args = ap.parse_args()

    t0 = time.time()
    b = load_bundle(args.dataset)
    cells = np.load(os.path.join(PROC, f"{args.dataset}_cells.npz"), allow_pickle=True)
    lab, ctrl_label, ctrl_mean = cells["label"], int(cells["ctrl_label"]), cells["ctrl_mean"]
    adata = ad.read_h5ad(os.path.join(ROOT, "data", "raw", args.dataset, "perturb_processed.h5ad"), backed="r")
    lab_full = cell_labels(adata.obs["condition"].values, b.pdat.perts)
    keep = lab_full >= 0
    assert np.array_equal(lab_full[keep], lab), "cell labels out of sync with 01_preprocess output"
    res_dir = ensure_dir(os.path.join(ROOT, "results", args.dataset))
    prefix = ("dev_" + ("varcond_" if args.var_cond else "meancond_")) if args.dev else ""
    outp = lambda name: os.path.join(res_dir, prefix + name)  # noqa: E731
    E_prior = normalize(np.concatenate([normalize(b.static[s_]) for s_ in b.static], 1))
    fold = kfold_perturbation_split(b.pdat.perts, 5, 0)
    ctrl_idx = np.nonzero(lab == ctrl_label)[0]
    rng = np.random.default_rng(args.seed)

    sc_rows, pb_preds, examples, cap_rows = [], {}, {}, []
    EXAMPLES = {"GATA1", "RPS19", "PSMB5", "MED12", "CHMP3", "POLR2A", "SUPT5H", "EIF2S1"}
    folds = [0] if args.dev else [int(x) for x in args.folds.split(",")]
    for f in folds:
        te = np.nonzero(fold == f)[0]
        tr_all = np.nonzero(fold != f)[0]
        tr, va = train_val_split(tr_all, 0.1, f)
        if args.dev:   # never touch the test fold: evaluate on the development perturbations (as 03b)
            tr_all, te = train_val_split(tr_all, 0.1, 0)
            tr, va = train_val_split(tr_all, 0.1, 77)
        te_set = set(int(x) for x in te)
        feats, masks = b.fold_features(tr_all)
        kd = knockdown_fraction(b.pdat.pert_mean[tr_all], ctrl_mean, b.target_idx[tr_all])

        # ---- 0. fold-specific latent space: PCA fitted on control + TRAINING perturbation cells only
        comps, evr = fit_cell_pca(adata.X, lab_full, tr_all, ctrl_label, ctrl_mean, seed=args.seed + f)
        Z = project_cells(adata.X, comps, ctrl_mean)[keep]
        for p in range(len(b.pdat.perts)):   # how much of each mean response does the latent space hold?
            d = b.Y[p]
            cap_rows.append({"fold": f, "pert": b.pdat.perts[p], "split": "test" if p in te_set else "train",
                             "frac_delta_captured": float(((d @ comps.T) ** 2).sum() / max((d ** 2).sum(), 1e-12))})
        print(f"[fold {f}] PCA on training cells: explained var {evr.sum():.3f} ({time.time() - t0:.0f}s)")

        # ---- 1-2. PerturbFormer predictions -> flow conditions ----------------------------------
        p_te, p_tr = [], []
        for e in range(args.n_ens):
            m = train_perturbformer(feats, masks, b.Y, b.W, b.S, tr, va, seed=args.seed * 100 + f * 10 + e)
            p_te.append(predict_perturbformer(m, feats, masks, b.Y, tr_all, te))
            p_tr.append(predict_perturbformer_loo(m, feats, masks, b.Y, tr_all))
        p_te = knockdown_rule(np.mean(p_te, 0), b.pdat.perts[te], b.target_idx[te], ctrl_mean, kd)
        p_tr = knockdown_rule(np.mean(p_tr, 0), b.pdat.perts[tr_all], b.target_idx[tr_all], ctrl_mean, kd)
        cond = {int(p): p_tr[i] @ comps.T for i, p in enumerate(tr_all)}
        print(f"[fold {f}] PerturbFormer done ({time.time() - t0:.0f}s)")

        # ---- 2b. retrieval of similar measured perturbations (prior similarity, training perts only)
        ctrl_lv = np.log(Z[ctrl_idx].var(0) + 1e-6)
        by = {int(p): np.nonzero(lab == p)[0] for p in tr_all}
        lv = {p: np.log(Z[ix].var(0) + 1e-6) - ctrl_lv for p, ix in by.items()}
        S_tr = E_prior[tr_all] @ E_prior[tr_all].T
        np.fill_diagonal(S_tr, -np.inf)                      # leave-self-out for training conditions
        nb_tr = tr_all[np.argsort(-S_tr, 1)[:, :5]]
        nb_te = tr_all[np.argsort(-(E_prior[te] @ E_prior[tr_all].T), 1)[:, :5]]
        v_tr = np.stack([np.mean([lv[int(j)] for j in row], 0) for row in nb_tr])
        v_te = np.stack([np.mean([lv[int(j)] for j in row], 0) for row in nb_te])
        if args.var_cond:
            alpha = np.std(np.stack(list(cond.values()))) / (np.std(v_tr) + 1e-8)
            cond = {int(p): np.concatenate([cond[int(p)], alpha * v_tr[i]]) for i, p in enumerate(tr_all)}

        # ---- 3. train flow ----------------------------------------------------------------------
        c_dim = len(next(iter(cond.values())))
        flow = FlowModel(d=Z.shape[1], c_dim=c_dim, seed=args.seed + f)
        hist = flow.fit(Z, lab, ctrl_label, cond, steps=args.steps)
        torch.save(flow.state_dict(), outp(f"flow_fold{f}.pt"))
        print(f"[fold {f}] flow trained ({time.time() - t0:.0f}s)")

        # ---- 4. evaluate on unseen perturbations -----------------------------------------------
        mean_shift = b.Y[tr_all].mean(0) @ comps.T
        for i, p in enumerate(te):
            x0 = Z[rng.choice(ctrl_idx, args.n_gen, replace=False)]
            true_idx = np.nonzero(lab == p)[0]
            if len(true_idx) > 500:
                true_idx = rng.choice(true_idx, 500, replace=False)
            xt = Z[true_idx]
            c = p_te[i] @ comps.T
            c_full = np.concatenate([c, alpha * v_te[i]]) if args.var_cond else c
            resid = np.concatenate([Z[by[int(j)]] - Z[by[int(j)]].mean(0) for j in nb_te[i]])
            gens = {
                "No-change": x0,
                "Mean-shift": x0 + mean_shift,
                "PerturbFormer-shift": x0 + c,
                "Borrowed-cells": resid[rng.choice(len(resid), args.n_gen, replace=len(resid) < args.n_gen)] + c,
                "Oracle-shift": x0 + (b.Y[p] @ comps.T),
                "CFM (ours)": flow.transport(x0, c_full),
            }
            if b.pdat.perts[p] in EXAMPLES:
                examples[b.pdat.perts[p]] = {k: v.astype(np.float32) for k, v in gens.items()}
                examples[b.pdat.perts[p]]["True"] = xt.astype(np.float32)
            for name, g in gens.items():
                vg, vt = g[:, :20].var(0), xt[:, :20].var(0)
                sc_rows.append({
                    "fold": f, "pert": b.pdat.perts[p], "model": name, "n_true": len(true_idx),
                    "energy_distance": energy_distance(g, xt),
                    "mmd": mmd_median(g, xt),
                    "logvar_err_top20pc": float(np.mean(np.abs(np.log(vg + 1e-6) - np.log(vt + 1e-6)))),
                })
                if name in ("CFM (ours)", "PerturbFormer-shift"):
                    X = decode(g, comps, ctrl_mean, p_te[i])
                    pb_preds.setdefault(name, np.zeros_like(b.Y))[p] = X.mean(0) - ctrl_mean
        print(f"[fold {f}] evaluated {len(te)} unseen perturbations ({time.time() - t0:.0f}s)")
        pd.DataFrame(sc_rows).to_csv(outp("flow_singlecell_metrics.csv"), index=False)
        pd.DataFrame(cap_rows).to_csv(outp("flow_pca_capture.csv"), index=False)

    np.savez_compressed(outp("flow_examples.npz"),
                        **{f"{g}|{k}": v for g, d in examples.items() for k, v in d.items()})
    df = pd.DataFrame(sc_rows)
    summ = df.groupby("model")[["energy_distance", "mmd", "logvar_err_top20pc"]].agg(["mean", "median"])
    summ.to_csv(outp("flow_singlecell_summary.csv"))
    print(summ.round(4))

    # pseudobulk metrics of the decoded generated populations
    idx = np.nonzero(np.abs(pb_preds["CFM (ours)"]).sum(1) > 0)[0]   # evaluated perturbations
    rows = []
    for name, P in pb_preds.items():
        r = summarise(evaluate(P[idx], b.Y[idx], b.pdat.tstat[idx], b.pdat.fdr[idx], b.Y.mean(0), b.ctrl_std))
        r["model"] = name
        rows.append(r)
    pd.DataFrame(rows).set_index("model").to_csv(outp("flow_pseudobulk_summary.csv"))
    print(pd.DataFrame(rows).set_index("model").round(3))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
