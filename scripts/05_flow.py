"""Step 5 - single-cell generative model: conditional flow matching (advanced version).

For every CV fold (same unseen-perturbation folds as 03_benchmark.py):
  1. train PerturbFormer on the training perturbations
  2. conditions: PerturbFormer predictions projected onto the PCA basis
       - training perturbations: leave-self-out predictions (no label leakage into the condition)
       - test perturbations: ordinary predictions (gene never seen)
  3. train the OT-CFM velocity field on single cells of the training perturbations
  4. for each unseen test perturbation, push 300 control cells through the flow and compare
     the generated population with the real perturbed cells.

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

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp.baselines import knockdown_fraction, knockdown_rule  # noqa: E402
from vcp.data import ensure_dir, kfold_perturbation_split, train_val_split  # noqa: E402
from vcp.flow import FlowModel, decode  # noqa: E402
from vcp.metrics import energy_distance, evaluate, summarise  # noqa: E402
from vcp.models import predict_perturbformer, predict_perturbformer_loo, train_perturbformer  # noqa: E402
from vcp.pipeline import PROC, ROOT, load_bundle  # noqa: E402

torch.set_num_threads(4)


def mmd_median(x, y):
    """RBF-MMD² with bandwidths at {0.5,1,2}× the median pairwise distance of y."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="replogle_k562_essential")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--n_gen", type=int, default=300)
    ap.add_argument("--n_ens", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    b = load_bundle(args.dataset)
    cells = np.load(os.path.join(PROC, f"{args.dataset}_cells.npz"), allow_pickle=True)
    Z, lab, ctrl_label = cells["Z"], cells["label"], int(cells["ctrl_label"])
    comps, ctrl_mean = cells["components"], cells["ctrl_mean"]
    res_dir = ensure_dir(os.path.join(ROOT, "results", args.dataset))
    fold = kfold_perturbation_split(b.pdat.perts, 5, 0)
    ctrl_idx = np.nonzero(lab == ctrl_label)[0]
    rng = np.random.default_rng(args.seed)

    sc_rows, pb_preds, examples = [], {}, {}
    EXAMPLES = {"GATA1", "RPS19", "PSMB5", "MED12", "CHMP3", "POLR2A", "SUPT5H", "EIF2S1"}
    for f in [int(x) for x in args.folds.split(",")]:
        te = np.nonzero(fold == f)[0]
        tr_all = np.nonzero(fold != f)[0]
        tr, va = train_val_split(tr_all, 0.1, f)
        feats, masks = b.fold_features(tr_all)
        kd = knockdown_fraction(b.pdat.pert_mean[tr_all], ctrl_mean, b.target_idx[tr_all])

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

        # ---- 3. train flow ----------------------------------------------------------------------
        flow = FlowModel(d=Z.shape[1], c_dim=Z.shape[1], seed=args.seed + f)
        hist = flow.fit(Z, lab, ctrl_label, cond, steps=args.steps)
        torch.save(flow.state_dict(), os.path.join(res_dir, f"flow_fold{f}.pt"))
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
            gens = {
                "No-change": x0,
                "Mean-shift": x0 + mean_shift,
                "PerturbFormer-shift": x0 + c,
                "Oracle-shift": x0 + (b.Y[p] @ comps.T),
                "CFM (ours)": flow.transport(x0, c),
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
        pd.DataFrame(sc_rows).to_csv(os.path.join(res_dir, "flow_singlecell_metrics.csv"), index=False)

    np.savez_compressed(os.path.join(res_dir, "flow_examples.npz"),
                        **{f"{g}|{k}": v for g, d in examples.items() for k, v in d.items()})
    df = pd.DataFrame(sc_rows)
    summ = df.groupby("model")[["energy_distance", "mmd", "logvar_err_top20pc"]].agg(["mean", "median"])
    summ.to_csv(os.path.join(res_dir, "flow_singlecell_summary.csv"))
    print(summ.round(4))

    # pseudobulk metrics of the decoded generated populations
    done = np.isin(fold, [int(x) for x in args.folds.split(",")])
    idx = np.nonzero(done)[0]
    rows = []
    for name, P in pb_preds.items():
        r = summarise(evaluate(P[idx], b.Y[idx], b.pdat.tstat[idx], b.pdat.fdr[idx], b.Y.mean(0), b.ctrl_std))
        r["model"] = name
        rows.append(r)
    pd.DataFrame(rows).set_index("model").to_csv(os.path.join(res_dir, "flow_pseudobulk_summary.csv"))
    print(pd.DataFrame(rows).set_index("model").round(3))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
