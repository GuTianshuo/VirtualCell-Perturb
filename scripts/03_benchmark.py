"""Step 3 - pseudobulk benchmark under the *unseen perturbation* setting.

5-fold cross-validation over perturbations: in each fold 20 % of the perturbed genes are
held out completely (never seen in training).  Every perturbation therefore receives one
out-of-fold (OOF) prediction, which is later used for biological interpretation.

Models
  No-change, Mean-shift                                   (trivial baselines)
  Linear-{GO,PPI,GenePT,Resp}                             (Ahlmann-Eltze et al. 2025)
  kNN-{GO,PPI,GenePT,Resp}                                (similar-gene transfer)
  PerturbNet-concat / PerturbNet-attn  (+ KD rule)        (ours: knowledge-fusion network)
  ablations: -GO, -PPI, -GenePT, -Resp, -dirloss, -KDrule

Usage:
  python scripts/03_benchmark.py --dataset replogle_k562_essential
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp.baselines import (KNNEmbedding, LinearEmbedding, MeanShift, NoChange,  # noqa: E402
                           knockdown_fraction, knockdown_rule)
from vcp.data import ensure_dir, kfold_perturbation_split, train_val_split  # noqa: E402
from vcp.metrics import evaluate  # noqa: E402
from vcp.pipeline import load_bundle  # noqa: E402
from vcp.models import (predict_perturbformer, predict_perturbnet, train_perturbformer,  # noqa: E402
                        train_perturbnet)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
torch.set_num_threads(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="replogle_k562_essential")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default="all")
    ap.add_argument("--n_ens", type=int, default=3, help="ensemble size for PerturbNet")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dev", action="store_true", help="development mode: evaluate on an inner "
                    "validation split of fold 0 only (test folds untouched)")
    args = ap.parse_args()

    t0 = time.time()
    res_dir = ensure_dir(os.path.join(ROOT, "results", args.dataset + args.tag))
    bnd = load_bundle(args.dataset)
    pdat, Y, W, S = bnd.pdat, bnd.Y, bnd.W, bnd.S
    static, sources = bnd.static, list(bnd.static)
    ctrl_std, target_idx = bnd.ctrl_std, bnd.target_idx
    print(f"[{args.dataset}] P={len(pdat.perts)} G={len(pdat.genes)} sources={sources}")

    fold = kfold_perturbation_split(pdat.perts, args.folds, args.seed)
    np.save(os.path.join(res_dir, "folds.npy"), fold)

    records, oof = [], {}
    n_src = len(sources) + 1
    attn_all = np.full((len(pdat.perts), n_src), np.nan, dtype=np.float32)
    neigh_idx = np.full((len(pdat.perts), 10), -1, dtype=np.int64)       # PerturbFormer retrieval
    neigh_w = np.zeros((len(pdat.perts), 10), dtype=np.float32)
    gate_all = np.full((len(pdat.perts), 10), np.nan, dtype=np.float32)
    want = lambda name: args.models == "all" or any(m in name for m in args.models.split(","))  # noqa: E731
    folds = [0] if args.dev else range(args.folds)

    for f in folds:
        te = np.nonzero(fold == f)[0]
        tr_all = np.nonzero(fold != f)[0]
        tr, va = train_val_split(tr_all, 0.1, args.seed + f)
        if args.dev:   # development mode: never touch the real test fold
            tr_all, te = tr, va
            tr, va = train_val_split(tr_all, 0.1, args.seed + 77)
        # fold-specific data-driven embedding (uses training perturbations only)
        feats, masks = bnd.fold_features(tr_all)
        kd_frac = knockdown_fraction(pdat.pert_mean[tr_all], pdat.ctrl_mean, target_idx[tr_all])
        ref = Y[tr_all].mean(0)
        preds = {}

        def add(name, P):
            preds[name] = P.astype(np.float32)

        def add_kd(name):
            add(name + "+KD", knockdown_rule(preds[name], pdat.perts[te], target_idx[te], pdat.ctrl_mean, kd_frac))

        add("No-change", NoChange().fit(Y[tr_all]).predict(te))
        add("Mean-shift", MeanShift().fit(Y[tr_all]).predict(te))
        for s in list(feats):
            E = feats[s]
            if want("Linear"):
                add(f"Linear-{s}", LinearEmbedding(k=32, lam=0.1).fit(Y[tr_all], E[tr_all]).predict(E[te]))
            if want("kNN"):
                add(f"kNN-{s}", KNNEmbedding(k=10, tau=0.1).fit(Y[tr_all], E[tr_all]).predict(E[te]))
        if want("Linear"):
            Ecat = np.concatenate([feats[s] for s in feats], 1)
            add("Linear-concat", LinearEmbedding(k=32, lam=0.1).fit(Y[tr_all], Ecat[tr_all]).predict(Ecat[te]))
            add_kd("Linear-concat")
        if want("kNN"):
            Ecat = np.concatenate([feats[s] for s in feats], 1)
            add("kNN-concat", KNNEmbedding(k=10, tau=0.1).fit(Y[tr_all], Ecat[tr_all]).predict(Ecat[te]))
            add_kd("kNN-concat")

        def run_net(name, use_src, fusion="attn", lam_dir=0.5):
            fs = {k: feats[k] for k in use_src}
            ms = {k: masks[k] for k in use_src}
            outs, attns = [], []
            for e in range(args.n_ens):
                model = train_perturbnet(fs, ms, Y, W, S, tr, va, fusion=fusion, lam_dir=lam_dir,
                                         epochs=args.epochs, seed=args.seed * 100 + f * 10 + e)
                if fusion == "attn":
                    o, a = predict_perturbnet(model, fs, ms, te, return_attn=True)
                    attns.append(a)
                else:
                    o = predict_perturbnet(model, fs, ms, te)
                outs.append(o)
            add(name, np.mean(outs, 0))
            if attns and name == "PerturbNet-attn":
                attn_all[te, : len(use_src)] = np.mean(attns, 0)
            add_kd(name)

        def run_former(name, use_src, lam_dir=0.5, record=False, **kw):
            fs = {k: feats[k] for k in use_src}
            ms = {k: masks[k] for k in use_src}
            outs, attns, gates = [], [], []
            for e in range(args.n_ens):
                model = train_perturbformer(fs, ms, Y, W, S, tr, va, lam_dir=lam_dir, epochs=args.epochs,
                                            seed=args.seed * 100 + f * 10 + e, **kw)
                # memory at test time = all training perturbations (train + validation)
                o, a, g = predict_perturbformer(model, fs, ms, Y, tr_all, te, return_attn=True)
                outs.append(o)
                attns.append(a)
                gates.append(g)
            add(name, np.mean(outs, 0))
            add_kd(name)
            if record:
                a = np.mean(attns, 0)
                top = np.argsort(-a, 1)[:, :10]
                neigh_idx[te] = tr_all[top]
                neigh_w[te] = np.take_along_axis(a, top, 1)
                g = np.mean(gates, 0)
                gate_all[te, : g.shape[1]] = g

        all_src = list(feats)
        if want("PerturbNet"):
            run_net("PerturbNet-attn", all_src, "attn")
            run_net("PerturbNet-concat", all_src, "concat")
        if want("PerturbFormer"):
            run_former("PerturbFormer", all_src, record=True)
        if want("Ablation"):
            for s in all_src:
                run_former(f"Ablation-no-{s}", [x for x in all_src if x != s])
            run_former("Ablation-no-dirloss", all_src, lam_dir=0.0)
            run_former("Ablation-concat-fusion", all_src, fusion="concat")
            run_former("Ablation-free-attention", all_src, kernel=False)
            run_former("Ablation-no-topk", all_src, topk=0)

        for name, P in preds.items():
            r = evaluate(P, Y[te], pdat.tstat[te], pdat.fdr[te], ref, ctrl_std)
            for i, p in enumerate(te):
                rec = {"fold": f, "model": name, "pert": pdat.perts[p], "n_cells": int(pdat.n_cells[p]),
                       "n_deg": int((pdat.fdr[p] < 0.05).sum())}
                rec.update({k: float(v[i]) for k, v in r.items()})
                records.append(rec)
            oof.setdefault(name, np.zeros_like(Y))[te] = P
        print(f"  fold {f} done ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(res_dir, "per_perturbation_metrics.csv"), index=False)
    summ = df.drop(columns=["fold", "pert", "n_cells", "n_deg"]).groupby("model").mean()
    summ = summ.sort_values("pearson_de20", ascending=False)
    summ.to_csv(os.path.join(res_dir, "summary.csv"))
    np.savez_compressed(os.path.join(res_dir, "oof_predictions.npz"),
                        **{k.replace("-", "_").replace("+", "_plus_"): v for k, v in oof.items()},
                        perts=pdat.perts, genes=pdat.genes, attn=attn_all, neigh_idx=neigh_idx,
                        neigh_w=neigh_w, gate=gate_all,
                        attn_sources=np.array(list(static) + ["resp"]))
    with open(os.path.join(res_dir, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)
    pd.set_option("display.width", 200)
    print(summ[["pearson_all", "pearson_de20", "mse_de20", "dir_acc_de20", "dir_acc_de100",
                "pearson_sys", "de_overlap20", "rank"]].round(3))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
