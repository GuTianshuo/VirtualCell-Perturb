"""Step 12 - published baseline: GEARS (Roohani, Huang & Leskovec, Nat Biotechnol 2024).

GEARS introduced the preprocessed Perturb-seq files used throughout this project, so it is the
natural published method to compare against.  It is trained here on exactly the same
train / validation / test perturbations as our models (CV fold `--fold` of split `--seed`),
and its predictions are scored with our metrics on the held-out genes.

CPU budget: GEARS trains on single cells (one graph of all genes per cell), so cells are
subsampled to `--cells_per_pert` per perturbation (+ `--n_ctrl` controls).  Everything else uses
the GEARS defaults (hidden 64, lr 1e-3, batch 32, GO + co-expression graphs).

Compatibility: cell-gears 0.1.2 predates pandas 3; two small functions are patched at runtime
(positional Series indexing and groupby().apply dropping the key column).  The library itself
is not modified.

Output: results/<ds>/gears_fold<f>_seed<s>.csv (per perturbation), results/<ds>/gears_summary.md
"""
import argparse
import os
import pickle
import shutil
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp.baselines import knockdown_fraction, knockdown_rule  # noqa: E402
from vcp.data import ensure_dir, kfold_perturbation_split, train_val_split  # noqa: E402
from vcp.metrics import evaluate  # noqa: E402
from vcp.pipeline import ROOT, load_bundle  # noqa: E402

DS = "replogle_k562_essential"
KEYS = ["pearson_all", "pearson_de20", "pearson_sys", "dir_acc_de20", "de_overlap20", "rank"]


def patch_gears():
    import gears.gears as gg
    import gears.pertdata as gp
    import gears.utils as gu

    def create_cell_graph_dataset(self, split_adata, pert_category, num_samples=1):
        num_de_genes = 20
        adata_ = split_adata[split_adata.obs["condition"] == pert_category]
        de = "rank_genes_groups_cov_all" in adata_.uns
        de_genes = adata_.uns["rank_genes_groups_cov_all"] if de else None
        Xs, ys = [], []
        if pert_category != "ctrl":
            pert_idx = self.get_pert_idx(pert_category)
            pert_de_category = adata_.obs["condition_name"].iloc[0]          # was [0] (pandas<3 only)
            if de:
                de_idx = np.where(adata_.var_names.isin(np.array(de_genes[pert_de_category][:num_de_genes])))[0]
            else:
                de_idx = [-1] * num_de_genes
            for cell_z in adata_.X:
                ctrl_samples = self.ctrl_adata[np.random.randint(0, len(self.ctrl_adata), num_samples), :]
                for c in ctrl_samples.X:
                    Xs.append(c)
                    ys.append(cell_z)
        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            for cell_z in adata_.X:
                Xs.append(cell_z)
                ys.append(cell_z)
        return [self.create_cell_graph(X.toarray(), y.toarray(), de_idx, pert_category, pert_idx)
                for X, y in zip(Xs, ys)]

    def get_similarity_network(network_type, adata, threshold, k, data_path, data_name, split, seed,
                               train_gene_set_size, set2conditions, default_pert_graph=True, pert_list=None):
        if network_type == "co-express":
            return gu.get_coexpression_network_from_train(adata, threshold, k, data_path, data_name, split,
                                                          seed, train_gene_set_size, set2conditions)
        gu.tar_data_download_wrapper("https://dataverse.harvard.edu/api/access/datafile/6934319",
                                     os.path.join(data_path, "go_essential_all"), data_path)
        df = pd.read_csv(os.path.join(data_path, "go_essential_all/go_essential_all.csv"))
        # was groupby('target').apply(nlargest) - pandas 3 drops the key column
        return df.sort_values("importance", ascending=False).groupby("target").head(k + 1).reset_index(drop=True)

    gp.PertData.create_cell_graph_dataset = create_cell_graph_dataset
    gg.get_similarity_network = get_similarity_network


class PandasMaskCSR(sp.csr_matrix):
    """csr_matrix that accepts a pandas boolean Series as row mask (newer scipy no longer does)."""

    def __getitem__(self, key):
        if isinstance(key, pd.Series):
            key = key.to_numpy()
        return super().__getitem__(key)


def subsample_h5ad(src, dst, cells_per_pert, n_ctrl, seed=0):
    if os.path.exists(dst):
        return
    a = ad.read_h5ad(src, backed="r")
    cond = a.obs["condition"].astype(str).values
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(cond):
        idx = np.nonzero(cond == c)[0]
        m = n_ctrl if c == "ctrl" else cells_per_pert
        keep.append(rng.choice(idx, min(m, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    sub = a[keep].to_memory()
    ensure_dir(os.path.dirname(dst))
    sub.write_h5ad(dst)
    print(f"subsampled {len(keep)} / {a.n_obs} cells -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="CV split seed (0 = main benchmark split)")
    ap.add_argument("--cells_per_pert", type=int, default=30)
    ap.add_argument("--n_ctrl", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max_batches", type=int, default=0, help="timing test: stop each epoch early")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed * 100 + args.fold)
    np.random.seed(args.seed * 100 + args.fold)
    patch_gears()
    from gears import GEARS, PertData

    t0 = time.time()
    raw = os.path.join(ROOT, "data", "raw")
    work = ensure_dir(os.path.join(ROOT, "data", "gears_work"))
    for srcf, dstf in [("gene2go.pkl", "gene2go_all.pkl"), ("gene2go.pkl", "gene2go.pkl"),
                       ("essential_all_data_pert_genes.pkl", "essential_all_data_pert_genes.pkl")]:
        if not os.path.exists(os.path.join(work, dstf)):
            shutil.copy(os.path.join(raw, srcf), os.path.join(work, dstf))
    name = f"k562_sub{args.cells_per_pert}"
    subsample_h5ad(os.path.join(raw, DS, "perturb_processed.h5ad"), os.path.join(work, name, "perturb_processed.h5ad"),
                   args.cells_per_pert, args.n_ctrl)

    pert_data = PertData(work)
    pert_data.load(data_path=os.path.join(work, name))
    pert_data.adata.X = PandasMaskCSR(pert_data.adata.X)   # GEARS indexes X with pandas boolean Series
    available = set(pert_data.adata.obs["condition"].astype(str).unique())

    b = load_bundle(DS)
    perts = b.pdat.perts
    fold = kfold_perturbation_split(perts, 5, args.seed)
    te = np.nonzero(fold == args.fold)[0]
    tr_all = np.nonzero(fold != args.fold)[0]
    tr, va = train_val_split(tr_all, 0.1, args.seed + args.fold)
    cname = lambda idx: [f"{perts[i]}+ctrl" for i in idx if f"{perts[i]}+ctrl" in available]  # noqa: E731
    split = {"train": cname(tr) + ["ctrl"], "val": cname(va), "test": cname(te)}
    print({k: len(v) for k, v in split.items()}, f"(test perts without GO coverage: {len(te) - len(split['test'])})")
    sp = os.path.join(work, f"split_seed{args.seed}_fold{args.fold}.pkl")
    pickle.dump(split, open(sp, "wb"))
    pert_data.prepare_split(split="custom", split_dict_path=sp, seed=1000 * args.seed + args.fold)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)
    if args.max_batches:
        for k in ("train_loader", "val_loader"):
            dl = pert_data.dataloader[k]
            pert_data.dataloader[k] = [bt for i, bt in zip(range(args.max_batches), dl)]
    model = GEARS(pert_data, device="cpu")
    model.model_initialize(hidden_size=64)
    t1 = time.time()
    model.train(epochs=args.epochs, lr=1e-3)
    t_train = time.time() - t1
    print(f"GEARS trained: {args.epochs} epochs in {t_train:.0f}s")
    if args.max_batches:
        print(f"timing: {t_train / args.epochs / args.max_batches:.3f} s per batch (train+val)")
        return

    # ---- predict held-out perturbations, score with our metrics ----------------------------------
    test_genes = [c.replace("+ctrl", "") for c in split["test"]]
    pred = model.predict([[g] for g in test_genes])
    idx = np.array([list(perts).index(g) for g in test_genes])
    P = np.stack([np.asarray(pred[g]).ravel() for g in test_genes]) - b.pdat.ctrl_mean[None]
    kd = knockdown_fraction(b.pdat.pert_mean[tr_all], b.pdat.ctrl_mean, b.target_idx[tr_all])
    ref = b.Y[tr_all].mean(0)
    rows = []
    for nm, M in [("GEARS", P), ("GEARS+KD", knockdown_rule(P, perts[idx], b.target_idx[idx], b.pdat.ctrl_mean, kd))]:
        ev = evaluate(M, b.Y[idx], b.pdat.tstat[idx], b.pdat.fdr[idx], ref, b.ctrl_std)
        for i, g in enumerate(test_genes):
            rows.append({"model": nm, "pert": g, "fold": args.fold, **{k: float(ev[k][i]) for k in KEYS}})
    res_dir = ensure_dir(os.path.join(ROOT, "results", DS + ("" if args.seed == 0 else f"_seed{args.seed}")))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(res_dir, f"gears_fold{args.fold}_seed{args.seed}.csv"), index=False)

    ours = pd.read_csv(os.path.join(res_dir, "per_perturbation_metrics.csv"))
    ours = ours[ours.pert.isin(test_genes)]
    allm = pd.concat([ours[["model", "pert"] + KEYS], out[["model", "pert"] + KEYS]])
    show = ["Mean-shift", "Linear-concat+KD", "PerturbNet-attn+KD", "kNN-concat+KD", "GEARS", "GEARS+KD", "PerturbFormer+KD"]
    tab = allm[allm.model.isin(show)].groupby("model")[KEYS].mean().loc[show]
    lines = [f"## GEARS（已发表方法）与本项目模型对比：第 {args.fold} 折，{len(test_genes)} 个未见基因", "",
             f"GEARS 设置：每个扰动 {args.cells_per_pert} 个细胞 + {args.n_ctrl} 个对照细胞，{args.epochs} 个 epoch，其余为默认参数；"
             f"训练用时 {t_train / 60:.0f} 分钟（4 核 CPU）。", "",
             "| 模型 | " + " | ".join(KEYS) + " |", "|---" * (len(KEYS) + 1) + "|"]
    for mdl, r in tab.iterrows():
        lines.append(f"| {mdl} | " + " | ".join(f"{r[k]:.3f}" for k in KEYS) + " |")
    pf = allm[allm.model == "PerturbFormer+KD"].set_index("pert")
    gk = allm[allm.model == "GEARS+KD"].set_index("pert").loc[pf.index]
    lines += ["", "PerturbFormer+KD 减去 GEARS+KD（配对 Wilcoxon）：" + "；".join(
        f"{k} {pf[k].mean() - gk[k].mean():+.3f} (p={wilcoxon(pf[k], gk[k]).pvalue:.0e})" for k in KEYS)]
    with open(os.path.join(res_dir, "gears_summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
