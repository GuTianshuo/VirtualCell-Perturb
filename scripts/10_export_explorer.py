"""Step 10 - export the out-of-fold predictions into a self-contained interactive explorer.

Reads the benchmark results and writes docs/explorer.html (open it in any browser).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp.interpret import gene_class, read_gmt  # noqa: E402
from vcp.pipeline import ROOT, load_bundle  # noqa: E402

DS = "replogle_k562_essential"
RES = os.path.join(ROOT, "results", DS)


def r3(x):
    return float(np.round(x, 3))


def main():
    b = load_bundle(DS)
    pdat, Y = b.pdat, b.Y
    genes, perts = list(pdat.genes), list(pdat.perts)
    oof = np.load(os.path.join(RES, "oof_predictions.npz"), allow_pickle=True)
    pred = oof["PerturbFormer_plus_KD"]
    met = pd.read_csv(os.path.join(RES, "per_perturbation_metrics.csv"))
    mm = met[met.model == "PerturbFormer+KD"].set_index("pert")
    summ = pd.read_csv(os.path.join(RES, "summary.csv"), index_col=0)
    nc = pd.read_csv(os.path.join(RES, "noise_ceiling.csv"), index_col=0)["value"]
    hall = read_gmt(os.path.join(ROOT, "data", "raw", "genesets", "MSigDB_Hallmark_2020.gmt"), universe=genes)
    gi = {g: i for i, g in enumerate(genes)}
    hsets = [(n, [gi[g] for g in gs]) for n, gs in hall.items() if len(gs) >= 10]
    order = np.argsort(-np.abs(pdat.tstat), 1)
    src_names = ["GO", "PPI", "GenePT", "应答谱"]
    t_idx = b.target_idx

    items = []
    for i, p in enumerate(perts):
        top = order[i, :20]
        top = top[np.argsort(Y[i, top])]
        ptop = [j for j in np.argsort(-np.abs(pred[i]))[:10]]
        pw = []
        for n, ii in hsets:
            pw.append((n, float(Y[i, ii].mean()), float(pred[i, ii].mean())))
        pw.sort(key=lambda r: -abs(r[1]))
        nb = [(perts[j], r3(w)) for j, w in zip(oof["neigh_idx"][i, :6], oof["neigh_w"][i, :6]) if j >= 0]
        g = oof["gate"][i]
        sw = [r3(x) for x in g[3:7]] if np.isfinite(g[3:7]).all() else []
        m = mm.loc[p]
        items.append({
            "g": p, "c": gene_class(p), "n": int(pdat.n_cells[i]), "d": int((pdat.fdr[i] < 0.05).sum()),
            "m": [r3(m.pearson_all), r3(m.pearson_de20), r3(m.pearson_sys),
                  None if np.isnan(m.dir_acc_de20) else r3(m.dir_acc_de20)],
            "top": [[genes[j], r3(Y[i, j]), r3(pred[i, j])] for j in top],
            "ptop": [[genes[j], r3(Y[i, j]), r3(pred[i, j])] for j in ptop],
            "pw": [[n.replace("HALLMARK_", ""), r3(a), r3(c)] for n, a, c in pw[:8]],
            "nb": nb, "sw": sw,
            "kd": None if t_idx[i] < 0 else [r3(Y[i, t_idx[i]]), r3(pred[i, t_idx[i]])],
        })
    rows = []
    for mdl, lab in [("PerturbFormer+KD", "PerturbFormer（本文）"), ("kNN-concat+KD", "kNN 多源先验"),
                     ("Linear-concat+KD", "线性基线"), ("PerturbNet-attn+KD", "PerturbNet"), ("Mean-shift", "均值平移")]:
        if mdl in summ.index:
            s = summ.loc[mdl]
            rows.append([lab, r3(s.pearson_all), r3(s.pearson_de20), r3(s.pearson_sys), r3(s.dir_acc_de20), r3(s["rank"])])
    rows.append(["测量噪声决定的可达上限", r3(nc["pearson_all_pred_ceiling"]), r3(nc["pearson_de20_pred_ceiling"]),
                 r3(nc["pearson_sys_pred_ceiling"]), None, None])
    med = [r3(np.nanmedian(mm[k])) for k in ("pearson_all", "pearson_de20", "pearson_sys", "dir_acc_de20")]
    data = {"items": items, "bench": rows, "median": med, "sources": src_names}
    with open(os.path.join(ROOT, "docs", "explorer_template.html")) as fh:
        html = fh.read()
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    out = os.path.join(ROOT, "docs", "explorer.html")
    with open(out, "w") as fh:   # standalone document: open it directly in any browser
        fh.write('<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
                 '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
                 '<style>:root{color-scheme:light}body{margin:0}[hidden]{display:none!important}</style>\n'
                 + html.replace("<div class=\"wrap\">", "</head>\n<body>\n<div class=\"wrap\">", 1) + "\n</body>\n</html>\n")
    if len(sys.argv) > 1:          # optional: page fragment (no html/head/body) for hosted viewers
        with open(sys.argv[1], "w") as fh:
            fh.write(html)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB, {len(items)} perturbations)")


if __name__ == "__main__":
    main()
