"""VirtualCell: the end-user API.

    vc = VirtualCell.load("models/k562")
    out = vc.predict("GATA1", cells=X_ctrl)     # X_ctrl: (n, G) log-normalised control cells (optional)
    out["cells"]          -> (n, G) predicted post-knockdown expression of *these* cells
    out["delta"]          -> (G,)   predicted pseudobulk change
    out["neighbors"]      -> measured perturbations the prediction is based on (attention weights)
    print(vc.report(out)) -> human-readable summary (top genes, pathways, evidence)
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from .flow import FlowModel, decode
from .models import KernelPerturbFormer, _to_t


class VirtualCell:
    def __init__(self, meta: dict, formers: list, flow: FlowModel, universe: dict, gene_sets: dict | None = None):
        self.m = meta
        self.genes = list(meta["genes"])
        self.perts = list(meta["perts"])
        self.formers = formers
        self.flow = flow
        self.uni = universe
        self.uni_idx = {g: i for i, g in enumerate(universe["genes"])}
        self.gene_idx = {g: i for i, g in enumerate(self.genes)}
        self.gene_sets = gene_sets or {}
        self.sources = [s for s in ("go", "ppi", "genept") if s in universe]

    # ------------------------------------------------------------------------------------
    @classmethod
    def load(cls, model_dir: str, gmt: str | None = None):
        z = np.load(os.path.join(model_dir, "meta.npz"), allow_pickle=True)
        meta = {k: z[k] for k in z.files}
        with open(os.path.join(model_dir, "config.json")) as fh:
            cfg = json.load(fh)
        dims = cfg["dims"]
        formers = []
        for i in range(cfg["n_ens"]):
            mdl = KernelPerturbFormer(dims, len(meta["genes"]), np.zeros((len(meta["genes"]), cfg["basis_k"])),
                                      np.zeros(len(meta["genes"])), **cfg["former_kwargs"])
            mdl.load_state_dict(torch.load(os.path.join(model_dir, f"perturbformer_{i}.pt"), weights_only=False))
            mdl.eval()
            formers.append(mdl)
        sd = torch.load(os.path.join(model_dir, "flow.pt"), weights_only=False)
        flow = FlowModel(d=meta["components"].shape[0], c_dim=meta["components"].shape[0], **cfg["flow_kwargs"])
        flow.load_state_dict(sd)
        uz = np.load(os.path.join(model_dir, "gene_universe_embeddings.npz"), allow_pickle=True)
        universe = {k: uz[k] for k in uz.files}
        sets = {}
        if gmt and os.path.exists(gmt):
            from .interpret import read_gmt
            sets = read_gmt(gmt, universe=meta["genes"])
        return cls(meta, formers, flow, universe, sets)

    # ------------------------------------------------------------------------------------
    def _features(self, genes):
        """Prior-knowledge features for arbitrary genes (+ response embedding if measured)."""
        f, m = {}, {}
        for s in self.sources:
            E = np.zeros((len(genes), self.uni[s].shape[1]), np.float32)
            M = np.zeros(len(genes), np.float32)
            for i, g in enumerate(genes):
                j = self.uni_idx.get(g)
                if j is not None:
                    E[i], M[i] = self.uni[s][j], self.uni[f"{s}_mask"][j]
            f[s], m[s] = E, M
        R = self.m["resp_gene_emb"]                    # (G, 64) response embedding of measured genes
        E = np.zeros((len(genes), R.shape[1]), np.float32)
        M = np.zeros(len(genes), np.float32)
        for i, g in enumerate(genes):
            j = self.gene_idx.get(g)
            if j is not None:
                E[i], M[i] = R[j], 1.0
        f["resp"], m["resp"] = E, M
        return f, m

    def known_gene(self, gene: str) -> bool:
        return gene in self.uni_idx and any(self.uni[f"{s}_mask"][self.uni_idx[gene]] for s in self.sources)

    @torch.no_grad()
    def predict_delta(self, gene: str, exclude_self: bool = True):
        """Pseudobulk prediction. If the gene was perturbed in the training screen and
        exclude_self=True, it is removed from the memory (genuine 'unseen' prediction)."""
        mem = [i for i, p in enumerate(self.perts) if not (exclude_self and p == gene)]
        fq, mq = self._features([gene])
        fk = {k: v[mem] for k, v in self.m_feats.items()}
        mk = {k: v[mem] for k, v in self.m_masks.items()}
        Yk = torch.tensor(self.m["Y"][mem], dtype=torch.float32)
        outs, attn, gates = [], [], []
        for mdl in self.formers:
            o, a, g = mdl(_to_t(fq), _to_t(mq), _to_t(fk), _to_t(mk), Yk, return_attn=True)
            outs.append(o.numpy()[0])
            attn.append(a.numpy()[0])
            gates.append(g.numpy()[0])
        delta = np.mean(outs, 0)
        a = np.mean(attn, 0)
        g = np.mean(gates, 0)
        t = self.gene_idx.get(gene, -1)
        if t >= 0:   # CRISPRi knockdown rule
            delta[t] = self.m["ctrl_mean"][t] * (float(self.m["kd_frac"]) - 1.0)
        top = np.argsort(-a)[:8]
        src = dict(zip(self.sources + ["resp"], g[3:].tolist()))
        return {"gene": gene, "delta": delta, "neighbors": [(self.perts[mem[j]], float(a[j])) for j in top],
                "source_weights": src, "retrieval_gate": float(g[:2].sum()), "target_idx": t}

    @property
    def m_feats(self):
        if not hasattr(self, "_mf"):
            self._mf, self._mm = self._features(self.perts)
        return self._mf

    @property
    def m_masks(self):
        self.m_feats
        return self._mm

    def predict(self, gene: str, cells: np.ndarray | None = None, n_cells: int = 300,
                exclude_self: bool = True, seed: int = 0):
        """Full virtual-cell prediction: pseudobulk + single-cell counterfactuals."""
        out = self.predict_delta(gene, exclude_self=exclude_self)
        comps, mu = self.m["components"], self.m["ctrl_mean"]
        if cells is None:
            rng = np.random.default_rng(seed)
            Zc = self.m["ctrl_latent"]
            z0 = Zc[rng.choice(len(Zc), min(n_cells, len(Zc)), replace=False)]
        else:
            z0 = (np.asarray(cells, np.float32) - mu[None]) @ comps.T
        z1 = self.flow.transport(z0, out["delta"] @ comps.T)
        out.update({"z_control": z0, "z_pred": z1, "cells": decode(z1, comps, mu, out["delta"]),
                    "cells_control": decode(z0, comps, mu)})
        if gene in self.perts:
            out["measured_delta"] = self.m["Y"][self.perts.index(gene)]
        return out

    # ------------------------------------------------------------------------------------
    def pathway_changes(self, delta, k: int = 6):
        rows = []
        for name, gs in self.gene_sets.items():
            ii = [self.gene_idx[g] for g in gs if g in self.gene_idx]
            if len(ii) >= 10:
                rows.append((name, float(delta[ii].mean()), len(ii)))
        rows.sort(key=lambda r: r[1])
        return rows[:k], rows[::-1][:k]

    def report(self, out: dict, k: int = 12) -> str:
        d = out["delta"]
        g = out["gene"]
        lines = [f"# 虚拟细胞预测: 敲低 {g} (CRISPRi, K562)", ""]
        seen = "measured_delta" in out
        lines.append("- 训练筛选中是否测过该扰动: " + ("是。已将其从检索记忆中移除，但部署模型的参数训练时见过它，下方与实测的比较偏乐观；"
                                                  "严格的未见基因评估见 5 折交叉验证结果" if seen else "否 (全新基因)"))
        if out["target_idx"] >= 0:
            lines.append(f"- 靶基因 {g} 预测变化: {d[out['target_idx']]:+.3f} (log 表达)")
        lines.append("- 依据的相似扰动 (注意力权重): " + ", ".join(f"{n} {w:.2f}" for n, w in out["neighbors"][:6]))
        lines.append("- 知识来源权重: " + ", ".join(f"{k_}={v:.2f}" for k_, v in out["source_weights"].items()))
        dn = np.argsort(d)[:k]
        up = np.argsort(-d)[:k]
        lines.append("- 预测下调基因: " + ", ".join(f"{self.genes[i]}({d[i]:+.2f})" for i in dn))
        lines.append("- 预测上调基因: " + ", ".join(f"{self.genes[i]}({d[i]:+.2f})" for i in up))
        if self.gene_sets:
            lo, hi = self.pathway_changes(d)
            lines.append("- 通路活性下降: " + "; ".join(f"{n} ({s:+.3f})" for n, s, _ in lo[:4]))
            lines.append("- 通路活性上升: " + "; ".join(f"{n} ({s:+.3f})" for n, s, _ in hi[:4]))
        if seen:
            md = out["measured_delta"]
            r = np.corrcoef(md, d)[0, 1]
            top = np.argsort(-np.abs(md))[:20]
            rde = np.corrcoef(md[top], d[top])[0, 1]
            lines.append(f"- 与实测比较 (偏乐观, 仅供参考): Pearson(全部基因) = {r:.3f}; Pearson(实测 Top-20 DEG) = {rde:.3f}; "
                         f"方向一致率(Top-20) = {(np.sign(md[top]) == np.sign(d[top])).mean():.0%}")
        if "cells" in out:
            shift = np.linalg.norm(out["z_pred"] - out["z_control"], axis=1)
            lines.append(f"- 单细胞: 生成 {len(out['cells'])} 个扰动后细胞; 潜空间平均位移 {shift.mean():.2f} "
                         f"(细胞间差异 sd={shift.std():.2f})")
        return "\n".join(lines)
