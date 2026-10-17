"""Shared helpers: load a processed dataset together with its perturbation encodings."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .data import PerturbData
from .embeddings import response_embedding

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROC = os.path.join(ROOT, "data", "processed")
SOURCES = ("go", "ppi", "genept")


@dataclass
class Bundle:
    pdat: PerturbData
    static: dict          # source -> (P, d) prior-knowledge embeddings
    static_m: dict        # source -> (P,) availability masks
    Y: np.ndarray         # (P, G) pseudobulk deltas
    W: np.ndarray         # (P, G) loss weights (DE emphasis)
    S: np.ndarray         # (P, G) DE direction labels {-1,0,1}

    @property
    def ctrl_std(self):
        return np.sqrt(self.pdat.ctrl_var)

    @property
    def target_idx(self):
        return self.pdat.extra["target_idx"]

    def fold_features(self, train_idx, extra_query=None):
        """Static embeddings + the response embedding computed from `train_idx` only."""
        q = self.pdat.perts if extra_query is None else np.concatenate([self.pdat.perts, extra_query])
        resp, resp_m = response_embedding(q, self.pdat.genes, self.Y[train_idx], dim=64)
        feats = dict(self.static, resp=resp[: len(self.pdat.perts)])
        masks = dict(self.static_m, resp=resp_m[: len(self.pdat.perts)].astype(np.float32))
        return feats, masks


def loss_weights(pdat: PerturbData):
    order = np.argsort(-np.abs(pdat.tstat), 1)
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.arange(order.shape[1])[None].repeat(len(order), 0), 1)
    W = (1.0 + 5.0 * (r < 20) + 2.0 * (r < 100)).astype(np.float32)
    S = np.where((r < 100) & (pdat.fdr < 0.05), np.sign(pdat.delta), 0).astype(np.float32)
    return W, S


def load_embeddings(genes, path=None):
    emb = np.load(path or os.path.join(PROC, "pert_embeddings.npz"))
    lut = {g: i for i, g in enumerate(emb["genes"])}
    sources = [s for s in SOURCES if s in emb.files]
    out, out_m = {}, {}
    for s in sources:
        E = np.zeros((len(genes), emb[s].shape[1]), dtype=np.float32)
        M = np.zeros(len(genes), dtype=np.float32)
        for i, g in enumerate(genes):
            if g in lut:
                E[i] = emb[s][lut[g]]
                M[i] = emb[f"{s}_mask"][lut[g]]
        out[s], out_m[s] = E, M
    return out, out_m


def load_bundle(dataset: str) -> Bundle:
    pdat = PerturbData.load(os.path.join(PROC, f"{dataset}_pseudobulk.npz"))
    static, static_m = load_embeddings(pdat.perts)
    W, S = loss_weights(pdat)
    return Bundle(pdat, static, static_m, pdat.delta.astype(np.float32), W, S)
