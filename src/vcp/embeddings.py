"""Perturbation encodings: how to represent "knock down gene g" as a vector.

Three complementary sources of prior knowledge are used so that the model can generalise
to genes that were never perturbed during training:

  GO   - Gene Ontology annotations (BP/CC/MF), TF-IDF weighted, truncated SVD
  PPI  - STRING v12 protein-protein interaction network, diffusion (P + P^2) spectral embedding
  RESP - data-driven "response-profile" embedding: how gene g itself responds to the
         *training* perturbations (rows of the training pseudobulk matrix, PCA-reduced).
         Only available when g is one of the measured genes. This is the gene embedding
         used by the linear baseline of Ahlmann-Eltze, Huber & Anders (Nat Methods 2025).

Each encoder returns (matrix (n_genes, d), availability mask (n_genes,)).
"""
from __future__ import annotations

import gzip
import pickle

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize


# ----------------------------------------------------------------------------------------
# GO
# ----------------------------------------------------------------------------------------
def go_embedding(query_genes, gene2go_path: str, dim: int = 128, min_genes: int = 3,
                 universe=None, seed: int = 0, extra_genes=None):
    """GO TF-IDF + SVD embedding.

    The transform is fitted on `universe ∪ query_genes`; `extra_genes` (optional) are embedded
    with the *same* fitted vocabulary / IDF / SVD, so adding them never changes the vectors of
    the query genes.  Returns (E, mask) or (E, mask, E_extra, mask_extra).
    """
    with open(gene2go_path, "rb") as f:
        g2go = pickle.load(f)
    universe = list(dict.fromkeys(list(universe or []) + list(query_genes)))
    universe = [g for g in universe if g in g2go]
    terms = {}
    rows, cols = [], []
    for i, g in enumerate(universe):
        for t in sorted(set(g2go[g])):
            j = terms.setdefault(t, len(terms))
            rows.append(i)
            cols.append(j)
    M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(universe), len(terms)))
    keep = np.asarray(M.sum(0)).ravel() >= min_genes
    M = M[:, keep]
    tfidf = TfidfTransformer().fit(M)
    M = tfidf.transform(M)
    svd = TruncatedSVD(n_components=dim, random_state=seed, n_iter=10).fit(M)
    E = normalize(svd.transform(M))
    lut = {g: i for i, g in enumerate(universe)}

    def lookup(genes, emb_of):
        out = np.zeros((len(genes), dim), dtype=np.float32)
        mask = np.zeros(len(genes), dtype=bool)
        for i, g in enumerate(genes):
            v = emb_of(g)
            if v is not None and np.abs(v).sum() > 0:
                out[i], mask[i] = v, True
        return out, mask

    out, mask = lookup(query_genes, lambda g: E[lut[g]] if g in lut else None)
    if extra_genes is None:
        return out, mask
    term_idx = {t: j for t, j in terms.items()}
    kept = np.nonzero(keep)[0]
    remap = {old: new for new, old in enumerate(kept)}

    def embed_new(g):
        if g in lut:
            return E[lut[g]]
        if g not in g2go:
            return None
        js = sorted({remap[term_idx[t]] for t in set(g2go[g]) if t in term_idx and term_idx[t] in remap})
        if not js:
            return None
        v = sp.csr_matrix((np.ones(len(js)), ([0] * len(js), js)), shape=(1, len(kept)))
        return normalize(svd.transform(tfidf.transform(v)))[0]

    ex, exm = lookup(extra_genes, embed_new)
    return out, mask, ex, exm


def go_term_matrix(genes, gene2go_path: str):
    """Binary gene x GO-term matrix (for interpretation / Jaccard similarity)."""
    with open(gene2go_path, "rb") as f:
        g2go = pickle.load(f)
    terms, rows, cols = {}, [], []
    for i, g in enumerate(genes):
        for t in sorted(set(g2go.get(g, []))):
            j = terms.setdefault(t, len(terms))
            rows.append(i)
            cols.append(j)
    M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(genes), len(terms)))
    return M, np.array(list(terms))


# ----------------------------------------------------------------------------------------
# STRING PPI
# ----------------------------------------------------------------------------------------
def load_string_graph(links_path: str, info_path: str, min_score: int = 400):
    with gzip.open(info_path, "rt") as f:
        next(f)
        ensp2name = {}
        for line in f:
            p = line.rstrip("\n").split("\t")
            ensp2name[p[0]] = p[1]
    names = sorted(set(ensp2name.values()))
    idx = {n: i for i, n in enumerate(names)}
    r, c, w = [], [], []
    with gzip.open(links_path, "rt") as f:
        next(f)
        for line in f:
            a, b, s = line.split()
            s = int(s)
            if s < min_score:
                continue
            r.append(idx[ensp2name[a]])
            c.append(idx[ensp2name[b]])
            w.append(s / 1000.0)
    A = sp.csr_matrix((w, (r, c)), shape=(len(names), len(names)))
    A = A.maximum(A.T)
    return A, np.array(names)


def ppi_embedding(query_genes, links_path: str, info_path: str, dim: int = 128,
                  min_score: int = 700, seed: int = 0, extra_genes=None):
    A, names = load_string_graph(links_path, info_path, min_score=min_score)
    deg = np.asarray(A.sum(1)).ravel()
    dinv = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    P = dinv @ A @ dinv
    M = (P + P @ P).tocsr()
    svd = TruncatedSVD(n_components=dim, random_state=seed, algorithm="randomized", n_iter=7)
    U = svd.fit_transform(M)
    E = normalize(U)
    lut = {g: i for i, g in enumerate(names)}

    def lookup(genes):
        out = np.zeros((len(genes), dim), dtype=np.float32)
        mask = np.zeros(len(genes), dtype=bool)
        for i, g in enumerate(genes):
            j = lut.get(g)
            if j is not None and deg[j] > 0:
                out[i] = E[j]
                mask[i] = True
        return out, mask

    out, mask = lookup(query_genes)
    if extra_genes is None:
        return out, mask
    return (out, mask, *lookup(extra_genes))


# ----------------------------------------------------------------------------------------
# response-profile embedding (fold specific, uses training perturbations only)
# ----------------------------------------------------------------------------------------
def response_embedding(query_genes, measured_genes, Y_train: np.ndarray, dim: int = 64,
                       seed: int = 0):
    """Y_train: (P_train, G) pseudobulk deltas of training perturbations.

    Gene embedding G = left singular vectors of Y_train^T (genes x dim).  For a perturbation of
    gene g we return G[g] if g is measured, else zeros (mask False).
    """
    Yc = Y_train - Y_train.mean(0, keepdims=True)
    k = min(dim, Yc.shape[0] - 1)
    svd = TruncatedSVD(n_components=k, random_state=seed).fit(Yc)
    G = svd.components_.T * svd.singular_values_[None, :]      # (G, k)
    G = normalize(G)
    lut = {g: i for i, g in enumerate(measured_genes)}
    out = np.zeros((len(query_genes), dim), dtype=np.float32)
    mask = np.zeros(len(query_genes), dtype=bool)
    for i, g in enumerate(query_genes):
        j = lut.get(g)
        if j is not None:
            out[i, :k] = G[j]
            mask[i] = True
    return out, mask
