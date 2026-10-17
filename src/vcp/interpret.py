"""Biological interpretation helpers: gene sets, functional classes, enrichment."""
from __future__ import annotations

import re

import numpy as np
from scipy.stats import hypergeom

from .data import bh_fdr


def read_gmt(path: str, min_size: int = 1, universe=None) -> dict:
    sets = {}
    uni = set(universe) if universe is not None else None
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            genes = [g for g in p[2:] if g]
            if uni is not None:
                genes = [g for g in genes if g in uni]
            if len(genes) >= min_size:
                sets[p[0]] = sorted(set(genes))
    return sets


# curated functional classes of essential genes (regex on HGNC symbol)
CLASS_RULES = [
    ("胞质核糖体 Ribosome", r"^RP[LS]\d+[A-Z]?\d*$|^RPLP\d$|^RPSA$"),
    ("线粒体核糖体 Mito-ribosome", r"^MRP[LS]\d+"),
    ("蛋白酶体 Proteasome", r"^PSM[ABCD]\d+"),
    ("Mediator 复合体", r"^MED\d+"),
    ("RNA Pol II / 转录起始", r"^POLR2[A-L]$|^GTF2|^TAF\d+|^SUPT\d+H|^NELF|^CDK9$|^CCNT1$|^INTS\d+"),
    ("RNA Pol I/III", r"^POLR[13]|^POLR2[EFHKL]$"),
    ("翻译起始 eIF", r"^EIF\d"),
    ("剪接体 Spliceosome", r"^SF3[AB]\d|^PRPF\d|^SNRP|^SNRNP|^LSM\d|^U2AF|^SRSF\d|^PHF5A$|^DHX(15|16|38)$|^AQR$|^SLU7$"),
    ("RNA 外切体 Exosome", r"^EXOSC\d+|^DIS3$"),
    ("DNA 复制", r"^MCM\d+$|^GINS\d$|^POL[ADE]\d$|^RFC\d$|^PCNA$|^CDC45$|^PRIM\d$"),
    ("伴侣蛋白 CCT/TRiC", r"^CCT\d|^TCP1$"),
    ("tRNA 合成酶", r"^[A-Z]ARS2?$"),
    ("核孔 Nuclear pore", r"^NUP\d+"),
    ("核糖体生物合成 (UTP/NOP/WDR)", r"^UTP\d+|^NOP\d+|^WDR(3|12|43|46|74|75)$|^NOL\d+|^RRP\d+|^BMS1$|^TSR\d$"),
]


def gene_class(g: str) -> str:
    for name, pat in CLASS_RULES:
        if re.search(pat, g):
            return name
    return "其他 Other"


def enrichment(hits, universe, gene_sets: dict, min_overlap: int = 2, top: int = 10):
    """Hypergeometric over-representation of `hits` within `universe`."""
    hits, uni = set(hits), set(universe)
    N, n = len(uni), len(hits)
    rows = []
    for name, genes in gene_sets.items():
        gs = set(genes) & uni
        k = len(gs & hits)
        if k < min_overlap or len(gs) < 5:
            continue
        p = hypergeom.sf(k - 1, N, len(gs), n)
        rows.append((name, k, len(gs), p, sorted(gs & hits)))
    if not rows:
        return []
    q = bh_fdr(np.array([r[3] for r in rows]))
    rows = [(*r[:4], float(qq), r[4]) for r, qq in zip(rows, q)]
    return sorted(rows, key=lambda r: r[3])[:top]


def pathway_scores(delta: np.ndarray, genes, gene_sets: dict, min_size: int = 10):
    """Mean delta of each gene set (P, n_sets)."""
    idx = {g: i for i, g in enumerate(genes)}
    names, cols = [], []
    for name, gs in gene_sets.items():
        ii = [idx[g] for g in gs if g in idx]
        if len(ii) >= min_size:
            names.append(name)
            cols.append(delta[:, ii].mean(1))
    return np.stack(cols, 1), names
