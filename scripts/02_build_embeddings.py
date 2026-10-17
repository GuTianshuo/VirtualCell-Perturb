"""Step 2 - prior-knowledge perturbation embeddings (dataset independent).

Builds, for every perturbed gene of every processed dataset:
  go     : GO-annotation TF-IDF + SVD (128-d)
  ppi    : STRING v12 (score >= 700) diffusion spectral embedding (128-d)
  genept : GenePT (Chen & Zou 2024) GPT-3.5 text embeddings of NCBI gene summaries,
           PCA-reduced to 128-d   [optional - used if data/raw/GenePT* is present]

Output: data/processed/pert_embeddings.npz          (perturbed genes; used for benchmarking)
        data/processed/gene_universe_embeddings.npz (~19k protein-coding genes, same fitted
                                                     transforms -> lets the predictor accept ANY gene)
"""
import glob
import os
import pickle
import sys
import zipfile

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vcp.data import PerturbData  # noqa: E402
from vcp.embeddings import go_embedding, load_string_graph, ppi_embedding  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")


def load_genept():
    pk = glob.glob(os.path.join(RAW, "GenePT*", "**", "*ada*.pickle"), recursive=True)
    if not pk:
        z = os.path.join(RAW, "GenePT_emebdding_v2.zip")
        if os.path.exists(z) and zipfile.is_zipfile(z):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(os.path.join(RAW, "GenePT"))
            pk = glob.glob(os.path.join(RAW, "GenePT*", "**", "*ada*.pickle"), recursive=True)
    if not pk:
        return None
    with open(pk[0], "rb") as f:
        emb = pickle.load(f)
    print(f"  GenePT: {len(emb)} genes from {os.path.basename(pk[0])}")
    return emb


def main():
    datasets = [p.replace("_pseudobulk.npz", "") for p in sorted(os.listdir(PROC)) if p.endswith("_pseudobulk.npz")]
    pds = [PerturbData.load(os.path.join(PROC, f"{d}_pseudobulk.npz")) for d in datasets]
    query = list(dict.fromkeys(g for p in pds for g in p.perts))
    measured = list(dict.fromkeys(g for p in pds for g in p.genes))
    print(f"datasets: {datasets}; {len(query)} perturbed genes")

    links = os.path.join(RAW, "string", "9606.protein.links.v12.0.txt.gz")
    info = os.path.join(RAW, "string", "9606.protein.info.v12.0.txt.gz")
    _, string_names = load_string_graph(links, info, min_score=900)
    extra = [g for g in dict.fromkeys(list(string_names) + measured) if g not in set(query)]
    print(f"  universe: {len(query)} perturbed + {len(extra)} extra genes")

    out = {"genes": np.array(query)}
    uni = {"genes": np.array(query + extra)}
    E, m, Ex, mx = go_embedding(query, os.path.join(RAW, "gene2go.pkl"), universe=measured + query, extra_genes=extra)
    out["go"], out["go_mask"] = E, m
    uni["go"], uni["go_mask"] = np.vstack([E, Ex]), np.concatenate([m, mx])
    print(f"  GO  coverage {m.mean():.3f} (universe {uni['go_mask'].mean():.3f})")
    E, m, Ex, mx = ppi_embedding(query, links, info, extra_genes=extra)
    out["ppi"], out["ppi_mask"] = E, m
    uni["ppi"], uni["ppi_mask"] = np.vstack([E, Ex]), np.concatenate([m, mx])
    print(f"  PPI coverage {m.mean():.3f} (universe {uni['ppi_mask'].mean():.3f})")

    gp = load_genept()
    if gp is not None:
        universe = list(dict.fromkeys([g for g in measured + query if g in gp]))
        X = np.array([gp[g] for g in universe], dtype=np.float32)
        pca = PCA(128, random_state=0).fit(X)

        def emb(genes):
            E = np.zeros((len(genes), 128), dtype=np.float32)
            m = np.zeros(len(genes), dtype=bool)
            have = [i for i, g in enumerate(genes) if g in gp]
            if have:
                E[have] = normalize(pca.transform(np.array([gp[genes[i]] for i in have], dtype=np.float32)))
                m[have] = True
            return E, m

        E, m = emb(query)
        Ex, mx = emb(extra)
        out["genept"], out["genept_mask"] = E, m
        uni["genept"], uni["genept_mask"] = np.vstack([E, Ex]), np.concatenate([m, mx])
        print(f"  GenePT coverage {m.mean():.3f} (universe {uni['genept_mask'].mean():.3f})")

    old = os.path.join(PROC, "pert_embeddings.npz")
    if os.path.exists(old):   # reproducibility check against the embeddings used in the benchmark
        z = np.load(old)
        for k in ("go", "ppi", "genept"):
            if k in z.files and k in out:
                print(f"  check {k}: max |diff| vs existing = {np.abs(z[k] - out[k]).max():.2e}")
    np.savez_compressed(old, **out)
    np.savez_compressed(os.path.join(PROC, "gene_universe_embeddings.npz"), **uni)
    print("saved", old, "and gene_universe_embeddings.npz")


if __name__ == "__main__":
    main()
