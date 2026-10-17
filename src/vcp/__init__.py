"""VirtualCell-Perturb: predicting single-cell transcriptional responses to gene knockdown.

Modules
-------
data        Perturb-seq loading, pseudobulk statistics, DEG calling, splits
embeddings  Prior-knowledge perturbation encodings (GO / STRING PPI / co-expression)
baselines   No-change, mean-shift, linear (Ahlmann-Eltze et al.), kNN
models      Knowledge-fusion perturbation network (MLP / attention fusion)
flow        Conditional flow matching generator on single-cell latent space
metrics     Evaluation metrics (delta correlation, DEG direction, distribution distances)
predictor   High-level API: VirtualCell.predict(cells, gene)
"""

__version__ = "1.0.0"
