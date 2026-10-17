"""Knowledge-fusion perturbation network (supervised, pseudobulk level).

    perturbation gene g ──► [GO emb] [PPI emb] [GenePT emb] [response emb]   (prior knowledge)
                                 │         │          │           │
                              per-source projection + learned "missing" token
                                 └──── fusion: Transformer encoder over source tokens ────┘
                                                 │ (CLS token; attention = interpretability)
                                         MLP decoder ─► K coefficients
                                                 │
            Δ̂ = μ_train  +  coefficients · B_trainᵀ  +  small full-rank residual
                (mean shift)   (PCA basis of training responses)

Loss = DE-weighted MSE on Δ  +  λ_dir · logistic loss on the up/down direction of
significant DE genes  (multi-task: expression change + DEG direction).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SourceTokenizer(nn.Module):
    def __init__(self, dims: dict, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.names = list(dims)
        self.proj = nn.ModuleDict({
            n: nn.Sequential(nn.Linear(d, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
            for n, d in dims.items()
        })
        self.missing = nn.Parameter(torch.zeros(len(dims), d_model))
        self.src_emb = nn.Parameter(torch.randn(len(dims), d_model) * 0.02)

    def forward(self, feats: dict, masks: dict):
        toks = []
        for i, n in enumerate(self.names):
            h = self.proj[n](feats[n])
            m = masks[n].unsqueeze(-1).float()
            toks.append(m * h + (1 - m) * self.missing[i] + self.src_emb[i])
        return torch.stack(toks, 1)          # (B, S, d)


class PerturbNet(nn.Module):
    def __init__(self, dims: dict, n_genes: int, basis: np.ndarray, mu: np.ndarray,
                 d_model: int = 128, fusion: str = "attn", n_layers: int = 2, dropout: float = 0.1,
                 residual: bool = True):
        super().__init__()
        self.fusion = fusion
        self.tok = SourceTokenizer(dims, d_model, dropout)
        S = len(dims)
        if fusion == "attn":
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=2 * d_model,
                                               dropout=dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
            h_dim = d_model
        else:
            h_dim = d_model * S
        k = basis.shape[1]
        self.dec = nn.Sequential(nn.Linear(h_dim, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, k))
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.float32))   # (G, k)
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.residual = nn.Linear(h_dim, n_genes) if residual else None
        if residual:
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)

    def encode(self, feats, masks, return_attn: bool = False):
        x = self.tok(feats, masks)
        if self.fusion != "attn":
            return x.flatten(1), None
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1)
        attn = None
        if return_attn:   # attention of CLS over sources, last layer (manual pass)
            h = x
            for i, layer in enumerate(self.enc.layers):
                if i == len(self.enc.layers) - 1:
                    q = layer.norm1(h)
                    _, attn = layer.self_attn(q, q, q, need_weights=True, average_attn_weights=True)
                h = layer(h)
            return h[:, 0], attn[:, 0, 1:]
        return self.enc(x)[:, 0], None

    def forward(self, feats, masks, return_attn: bool = False):
        h, attn = self.encode(feats, masks, return_attn)
        out = self.mu + self.dec(h) @ self.basis.T
        if self.residual is not None:
            out = out + self.residual(h)
        return (out, attn) if return_attn else out


def _to_t(d: dict, idx=None):
    return {k: torch.tensor(v if idx is None else v[idx], dtype=torch.float32) for k, v in d.items()}


def train_perturbnet(feats: dict, masks: dict, Y: np.ndarray, W: np.ndarray, S: np.ndarray,
                     tr: np.ndarray, va: np.ndarray, basis_k: int = 64, fusion: str = "attn",
                     d_model: int = 128, epochs: int = 400, lr: float = 1e-3, wd: float = 1e-4,
                     lam_dir: float = 0.5, batch: int = 64, seed: int = 0, patience: int = 60,
                     verbose: bool = False):
    """Train on perturbations `tr`, early-stop on `va`.

    Y: (P, G) true deltas; W: (P, G) per-gene loss weights (DE emphasis);
    S: (P, G) in {-1, 0, +1}: direction label of significant DE genes (0 = ignore).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = Y[tr].mean(0)
    Yc = Y[tr] - mu
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    basis = Vt[:basis_k].T                                          # (G, k)
    dims = {k: v.shape[1] for k, v in feats.items()}
    model = PerturbNet(dims, Y.shape[1], basis, mu, d_model=d_model, fusion=fusion)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Ft, Mt = _to_t(feats), _to_t(masks)
    Yt, Wt, St = (torch.tensor(a, dtype=torch.float32) for a in (Y, W, S))
    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            f = {k: v[b] for k, v in Ft.items()}
            m = {k: v[b] for k, v in Mt.items()}
            # source dropout: randomly hide one knowledge source -> robustness to missing priors
            if len(m) > 1:
                drop = np.random.randint(len(m))
                key = list(m)[drop]
                keep = (torch.rand(len(b)) > 0.25).float()
                m[key] = m[key] * keep
            pred = model(f, m)
            mse = (Wt[b] * (pred - Yt[b]) ** 2).sum(1) / Wt[b].sum(1)
            dmask = (St[b] != 0).float()
            dir_loss = (F.softplus(-St[b] * pred / 0.05) * dmask).sum(1) / dmask.sum(1).clamp(min=1)
            loss = (mse + lam_dir * 0.01 * dir_loss).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model({k: v[va] for k, v in Ft.items()}, {k: v[va] for k, v in Mt.items()}).numpy()
        # validation score: mean Pearson on DE-weighted genes
        yv = Y[va]
        top = np.argsort(-W[va], 1)[:, :20]
        a = np.take_along_axis(pv, top, 1)
        c = np.take_along_axis(yv, top, 1)
        a = a - a.mean(1, keepdims=True)
        c = c - c.mean(1, keepdims=True)
        score = float(np.mean((a * c).sum(1) / (np.sqrt((a * a).sum(1) * (c * c).sum(1)) + 1e-9)))
        score -= float(((pv - yv) ** 2).mean()) * 10
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 20 == 0:
            print(f"    ep {ep:3d} loss {loss.item():.4f} val {score:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_perturbnet(model, feats, masks, idx, return_attn=False):
    f = {k: torch.tensor(v[idx], dtype=torch.float32) for k, v in feats.items()}
    m = {k: torch.tensor(v[idx], dtype=torch.float32) for k, v in masks.items()}
    if return_attn:
        out, attn = model(f, m, return_attn=True)
        return out.numpy(), (attn.numpy() if attn is not None else None)
    return model(f, m).numpy()


# ========================================================================================
# PerturbFormer: retrieval-augmented cross-attention over known perturbations
# ========================================================================================
class PerturbFormer(nn.Module):
    """Predict an unseen perturbation by *attending to perturbations we have already measured*.

        query  q_i = Enc(prior knowledge of gene i)        (GO / PPI / GenePT / response tokens)
        keys   k_j = Enc(prior knowledge of training gene j)
        values v_j = measured response Δ_j  (G genes)

        Δ̂_i = Σ_h  g_h · softmax_j( <W_q^h q_i, W_k^h k_j> / (√d τ_h) ) · Δ_j      (retrieval)
             + μ + dec(q_i) · B^T                                                (parametric)

    The attention weights a_ij are directly interpretable: "knocking down i is predicted to
    look like knocking down j1, j2, ..." (e.g. other subunits of the same complex).
    """

    def __init__(self, dims: dict, n_genes: int, basis: np.ndarray, mu: np.ndarray,
                 d_model: int = 128, n_heads: int = 4, fusion: str = "attn", dropout: float = 0.1):
        super().__init__()
        self.backbone = PerturbNet(dims, n_genes, basis, mu, d_model=d_model, fusion=fusion,
                                   dropout=dropout, residual=False)
        h_dim = d_model if fusion == "attn" else d_model * len(dims)
        self.n_heads, self.d_head = n_heads, 32
        self.q = nn.Linear(h_dim, n_heads * self.d_head)
        self.k = nn.Linear(h_dim, n_heads * self.d_head)
        self.log_tau = nn.Parameter(torch.zeros(n_heads))
        self.gate = nn.Linear(h_dim, n_heads + 1)      # mixture over heads + parametric branch

    def attend(self, hq, hk, exclude_self: torch.Tensor | None = None):
        B, M = len(hq), len(hk)
        q = self.q(hq).view(B, self.n_heads, self.d_head)
        k = self.k(hk).view(M, self.n_heads, self.d_head)
        logits = torch.einsum("bhd,mhd->bhm", q, k) / np.sqrt(self.d_head)
        logits = logits / torch.exp(self.log_tau)[None, :, None]
        if exclude_self is not None:
            logits = logits.masked_fill(exclude_self[:, None, :], -1e9)
        return torch.softmax(logits, -1)                               # (B, H, M)

    def forward(self, fq, mq, fk, mk, Yk, exclude_self=None, return_attn=False):
        hq, _ = self.backbone.encode(fq, mq)
        hk, _ = self.backbone.encode(fk, mk)
        a = self.attend(hq, hk, exclude_self)                          # (B, H, M)
        retr = torch.einsum("bhm,mg->bhg", a, Yk)                       # (B, H, G)
        param = self.backbone.mu + self.backbone.dec(hq) @ self.backbone.basis.T    # (B, G)
        g = torch.softmax(self.gate(hq), -1)                           # (B, H+1)
        out = (g[:, :-1, None] * retr).sum(1) + g[:, -1:] * param
        if return_attn:
            return out, (g[:, :-1, None] * a).sum(1), g             # combined attention (B, M)
        return out


def train_perturbformer(feats: dict, masks: dict, Y: np.ndarray, W: np.ndarray, S: np.ndarray,
                        tr: np.ndarray, va: np.ndarray, basis_k: int = 32, d_model: int = 128,
                        fusion: str = "attn", epochs: int = 300, lr: float = 1e-3, wd: float = 1e-3,
                        lam_dir: float = 0.5, batch: int = 64, seed: int = 0, patience: int = 40,
                        dropout: float = 0.1, kernel: bool = True, topk: int = 20, n_heads: int = 2,
                        verbose: bool = False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = Y[tr].mean(0)
    _, _, Vt = np.linalg.svd(Y[tr] - mu, full_matrices=False)
    dims = {k: v.shape[1] for k, v in feats.items()}
    if kernel:
        model = KernelPerturbFormer(dims, Y.shape[1], Vt[:basis_k].T, mu, d_model=d_model, fusion=fusion,
                                    dropout=dropout, topk=topk, n_heads=n_heads)
    else:
        model = PerturbFormer(dims, Y.shape[1], Vt[:basis_k].T, mu, d_model=d_model, fusion=fusion,
                              dropout=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Ft, Mt = _to_t(feats), _to_t(masks)
    Yt, Wt, St = (torch.tensor(a, dtype=torch.float32) for a in (Y, W, S))
    trt = torch.tensor(tr)
    mem_f = {k: v[trt] for k, v in Ft.items()}
    mem_m = {k: v[trt] for k, v in Mt.items()}
    pos = {int(p): i for i, p in enumerate(tr)}
    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            excl = torch.zeros(len(b), len(tr), dtype=torch.bool)
            excl[torch.arange(len(b)), torch.tensor([pos[int(x)] for x in b])] = True
            f = {k: v[b] for k, v in Ft.items()}
            m = {k: v[b].clone() for k, v in Mt.items()}
            if len(m) > 1:   # knowledge-source dropout on the query side
                key = list(m)[np.random.randint(len(m))]
                m[key] = m[key] * (torch.rand(len(b)) > 0.25).float()
            pred = model(f, m, mem_f, mem_m, Yt[trt], exclude_self=excl)
            mse = (Wt[b] * (pred - Yt[b]) ** 2).sum(1) / Wt[b].sum(1)
            dmask = (St[b] != 0).float()
            dir_loss = (F.softplus(-St[b] * pred / 0.05) * dmask).sum(1) / dmask.sum(1).clamp(min=1)
            loss = (mse + lam_dir * 0.01 * dir_loss).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model({k: v[va] for k, v in Ft.items()}, {k: v[va] for k, v in Mt.items()},
                       mem_f, mem_m, Yt[trt]).numpy()
        yv = Y[va]
        top = np.argsort(-W[va], 1)[:, :20]
        a_ = np.take_along_axis(pv, top, 1)
        c_ = np.take_along_axis(yv, top, 1)
        a_ = a_ - a_.mean(1, keepdims=True)
        c_ = c_ - c_.mean(1, keepdims=True)
        score = float(np.mean((a_ * c_).sum(1) / (np.sqrt((a_ * a_).sum(1) * (c_ * c_).sum(1)) + 1e-9)))
        score -= float(((pv - yv) ** 2).mean()) * 10
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose and ep % 10 == 0:
            print(f"    ep {ep:3d} loss {loss.item():.4f} val {score:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_perturbformer(model, feats, masks, Y, mem_idx, q_idx, return_attn=False):
    """Predict perturbations q_idx using measured perturbations mem_idx as memory."""
    Ft, Mt = _to_t(feats), _to_t(masks)
    mi, qi = torch.tensor(mem_idx), torch.tensor(q_idx)
    out = model({k: v[qi] for k, v in Ft.items()}, {k: v[qi] for k, v in Mt.items()},
                {k: v[mi] for k, v in Ft.items()}, {k: v[mi] for k, v in Mt.items()},
                torch.tensor(Y[mem_idx], dtype=torch.float32), return_attn=return_attn)
    if return_attn:
        o, a, g = out
        return o.numpy(), a.numpy(), g.numpy()
    return out.numpy()


# ========================================================================================
# PerturbFormer v2: knowledge-kernel cross-attention (default model)
# ========================================================================================
class KernelPerturbFormer(nn.Module):
    """Cross-attention whose logits are built from prior-knowledge similarities.

        logit_h(i, j) = Σ_s w_hs(i) m_s(i) m_s(j) cos(A_s e_s(i), A_s e_s(j))
                        / Σ_s w_hs(i) m_s(i) m_s(j)  / τ_h                       (top-k sparse)

    A_s is a learnable linear map initialised to the identity (so at initialisation the model
    *is* a multi-source cosine kNN), w_hs(i) are query-dependent knowledge-source weights
    produced from the fused query encoding (interpretable: which prior was trusted), and τ_h
    are per-head temperatures. A parametric low-rank branch is mixed in through a gate.
    """

    def __init__(self, dims: dict, n_genes: int, basis: np.ndarray, mu: np.ndarray,
                 d_model: int = 128, n_heads: int = 2, topk: int = 20, dropout: float = 0.1,
                 fusion: str = "attn"):
        super().__init__()
        self.names = list(dims)
        self.backbone = PerturbNet(dims, n_genes, basis, mu, d_model=d_model, fusion=fusion,
                                   dropout=dropout, residual=False)
        h_dim = d_model if fusion == "attn" else d_model * len(dims)
        self.A = nn.ModuleDict({n: nn.Linear(d, d, bias=False) for n, d in dims.items()})
        for n in self.names:
            nn.init.eye_(self.A[n].weight)
        self.n_heads, self.topk = n_heads, topk
        self.src_w = nn.Linear(h_dim, n_heads * len(dims))
        nn.init.zeros_(self.src_w.weight)
        nn.init.zeros_(self.src_w.bias)
        self.log_tau = nn.Parameter(torch.log(torch.tensor([0.1, 0.05][:n_heads] + [0.1] * max(0, n_heads - 2))))
        self.gate = nn.Linear(h_dim, n_heads + 1)
        nn.init.zeros_(self.gate.weight)
        with torch.no_grad():
            self.gate.bias.copy_(torch.tensor([1.0] * n_heads + [-1.0]))

    def sims(self, fq, mq, fk, mk):
        """Per-source cosine similarity (S, B, M) and joint availability mask."""
        S, A = [], []
        for n in self.names:
            zq = F.normalize(self.A[n](fq[n]), dim=-1)
            zk = F.normalize(self.A[n](fk[n]), dim=-1)
            S.append(zq @ zk.T)
            A.append(mq[n][:, None] * mk[n][None, :])
        return torch.stack(S), torch.stack(A)

    def forward(self, fq, mq, fk, mk, Yk, exclude_self=None, return_attn=False):
        hq, _ = self.backbone.encode(fq, mq)
        sim, avail = self.sims(fq, mq, fk, mk)                                # (S, B, M)
        w = F.softplus(self.src_w(hq).view(len(hq), self.n_heads, len(self.names)) + 0.5413)  # init 1.0
        num = torch.einsum("bhs,sbm->bhm", w, sim * avail)
        den = torch.einsum("bhs,sbm->bhm", w, avail) + 1e-6
        logits = (num / den) / torch.exp(self.log_tau)[None, :, None]
        if exclude_self is not None:
            logits = logits.masked_fill(exclude_self[:, None, :], -1e9)
        if self.topk and self.topk < logits.shape[-1]:
            kth = logits.topk(self.topk, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, -1e9)
        a = torch.softmax(logits, -1)                                          # (B, H, M)
        retr = torch.einsum("bhm,mg->bhg", a, Yk)
        param = self.backbone.mu + self.backbone.dec(hq) @ self.backbone.basis.T
        g = torch.softmax(self.gate(hq), -1)
        out = (g[:, :-1, None] * retr).sum(1) + g[:, -1:] * param
        if return_attn:
            src = (w * g[:, :-1, None]).sum(1)                                 # (B, S) source weights
            return out, (g[:, :-1, None] * a).sum(1), torch.cat([g, src / src.sum(1, keepdim=True)], 1)
        return out


@torch.no_grad()
def predict_perturbformer_loo(model, feats, masks, Y, mem_idx):
    """Leave-self-out predictions for the memory perturbations themselves (each perturbation is
    predicted from all *other* measured perturbations) - realistic conditions for training the
    downstream single-cell flow model."""
    Ft, Mt = _to_t(feats), _to_t(masks)
    mi = torch.tensor(mem_idx)
    f = {k: v[mi] for k, v in Ft.items()}
    m = {k: v[mi] for k, v in Mt.items()}
    excl = torch.eye(len(mem_idx), dtype=torch.bool)
    return model(f, m, f, m, torch.tensor(Y[mem_idx], dtype=torch.float32), exclude_self=excl).numpy()
