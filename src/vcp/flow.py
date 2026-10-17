"""Conditional flow matching (CFM) generator of perturbed single cells.

Given a control cell x0 (PCA latent) and a perturbation condition c, integrate the learned
velocity field  dx/dt = v_θ(x, t, c)  from t = 0 to 1 to obtain the counterfactual state x1
of *that same cell* after the perturbation.

Training (Lipman et al. 2023; Tong et al. 2024, OT-CFM):
    x0 ~ control cells, x1 ~ cells carrying perturbation p, paired by mini-batch optimal transport
    x_t = (1 - t) x0 + t x1 + σ ε,      target velocity u = x1 - x0
    L = || v_θ(x_t, t, c_p) - u ||²

Condition c_p: the PerturbFormer-predicted pseudobulk shift of p projected onto the PCA basis.
Hence the flow inherits PerturbFormer's ability to generalise to unseen genes, and adds what a
mean-shift cannot express: changes in variance, shape and sub-population structure.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(1000.0) * torch.arange(half, dtype=torch.float32) / half)
        a = t[:, None] * freqs[None] * 2 * math.pi
        return torch.cat([torch.sin(a), torch.cos(a)], -1)


class FiLMBlock(nn.Module):
    def __init__(self, h: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(h, elementwise_affine=False)
        self.film = nn.Linear(h, 2 * h)
        self.ff = nn.Sequential(nn.Linear(h, 2 * h), nn.SiLU(), nn.Dropout(dropout), nn.Linear(2 * h, h))
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, cond):
        scale, shift = self.film(cond).chunk(2, -1)
        return x + self.ff(self.norm(x) * (1 + scale) + shift)


class VectorField(nn.Module):
    def __init__(self, d: int, c_dim: int, hidden: int = 384, n_blocks: int = 4, dropout: float = 0.0):
        super().__init__()
        self.t_emb = nn.Sequential(TimeEmbedding(32), nn.Linear(32, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.c_emb = nn.Sequential(nn.Linear(c_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.inp = nn.Linear(d + c_dim, hidden)
        self.blocks = nn.ModuleList([FiLMBlock(hidden, dropout) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, d))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, x, t, c):
        cond = self.t_emb(t) + self.c_emb(c)
        h = self.inp(torch.cat([x, c], -1))
        for b in self.blocks:
            h = b(h, cond)
        return self.out(h)


class FlowModel:
    """Thin wrapper: training loop, ODE sampler, scaling of the latent space."""

    def __init__(self, d: int, c_dim: int, hidden: int = 256, n_blocks: int = 3, sigma: float = 0.05,
                 seed: int = 0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.net = VectorField(d, c_dim, hidden, n_blocks)
        self.sigma = sigma
        self.scale = 1.0
        self.c_scale = 1.0

    # ------------------------------------------------------------------------------------
    def fit(self, Z: np.ndarray, labels: np.ndarray, ctrl_label: int, cond: dict, steps: int = 8000,
            perts_per_batch: int = 12, cells_per_pert: int = 48, lr: float = 1e-3, verbose: bool = True):
        """cond: {pert_label: condition vector (c_dim,)} for training perturbations."""
        ctrl_idx = np.nonzero(labels == ctrl_label)[0]
        by_pert = {p: np.nonzero(labels == p)[0] for p in cond}
        keys = np.array(list(cond))
        self.scale = float(Z[ctrl_idx].std())
        C = np.stack([cond[p] for p in keys])
        self.c_scale = float(np.abs(C).std()) + 1e-8
        Zt = torch.tensor(Z / self.scale, dtype=torch.float32)
        Ct = {p: torch.tensor(cond[p] / self.c_scale, dtype=torch.float32) for p in keys}
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
        self.net.train()
        n = cells_per_pert
        hist = []
        for step in range(steps):
            ps = self.rng.choice(keys, perts_per_batch, replace=False)
            x0s, x1s, cs = [], [], []
            for p in ps:
                i1 = self.rng.choice(by_pert[p], n, replace=len(by_pert[p]) < n)
                i0 = self.rng.choice(ctrl_idx, n, replace=False)
                x0, x1 = Zt[i0], Zt[i1]
                cost = torch.cdist(x0, x1).pow(2).numpy()
                r, c = linear_sum_assignment(cost)            # mini-batch exact OT coupling
                x0s.append(x0[r])
                x1s.append(x1[c])
                cs.append(Ct[p].expand(n, -1))
            x0, x1, cc = torch.cat(x0s), torch.cat(x1s), torch.cat(cs)
            t = torch.rand(len(x0))
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1 + self.sigma * torch.randn_like(x0)
            loss = ((self.net(xt, t, cc) - (x1 - x0)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step()
            sched.step()
            hist.append(loss.item())
            if verbose and (step + 1) % 2000 == 0:
                print(f"    flow step {step + 1}/{steps}  loss {np.mean(hist[-500:]):.4f}")
        self.net.eval()
        return hist

    @torch.no_grad()
    def transport(self, x0: np.ndarray, c: np.ndarray, n_steps: int = 20) -> np.ndarray:
        """Integrate the ODE (midpoint rule) from control state x0 under condition c."""
        x = torch.tensor(x0 / self.scale, dtype=torch.float32)
        cc = torch.tensor(np.broadcast_to(c / self.c_scale, (len(x0), len(c))).copy(), dtype=torch.float32)
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((len(x),), k * dt)
            v1 = self.net(x, t, cc)
            v2 = self.net(x + 0.5 * dt * v1, t + 0.5 * dt, cc)
            x = x + dt * v2
        return x.numpy() * self.scale

    def state_dict(self):
        return {"net": self.net.state_dict(), "scale": self.scale, "c_scale": self.c_scale, "sigma": self.sigma}

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd["net"])
        self.scale, self.c_scale, self.sigma = sd["scale"], sd["c_scale"], sd["sigma"]
        self.net.eval()


def decode(z: np.ndarray, components: np.ndarray, ctrl_mean: np.ndarray, delta_full: np.ndarray | None = None):
    """PCA latent -> log-normalised expression.

    If `delta_full` (predicted pseudobulk shift in gene space) is given, the part of it that is
    orthogonal to the PCA subspace is added back, so gene-level effects such as the knockdown of
    the target gene itself (invisible in 100 PCs) are retained.
    """
    x = ctrl_mean[None] + z @ components
    if delta_full is not None:
        resid = delta_full - (delta_full @ components.T) @ components
        x = x + resid[None]
    return np.maximum(x, 0.0)
