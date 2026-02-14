"""lob_model.py

Neural generators for Level-2 (L2) limit order book *parameter* sequences.

This file contains ONLY model code:
- LOBConfig
- MLP
- BiFlowLOB (rectified-flow / flow-matching baseline)
- BiMeanFlowLOB (BiFlow + MeanFlow hybrid; 1-step sampling; imbalance + rollout losses)

Data loading, feature mapping, standardization, and metrics live in `lob_datasets.py`.
Training/evaluation loops live in `lob_train_val.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Config
# -----------------------------

@dataclass
class LOBConfig:
    # L2 shape
    levels: int = 10
    history_len: int = 50

    # Model
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.1

    # Training
    batch_size: int = 64
    steps: int = 20_000
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Conditioning (optional)
    cond_dim: int = 0
    cfg_dropout: float = 0.1

    # Numerics
    eps: float = 1e-8
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- BiMeanFlow core losses ----
    lambda_mean: float = 1.0
    lambda_xrec: float = 1.0
    lambda_zcycle: float = 0.25
    lambda_prior: float = 0.1

    # ---- Realism/drift regularizers ----
    lambda_imb: float = 0.2           # imbalance moment matching weight (raw space)
    rollout_K: int = 4                # short unroll length (0 disables)
    lambda_rollout: float = 0.25      # rollout loss weight
    rollout_detach_ctx: bool = True   # detach generated x when feeding back to ctx

    # MeanFlow path (implemented for linear only here)
    path: str = "linear"

    # Optional: clip generated normalized params to limit AR drift (0 disables)
    sample_clip: float = 0.0

    # Dataset standardization (Mode A)
    standardize: bool = True

    @property
    def state_dim(self) -> int:
        return 4 * self.levels


# -----------------------------
# Small network utilities
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Baseline: rectified-flow / flow-matching
# -----------------------------

class BiFlowLOB(nn.Module):
    """Conditional rectified-flow model for L2 params (baseline)."""

    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.state_dim

        self.ctx_rnn = nn.LSTM(
            input_size=D,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        self.t_mlp = MLP(1, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0)

        if cfg.cond_dim > 0:
            self.cond_mlp = MLP(cfg.cond_dim, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0)
        else:
            self.cond_mlp = None

        in_dim = D + cfg.hidden_dim + cfg.hidden_dim + (cfg.hidden_dim if cfg.cond_dim > 0 else 0)
        self.v_net = MLP(in_dim, cfg.hidden_dim, D, dropout=cfg.dropout)

    def encode_ctx(self, ctx: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.ctx_rnn(ctx)
        return h_n[-1]  # [B, hidden_dim]

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        ctx: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        cfg_drop: bool = False,
    ) -> torch.Tensor:
        B = x_t.shape[0]
        ctx_emb = self.encode_ctx(ctx)
        t_emb = self.t_mlp(t)
        parts = [x_t, ctx_emb, t_emb]

        if self.cond_mlp is not None:
            assert cond is not None
            if cfg_drop:
                drop_mask = (torch.rand(B, device=cond.device) < self.cfg.cfg_dropout).float().unsqueeze(1)
                cond_in = cond * (1.0 - drop_mask)
            else:
                cond_in = cond
            parts.append(self.cond_mlp(cond_in))

        return self.v_net(torch.cat(parts, dim=1))

    def fm_loss(self, x: torch.Tensor, ctx: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, _ = x.shape
        z = torch.randn_like(x)
        t = torch.rand(B, 1, device=x.device)
        x_t = (1.0 - t) * z + t * x
        v_target = x - z
        v_hat = self.forward(x_t, t, ctx, cond=cond, cfg_drop=True)
        return F.mse_loss(v_hat, v_target)

    @torch.no_grad()
    def sample(
        self,
        ctx: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        steps: int = 32,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        device = ctx.device
        B = ctx.shape[0]
        D = self.cfg.state_dim
        x = torch.randn(B, D, device=device)

        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((B, 1), k * dt, device=device)
            if (self.cond_mlp is None) or (cond is None) or (guidance_scale == 1.0):
                v = self.forward(x, t, ctx, cond=cond, cfg_drop=False)
            else:
                v_cond = self.forward(x, t, ctx, cond=cond, cfg_drop=False)
                v_uncond = self.forward(x, t, ctx, cond=torch.zeros_like(cond), cfg_drop=False)
                v = v_uncond + guidance_scale * (v_cond - v_uncond)
            x = x + dt * v

        if self.cfg.sample_clip and self.cfg.sample_clip > 0:
            x = x.clamp(-self.cfg.sample_clip, self.cfg.sample_clip)
        return x


# -----------------------------
# Hybrid: BiFlow + MeanFlow (paired x<->z + 1-step sampling)
# -----------------------------

class BiMeanFlowLOB(nn.Module):
    """BiFlow + MeanFlow hybrid for L2 params.

    - Forward net f_psi(x, ctx, cond) -> z_hat  (paired)
    - Reverse net u_theta(x_t, t, ctx, cond) -> mean displacement
    - One-step sampling: x = z + u(z, t=0, ctx, cond)

    Extras:
    - imbalance moment matching loss (raw space)
    - short rollout consistency loss (normalized space)
    """

    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.state_dim

        self.ctx_rnn = nn.LSTM(
            input_size=D,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        self.t_mlp = MLP(1, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0)

        if cfg.cond_dim > 0:
            self.cond_mlp = MLP(cfg.cond_dim, cfg.hidden_dim, cfg.hidden_dim, dropout=0.0)
        else:
            self.cond_mlp = None

        f_in = D + cfg.hidden_dim + (cfg.hidden_dim if cfg.cond_dim > 0 else 0)
        u_in = D + cfg.hidden_dim + cfg.hidden_dim + (cfg.hidden_dim if cfg.cond_dim > 0 else 0)

        self.f_net = MLP(f_in, cfg.hidden_dim, D, dropout=cfg.dropout)
        self.u_net = MLP(u_in, cfg.hidden_dim, D, dropout=cfg.dropout)

        # Standardizer buffers (for imbalance loss in raw space)
        self.register_buffer("scaler_mu", torch.zeros(D), persistent=False)
        self.register_buffer("scaler_sigma", torch.ones(D), persistent=False)
        self.has_scaler: bool = False

    def set_scaler(self, mu: np.ndarray, sigma: np.ndarray):
        """Set (mu, sigma) used for denormalization inside the model (imbalance loss)."""
        mu_t = torch.from_numpy(mu.astype(np.float32))
        sig_t = torch.from_numpy(sigma.astype(np.float32))
        if mu_t.numel() != self.scaler_mu.numel():
            raise ValueError("Scaler dimension mismatch.")
        self.scaler_mu.copy_(mu_t)
        self.scaler_sigma.copy_(sig_t)
        self.has_scaler = True

    def encode_ctx(self, ctx: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.ctx_rnn(ctx)
        return h_n[-1]

    def _cond_emb(self, cond: Optional[torch.Tensor], B: int, device: torch.device, cfg_drop: bool) -> Optional[torch.Tensor]:
        if self.cond_mlp is None:
            return None
        assert cond is not None
        if cfg_drop:
            drop_mask = (torch.rand(B, device=device) < self.cfg.cfg_dropout).float().unsqueeze(1)
            cond = cond * (1.0 - drop_mask)
        return self.cond_mlp(cond)

    def f_forward(self, x: torch.Tensor, ctx: torch.Tensor, cond: Optional[torch.Tensor], cfg_drop: bool = False) -> torch.Tensor:
        B = x.shape[0]
        ctx_emb = self.encode_ctx(ctx)
        parts = [x, ctx_emb]
        ce = self._cond_emb(cond, B, x.device, cfg_drop)
        if ce is not None:
            parts.append(ce)
        return self.f_net(torch.cat(parts, dim=1))

    def u_forward(self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor, cond: Optional[torch.Tensor], cfg_drop: bool = False) -> torch.Tensor:
        B = x_t.shape[0]
        ctx_emb = self.encode_ctx(ctx)
        t_emb = self.t_mlp(t)
        parts = [x_t, ctx_emb, t_emb]
        ce = self._cond_emb(cond, B, x_t.device, cfg_drop)
        if ce is not None:
            parts.append(ce)
        return self.u_net(torch.cat(parts, dim=1))

    # ---- helpers for imbalance loss ----

    def _denorm_torch(self, x_norm: torch.Tensor) -> torch.Tensor:
        mu = self.scaler_mu.to(device=x_norm.device, dtype=x_norm.dtype)
        sig = self.scaler_sigma.to(device=x_norm.device, dtype=x_norm.dtype)
        return x_norm * sig + mu

    def _abs_imb_from_raw(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Compute |imb| from RAW params using best-level sizes."""
        L = self.cfg.levels
        ask_logv0 = 2 * L        # start of log_ask_v
        bid_logv0 = 3 * L        # start of log_bid_v

        # stabilize exp()
        ask_logv1 = x_raw[:, ask_logv0].clamp(-10.0, 10.0)
        bid_logv1 = x_raw[:, bid_logv0].clamp(-10.0, 10.0)
        ask_v1 = torch.exp(ask_logv1)
        bid_v1 = torch.exp(bid_logv1)
        imb = (bid_v1 - ask_v1) / (bid_v1 + ask_v1 + 1e-8)
        return imb.abs()

    # ---- main loss ----

    def loss(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        x_future: Optional[torch.Tensor] = None,  # [B,K,D] (normalized) optional
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        if cfg.path != "linear":
            raise NotImplementedError("This implementation supports path='linear' only.")

        B, D = x.shape
        device = x.device

        # ----- Forward pairing -----
        z_hat = self.f_forward(x, ctx, cond=cond, cfg_drop=True)

        # Prior moment match for z_hat ~ N(0, I)
        z_mean = z_hat.mean(dim=0)
        z_var = z_hat.var(dim=0, unbiased=False)
        loss_prior = (z_mean.pow(2).mean() + (z_var - 1.0).pow(2).mean())

        # ----- MeanFlow displacement loss (linear path) -----
        t = torch.rand(B, 1, device=device)
        z_for_path = z_hat.detach()
        x_t = (1.0 - t) * z_for_path + t * x

        u_target = x - z_for_path
        u_hat = self.u_forward(x_t, t, ctx, cond=cond, cfg_drop=True)
        loss_mean = F.mse_loss(u_hat, u_target)

        # ----- One-step x reconstruction via paired z_hat -----
        t0 = torch.zeros(B, 1, device=device)
        u0_hat = self.u_forward(z_hat, t0, ctx, cond=cond, cfg_drop=False)
        x_rec = z_hat + u0_hat
        loss_xrec = F.mse_loss(x_rec, x)

        # ----- z cycle: z0 -> x_fake -> z_rec -----
        z0 = torch.randn(B, D, device=device)
        u0 = self.u_forward(z0, t0, ctx, cond=cond, cfg_drop=False)
        x_fake = z0 + u0
        z_rec = self.f_forward(x_fake, ctx, cond=cond, cfg_drop=False)
        loss_zcycle = F.mse_loss(z_rec, z0)

        # ----- Imbalance moment matching (RAW space) -----
        loss_imb = torch.zeros((), device=device)
        if (cfg.lambda_imb > 0) and self.has_scaler:
            x_raw = self._denorm_torch(x)
            xfake_raw = self._denorm_torch(x_fake)
            a_real = self._abs_imb_from_raw(x_raw)
            a_fake = self._abs_imb_from_raw(xfake_raw)

            mean_real = a_real.mean()
            mean_fake = a_fake.mean()
            std_real = a_real.std(unbiased=False)
            std_fake = a_fake.std(unbiased=False)

            loss_imb = (mean_real - mean_fake).abs() + (std_real - std_fake).abs()

        # ----- Short rollout consistency loss (normalized space) -----
        loss_roll = torch.zeros((), device=device)
        if (cfg.lambda_rollout > 0) and (cfg.rollout_K > 0) and (x_future is not None):
            K = min(cfg.rollout_K, x_future.shape[1])
            ctx_roll = ctx
            losses = []
            for i in range(K):
                z = torch.randn(B, D, device=device)
                u = self.u_forward(z, t0, ctx_roll, cond=cond, cfg_drop=True)
                x_pred = z + u
                losses.append(F.mse_loss(x_pred, x_future[:, i, :]))

                x_to_ctx = x_pred.detach() if cfg.rollout_detach_ctx else x_pred
                ctx_roll = torch.cat([ctx_roll[:, 1:, :], x_to_ctx.unsqueeze(1)], dim=1)
            loss_roll = torch.stack(losses).mean()

        total = (
            cfg.lambda_mean * loss_mean
            + cfg.lambda_xrec * loss_xrec
            + cfg.lambda_zcycle * loss_zcycle
            + cfg.lambda_prior * loss_prior
            + cfg.lambda_imb * loss_imb
            + cfg.lambda_rollout * loss_roll
        )

        logs = {
            "loss_total": float(total.detach().cpu()),
            "loss_mean": float(loss_mean.detach().cpu()),
            "loss_xrec": float(loss_xrec.detach().cpu()),
            "loss_zcycle": float(loss_zcycle.detach().cpu()),
            "loss_prior": float(loss_prior.detach().cpu()),
            "loss_imb": float(loss_imb.detach().cpu()),
            "loss_roll": float(loss_roll.detach().cpu()),
        }
        return total, logs

    @torch.no_grad()
    def sample(
        self,
        ctx: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """One-step sampling (1 NFE): x = z + u(z, t=0)."""
        B = ctx.shape[0]
        D = self.cfg.state_dim
        device = ctx.device
        z = torch.randn(B, D, device=device)
        t0 = torch.zeros(B, 1, device=device)

        if (self.cond_mlp is None) or (cond is None) or (guidance_scale == 1.0):
            u = self.u_forward(z, t0, ctx, cond=cond, cfg_drop=False)
        else:
            u_cond = self.u_forward(z, t0, ctx, cond=cond, cfg_drop=False)
            u_uncond = self.u_forward(z, t0, ctx, cond=torch.zeros_like(cond), cfg_drop=False)
            u = u_uncond + guidance_scale * (u_cond - u_uncond)

        x = z + u
        if self.cfg.sample_clip and self.cfg.sample_clip > 0:
            x = x.clamp(-self.cfg.sample_clip, self.cfg.sample_clip)
        return x
