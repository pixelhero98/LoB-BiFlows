"""demo_model.py

Flow-matching generative model for L2 limit order book (LOB) snapshot sequences.

Key idea
--------
Instead of modeling raw prices/volumes directly (which can violate monotonicity or
positivity), we model an unconstrained parameter vector that decodes into a valid
L2 book:

For L levels (default 10), define per snapshot:
  delta_mid: mid_t - mid_{t-1}
  log_spread: log(ask1 - bid1)
  log_ask_gaps: log(ask_i - ask_{i-1}) for i=2..L
  log_bid_gaps: log(bid_{i-1} - bid_i) for i=2..L
  log_ask_sizes: log(size_ask_i) for i=1..L
  log_bid_sizes: log(size_bid_i) for i=1..L

This yields 2 + (L-1) + (L-1) + L + L = 4L dims (40 when L=10).
Decoding uses exp() to guarantee positivity and enforces price ladder ordering.

Notes
-----
- The model is a rectified-flow / flow-matching style model.
"""

from __future__ import annotations

import math
import os
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

    # Flow matching
    sigma_min: float = 0.0  # keep 0 for rectified flow

    # Conditioning (optional)
    cond_dim: int = 0
    cfg_dropout: float = 0.1
    lambda_cycle: float = 0.0  # keep 0 unless you add cycle-consistency

    # Numerics
    eps: float = 1e-8
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def state_dim(self) -> int:
        return 4 * self.levels


# -----------------------------
# Feature map: valid L2 <-> unconstrained params
# -----------------------------

class L2FeatureMap:
    """Encode/decode between raw L2 snapshots and an unconstrained vector.

    Raw format expected by encode():
      ask_p, ask_v, bid_p, bid_v each shape [T, L]

    Decoding needs an initial mid (scalar) to integrate delta_mid.
    """

    def __init__(self, levels: int = 10, eps: float = 1e-8):
        self.L = int(levels)
        self.eps = float(eps)

    def encode_sequence(
        self,
        ask_p: np.ndarray,
        ask_v: np.ndarray,
        bid_p: np.ndarray,
        bid_v: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (params, mids).

        params: [T, 4L]
        mids:   [T]
        """
        T, L = ask_p.shape
        assert L == self.L
        assert ask_v.shape == (T, L)
        assert bid_p.shape == (T, L)
        assert bid_v.shape == (T, L)

        # mid and spread
        mid = 0.5 * (ask_p[:, 0] + bid_p[:, 0])
        spread = np.clip(ask_p[:, 0] - bid_p[:, 0], self.eps, None)
        log_spread = np.log(spread)

        # delta mid (first is 0)
        delta_mid = np.zeros(T, dtype=np.float32)
        delta_mid[1:] = (mid[1:] - mid[:-1]).astype(np.float32)

        # gaps (positive)
        ask_gaps = np.clip(ask_p[:, 1:] - ask_p[:, :-1], self.eps, None)
        bid_gaps = np.clip(bid_p[:, :-1] - bid_p[:, 1:], self.eps, None)
        log_ask_gaps = np.log(ask_gaps)
        log_bid_gaps = np.log(bid_gaps)

        # sizes (positive)
        ask_v = np.clip(ask_v, self.eps, None)
        bid_v = np.clip(bid_v, self.eps, None)
        log_ask_v = np.log(ask_v)
        log_bid_v = np.log(bid_v)

        params = np.concatenate(
            [
                delta_mid[:, None],
                log_spread[:, None],
                log_ask_gaps,
                log_bid_gaps,
                log_ask_v,
                log_bid_v,
            ],
            axis=1,
        ).astype(np.float32)

        assert params.shape == (T, 4 * L)
        return params, mid.astype(np.float32)

    def decode_sequence(
        self,
        params: np.ndarray,
        init_mid: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decode params -> valid L2 snapshot sequence.

        params: [T, 4L]

        Returns: (ask_p, ask_v, bid_p, bid_v, mids)
          - ask_p/bid_p: [T, L]
          - ask_v/bid_v: [T, L]
          - mids: [T]
        """
        T, D = params.shape
        L = self.L
        assert D == 4 * L

        delta_mid = params[:, 0]
        log_spread = params[:, 1]

        log_ask_gaps = params[:, 2 : 2 + (L - 1)]
        log_bid_gaps = params[:, 2 + (L - 1) : 2 + 2 * (L - 1)]

        log_ask_v = params[:, 2 + 2 * (L - 1) : 2 + 2 * (L - 1) + L]
        log_bid_v = params[:, 2 + 2 * (L - 1) + L :]

        spread = np.exp(log_spread)
        ask_gaps = np.exp(log_ask_gaps)
        bid_gaps = np.exp(log_bid_gaps)
        ask_v = np.exp(log_ask_v)
        bid_v = np.exp(log_bid_v)

        mids = np.zeros(T, dtype=np.float32)
        mids[0] = float(init_mid)
        if T > 1:
            mids[1:] = mids[0] + np.cumsum(delta_mid[1:]).astype(np.float32)

        ask_p = np.zeros((T, L), dtype=np.float32)
        bid_p = np.zeros((T, L), dtype=np.float32)

        ask_p[:, 0] = mids + 0.5 * spread
        bid_p[:, 0] = mids - 0.5 * spread

        for i in range(1, L):
            ask_p[:, i] = ask_p[:, i - 1] + ask_gaps[:, i - 1]
            bid_p[:, i] = bid_p[:, i - 1] - bid_gaps[:, i - 1]

        return ask_p, ask_v.astype(np.float32), bid_p, bid_v.astype(np.float32), mids

    # ---------- FI-2010 helpers ----------

    @staticmethod
    def _split_interleaved(x: np.ndarray, L: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """[ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ...]"""
        ask_p = x[:, 0::4][:, :L]
        ask_v = x[:, 1::4][:, :L]
        bid_p = x[:, 2::4][:, :L]
        bid_v = x[:, 3::4][:, :L]
        return ask_p, ask_v, bid_p, bid_v

    @staticmethod
    def _split_blocks(x: np.ndarray, L: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """[ask_p(1..L), ask_v(1..L), bid_p(1..L), bid_v(1..L)]"""
        ask_p = x[:, :L]
        ask_v = x[:, L : 2 * L]
        bid_p = x[:, 2 * L : 3 * L]
        bid_v = x[:, 3 * L : 4 * L]
        return ask_p, ask_v, bid_p, bid_v

    @staticmethod
    def _valid_ratio(ask_p: np.ndarray, bid_p: np.ndarray) -> float:
        # ask ladder increasing, bid ladder decreasing, spread positive
        spread_ok = (ask_p[:, 0] - bid_p[:, 0]) > 0
        ask_ok = np.all(np.diff(ask_p, axis=1) >= 0, axis=1)
        bid_ok = np.all(np.diff(bid_p, axis=1) <= 0, axis=1)
        ok = spread_ok & ask_ok & bid_ok
        return float(ok.mean())

    def split_l2_from_array(
        self,
        x: np.ndarray,
        layout: str = "auto",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split a 2D array into (ask_p, ask_v, bid_p, bid_v).

        Supports FI-2010-like arrays and many ABIDES exports.

        layout:
          - "interleaved"  : [ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ...]
          - "blocks"       : [ask_p(1..L), ask_v(1..L), bid_p(1..L), bid_v(1..L)]
          - "auto"         : pick the one with best validity ratio
        """
        assert x.ndim == 2 and x.shape[1] >= 4 * self.L
        x40 = x[:, : 4 * self.L]

        if layout == "interleaved":
            return self._split_interleaved(x40, self.L)
        if layout == "blocks":
            return self._split_blocks(x40, self.L)
        if layout != "auto":
            raise ValueError(f"Unknown layout: {layout}")

        a1 = self._split_interleaved(x40, self.L)
        a2 = self._split_blocks(x40, self.L)
        r1 = self._valid_ratio(a1[0], a1[2])
        r2 = self._valid_ratio(a2[0], a2[2])
        return a1 if r1 >= r2 else a2


# -----------------------------
# Datasets
# -----------------------------

class WindowedLOBParamsDataset(torch.utils.data.Dataset):
    """Takes a long params sequence and yields (history, target, meta).

    history: [H, D]
    target:  [D]

    meta is a dict containing:
      - mid_prev: mid at the last step of history (float)
      - init_mid_for_window: mid at t=0 of the window (float)
    """

    def __init__(
        self,
        params: np.ndarray,
        mids: np.ndarray,
        history_len: int,
        stride: int = 1,
        cond: Optional[np.ndarray] = None,
    ):
        assert params.ndim == 2
        assert mids.ndim == 1
        assert len(params) == len(mids)
        self.params = params
        self.mids = mids
        self.H = int(history_len)
        self.stride = int(stride)
        self.cond = cond

        self.start_indices = np.arange(self.H, len(params), self.stride)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int):
        t = int(self.start_indices[idx])
        hist = self.params[t - self.H : t]
        tgt = self.params[t]
        meta = {
            "mid_prev": float(self.mids[t - 1]),
            "init_mid_for_window": float(self.mids[t - self.H]),
        }
        if self.cond is None:
            return (
                torch.from_numpy(hist),
                torch.from_numpy(tgt),
                meta,
            )
        c = self.cond[t]
        return (
            torch.from_numpy(hist),
            torch.from_numpy(tgt),
            torch.from_numpy(c),
            meta,
        )


def load_fi2010_like_array(path: str) -> np.ndarray:
    """Load a FI-2010-like array.

    Supports:
      - .npy (2D array)
      - .npz (expects 'data' or first array)
      - .csv (numeric)
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        x = np.load(path)
    elif ext == ".npz":
        z = np.load(path)
        if "data" in z:
            x = z["data"]
        else:
            # take first key
            k0 = list(z.keys())[0]
            x = z[k0]
    elif ext in (".csv", ".txt"):
        x = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(f"Unsupported file type: {path}")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    return x.astype(np.float32)


def load_abides_l2_npz(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ABIDES L2 snapshots exported as npz.

    Expected keys:
      - times: [T] (optional)
      - bids:  [T, L, 2] where [:,:,0]=price, [:,:,1]=size
      - asks:  [T, L, 2] where [:,:,0]=price, [:,:,1]=size

    This matches the common ABIDES-Markets snapshot representation used in demos.
    """
    z = np.load(path, allow_pickle=True)
    bids = z["bids"].astype(np.float32)
    asks = z["asks"].astype(np.float32)
    times = z["times"].astype(np.float32) if "times" in z else np.arange(len(bids), dtype=np.float32)
    return times, bids, asks


def build_dataset_from_fi2010(
    path: str,
    cfg: LOBConfig,
    layout: str = "auto",
    stride: int = 1,
) -> WindowedLOBParamsDataset:
    fm = L2FeatureMap(cfg.levels, cfg.eps)
    x = load_fi2010_like_array(path)
    ask_p, ask_v, bid_p, bid_v = fm.split_l2_from_array(x, layout=layout)
    params, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)
    return WindowedLOBParamsDataset(params, mids, history_len=cfg.history_len, stride=stride)


def build_dataset_from_abides(
    path_npz: str,
    cfg: LOBConfig,
    stride: int = 1,
) -> WindowedLOBParamsDataset:
    fm = L2FeatureMap(cfg.levels, cfg.eps)
    _, bids, asks = load_abides_l2_npz(path_npz)
    if bids.shape[1] != cfg.levels:
        raise ValueError(f"bids shape {bids.shape} incompatible with levels={cfg.levels}")

    bid_p = bids[:, :, 0]
    bid_v = bids[:, :, 1]
    ask_p = asks[:, :, 0]
    ask_v = asks[:, :, 1]

    params, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)
    return WindowedLOBParamsDataset(params, mids, history_len=cfg.history_len, stride=stride)


# -----------------------------
# Model
# -----------------------------

class BiFlowLOB(nn.Module):
    """Conditional rectified-flow model for L2 params.

    Inputs:
      x_t:  [B, D]
      t:    [B, 1]
      ctx:  [B, H, D] (history params)
      cond: [B, C] optional

    Output:
      v_hat: [B, D]
    """

    def __init__(self, cfg: LOBConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.state_dim

        # Encode history
        self.ctx_rnn = nn.LSTM(
            input_size=D,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        # Time embedding
        self.t_mlp = nn.Sequential(
            nn.Linear(1, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        # Optional conditioning
        if cfg.cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cfg.cond_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
        else:
            self.cond_mlp = None

        # Velocity network
        in_dim = D + cfg.hidden_dim + cfg.hidden_dim  # x_t + ctx + t
        if cfg.cond_dim > 0:
            in_dim += cfg.hidden_dim

        self.v_net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, D),
        )

    def encode_ctx(self, ctx: torch.Tensor) -> torch.Tensor:
        # ctx: [B, H, D]
        _, (h_n, _) = self.ctx_rnn(ctx)
        # last layer hidden
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
                # classifier-free guidance dropout
                drop_mask = (torch.rand(B, device=cond.device) < self.cfg.cfg_dropout).float().unsqueeze(1)
                cond_in = cond * (1.0 - drop_mask)
            else:
                cond_in = cond
            cond_emb = self.cond_mlp(cond_in)
            parts.append(cond_emb)

        h = torch.cat(parts, dim=1)
        return self.v_net(h)

    # --------- flow-matching loss + sampling ---------

    def fm_loss(
        self,
        x: torch.Tensor,
        ctx: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rectified flow matching loss.

        Sample z ~ N(0,I), t~U(0,1), x_t=(1-t)z + t x.
        Target v = x - z
        """
        B, D = x.shape
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
        """Generate x via Euler integration of the rectified flow ODE.

        Starts at z~N(0,I) and integrates dx/dt = v(x,t).
        """
        device = ctx.device
        B = ctx.shape[0]
        D = self.cfg.state_dim
        x = torch.randn(B, D, device=device)

        # Simple Euler
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
        return x


# -----------------------------
# Simple metrics (for quick sanity checks)
# -----------------------------

def compute_basic_l2_metrics(ask_p: np.ndarray, ask_v: np.ndarray, bid_p: np.ndarray, bid_v: np.ndarray) -> Dict[str, float]:
    """A tiny set of paper-friendly sanity metrics."""
    mid = 0.5 * (ask_p[:, 0] + bid_p[:, 0])
    ret = np.diff(mid)
    spread = ask_p[:, 0] - bid_p[:, 0]
    depth = ask_v[:, 0] + bid_v[:, 0]
    imb = (bid_v[:, 0] - ask_v[:, 0]) / (bid_v[:, 0] + ask_v[:, 0] + 1e-8)

    def kurt(x):
        x = x - x.mean()
        v = (x**2).mean() + 1e-12
        return float((x**4).mean() / (v * v))

    return {
        "spread_mean": float(np.mean(spread)),
        "spread_median": float(np.median(spread)),
        "ret_std": float(np.std(ret)),
        "ret_kurtosis": kurt(ret) if len(ret) > 10 else float("nan"),
        "depth_mean": float(np.mean(depth)),
        "imb_abs_mean": float(np.mean(np.abs(imb))),
        "valid_ratio": float(
            np.mean(
                (spread > 0)
                & np.all(np.diff(ask_p, axis=1) >= 0, axis=1)
                & np.all(np.diff(bid_p, axis=1) <= 0, axis=1)
            )
        ),
    }
