"""lob_datasets.py

Data + representation utilities for Level-2 (L2) limit order books.

This file contains:
- L2FeatureMap: valid-by-construction encoding/decoding (raw L2 <-> unconstrained params)
- Standardization helpers (Mode A)
- WindowedLOBParamsDataset (history->target windows; optional future horizon for rollout loss)
- Loaders/builders:
    * FI-2010/FI-2020-like arrays (.npy/.npz/.csv)
    * ABIDES L2 npz (bids/asks arrays)
    * Synthetic generator (sanity + ablations)
- Basic metrics for quick checks (computed in raw L2 space)

Models live in `lob_model.py`.
Training/evaluation loops live in `lob_train_val.py`.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from lob_model import LOBConfig


ArrayLike = Union[np.ndarray, torch.Tensor]


# -----------------------------
# Feature map: valid L2 <-> unconstrained params
# -----------------------------

class L2FeatureMap:
    """Encode/decode between raw L2 snapshots and an unconstrained vector.

    Raw format expected by encode_sequence():
      ask_p, ask_v, bid_p, bid_v each shape [T, L]

    Parameter vector per snapshot (dim=4L):
      [delta_mid,
       log_spread,
       log_ask_gaps(2..L),
       log_bid_gaps(2..L),
       log_ask_sizes(1..L),
       log_bid_sizes(1..L)]
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
        if T > 1:
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

        params must be in RAW scale (not standardized).
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

    # ---------- FI-like helpers ----------

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
# Standardization helpers (Mode A)
# -----------------------------

def standardize_params(params: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = params.mean(axis=0).astype(np.float32)
    sigma = params.std(axis=0).astype(np.float32)
    sigma = np.maximum(sigma, eps).astype(np.float32)
    params_norm = ((params - mu) / sigma).astype(np.float32)
    return params_norm, mu, sigma


def _apply_affine(x: ArrayLike, mu: np.ndarray, sigma: np.ndarray, inverse: bool) -> ArrayLike:
    if isinstance(x, np.ndarray):
        if inverse:
            return (x * sigma[None, :] + mu[None, :]).astype(np.float32)
        return ((x - mu[None, :]) / sigma[None, :]).astype(np.float32)

    if not torch.is_tensor(x):
        raise TypeError(f"Unsupported type: {type(x)}")

    mu_t = torch.from_numpy(mu).to(x.device, dtype=x.dtype)
    sig_t = torch.from_numpy(sigma).to(x.device, dtype=x.dtype)
    if inverse:
        return x * sig_t + mu_t
    return (x - mu_t) / sig_t


# -----------------------------
# Dataset
# -----------------------------

class WindowedLOBParamsDataset(torch.utils.data.Dataset):
    """History->target windows on (optionally standardized) params.

    If future_horizon=K>0, __getitem__ additionally returns a tensor fut [K, D]
    containing the next K targets (for rollout loss).

    Return formats:
      - no cond:
            (hist, tgt, meta)                    if K==0
            (hist, tgt, fut, meta)               if K>0
      - with cond:
            (hist, tgt, cond, meta)              if K==0
            (hist, tgt, fut, cond, meta)         if K>0
    """

    def __init__(
        self,
        params: np.ndarray,
        mids: np.ndarray,
        history_len: int,
        stride: int = 1,
        cond: Optional[np.ndarray] = None,
        params_mean: Optional[np.ndarray] = None,
        params_std: Optional[np.ndarray] = None,
        future_horizon: int = 0,
    ):
        assert params.ndim == 2
        assert mids.ndim == 1
        assert len(params) == len(mids)

        self.params = params.astype(np.float32)
        self.mids = mids.astype(np.float32)
        self.H = int(history_len)
        self.stride = int(stride)
        self.cond = cond

        self.params_mean = params_mean.astype(np.float32) if params_mean is not None else None
        self.params_std = params_std.astype(np.float32) if params_std is not None else None

        self.future_horizon = int(max(future_horizon, 0))

        max_t = (len(self.params) - 1) - self.future_horizon
        if max_t < self.H:
            raise ValueError("Not enough length for the requested (history_len + future_horizon).")
        self.start_indices = np.arange(self.H, max_t + 1, self.stride)

    @property
    def is_standardized(self) -> bool:
        return (self.params_mean is not None) and (self.params_std is not None)

    def norm(self, x_raw: ArrayLike) -> ArrayLike:
        if not self.is_standardized:
            return x_raw
        return _apply_affine(x_raw, self.params_mean, self.params_std, inverse=False)

    def denorm(self, x_norm: ArrayLike) -> ArrayLike:
        if not self.is_standardized:
            return x_norm
        return _apply_affine(x_norm, self.params_mean, self.params_std, inverse=True)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int):
        t = int(self.start_indices[idx])
        hist = self.params[t - self.H : t]
        tgt = self.params[t]

        meta = {
            "t": int(t),
            "mid_prev": float(self.mids[t - 1]),
            "init_mid_for_window": float(self.mids[t - self.H]),
        }

        fut_t = None
        if self.future_horizon > 0:
            fut = self.params[t + 1 : t + 1 + self.future_horizon]
            fut_t = torch.from_numpy(fut)

        hist_t = torch.from_numpy(hist)
        tgt_t = torch.from_numpy(tgt)

        if self.cond is None:
            if fut_t is None:
                return hist_t, tgt_t, meta
            return hist_t, tgt_t, fut_t, meta

        c = torch.from_numpy(self.cond[t])
        if fut_t is None:
            return hist_t, tgt_t, c, meta
        return hist_t, tgt_t, fut_t, c, meta


# -----------------------------
# Loaders / builders
# -----------------------------

def load_fi2010_like_array(path: str) -> np.ndarray:
    """Load a FI-2010/FI-2020-like array (.npy/.npz/.csv)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        x = np.load(path)
    elif ext == ".npz":
        z = np.load(path)
        x = z["data"] if "data" in z else z[list(z.keys())[0]]
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
    """
    z = np.load(path, allow_pickle=True)
    bids = z["bids"].astype(np.float32)
    asks = z["asks"].astype(np.float32)
    times = z["times"].astype(np.float32) if "times" in z else np.arange(len(bids), dtype=np.float32)
    return times, bids, asks


def build_dataset_from_fi2010(path: str, cfg: LOBConfig, layout: str = "auto", stride: int = 1) -> WindowedLOBParamsDataset:
    fm = L2FeatureMap(cfg.levels, cfg.eps)
    x = load_fi2010_like_array(path)
    ask_p, ask_v, bid_p, bid_v = fm.split_l2_from_array(x, layout=layout)
    params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)

    if cfg.standardize:
        params, mu, sig = standardize_params(params_raw)
    else:
        params, mu, sig = params_raw, None, None

    return WindowedLOBParamsDataset(
        params=params,
        mids=mids,
        history_len=cfg.history_len,
        stride=stride,
        params_mean=mu,
        params_std=sig,
        future_horizon=cfg.rollout_K,
    )


def build_dataset_from_abides(path_npz: str, cfg: LOBConfig, stride: int = 1) -> WindowedLOBParamsDataset:
    fm = L2FeatureMap(cfg.levels, cfg.eps)
    _, bids, asks = load_abides_l2_npz(path_npz)

    if bids.shape[1] != cfg.levels:
        raise ValueError(f"bids shape {bids.shape} incompatible with levels={cfg.levels}")

    bid_p = bids[:, :, 0]
    bid_v = bids[:, :, 1]
    ask_p = asks[:, :, 0]
    ask_v = asks[:, :, 1]

    params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)

    if cfg.standardize:
        params, mu, sig = standardize_params(params_raw)
    else:
        params, mu, sig = params_raw, None, None

    return WindowedLOBParamsDataset(
        params=params,
        mids=mids,
        history_len=cfg.history_len,
        stride=stride,
        params_mean=mu,
        params_std=sig,
        future_horizon=cfg.rollout_K,
    )


def build_dataset_synthetic(
    cfg: LOBConfig,
    length: int = 200_000,
    seed: int = 0,
    tick: float = 0.01,
    stride: int = 1,
) -> WindowedLOBParamsDataset:
    """Generate a quick synthetic L2 dataset (valid by construction)."""
    rng = np.random.default_rng(seed)
    L = cfg.levels
    T = int(length)
    fm = L2FeatureMap(L, cfg.eps)

    # Mid random walk
    mid = np.zeros(T, dtype=np.float32)
    mid[0] = 100.0
    vol = 0.02
    shocks = rng.normal(0.0, vol, size=T).astype(np.float32)
    mid[1:] = mid[0] + np.cumsum(shocks[1:]).astype(np.float32)

    # Spread (lognormal-ish) in price units; ensure >= 2*tick
    spread = (2.0 * tick) * np.exp(rng.normal(0.0, 0.25, size=T).astype(np.float32))
    spread = np.clip(spread, 2.0 * tick, None)

    # Level gaps: positive, around 1-3 ticks
    ask_gaps = (tick) * np.exp(rng.normal(math.log(2.0), 0.35, size=(T, L - 1)).astype(np.float32))
    bid_gaps = (tick) * np.exp(rng.normal(math.log(2.0), 0.35, size=(T, L - 1)).astype(np.float32))

    # Sizes: positive, decay with level
    base_size = np.exp(rng.normal(math.log(50.0), 0.5, size=(T, 1)).astype(np.float32))
    decay = np.exp(-0.15 * np.arange(L, dtype=np.float32))[None, :]
    ask_v = base_size * decay * np.exp(rng.normal(0.0, 0.2, size=(T, L)).astype(np.float32))
    bid_v = base_size * decay * np.exp(rng.normal(0.0, 0.2, size=(T, L)).astype(np.float32))
    ask_v = np.clip(ask_v, cfg.eps, None).astype(np.float32)
    bid_v = np.clip(bid_v, cfg.eps, None).astype(np.float32)

    ask_p = np.zeros((T, L), dtype=np.float32)
    bid_p = np.zeros((T, L), dtype=np.float32)
    ask_p[:, 0] = mid + 0.5 * spread
    bid_p[:, 0] = mid - 0.5 * spread
    for i in range(1, L):
        ask_p[:, i] = ask_p[:, i - 1] + ask_gaps[:, i - 1]
        bid_p[:, i] = bid_p[:, i - 1] - bid_gaps[:, i - 1]

    params_raw, mids = fm.encode_sequence(ask_p, ask_v, bid_p, bid_v)

    if cfg.standardize:
        params, mu, sig = standardize_params(params_raw)
    else:
        params, mu, sig = params_raw, None, None

    return WindowedLOBParamsDataset(
        params=params,
        mids=mids,
        history_len=cfg.history_len,
        stride=stride,
        params_mean=mu,
        params_std=sig,
        future_horizon=cfg.rollout_K,
    )


# -----------------------------
# Metrics
# -----------------------------

def compute_basic_l2_metrics(
    ask_p: np.ndarray, ask_v: np.ndarray, bid_p: np.ndarray, bid_v: np.ndarray
) -> Dict[str, float]:
    mid = 0.5 * (ask_p[:, 0] + bid_p[:, 0])
    ret = np.diff(mid)
    spread = ask_p[:, 0] - bid_p[:, 0]
    depth = ask_v[:, 0] + bid_v[:, 0]
    imb = (bid_v[:, 0] - ask_v[:, 0]) / (bid_v[:, 0] + ask_v[:, 0] + 1e-8)

    def kurt(x: np.ndarray) -> float:
        if len(x) < 16:
            return float("nan")
        x = x - x.mean()
        v = (x**2).mean() + 1e-12
        return float((x**4).mean() / (v * v))

    return {
        "spread_mean": float(np.mean(spread)),
        "spread_median": float(np.median(spread)),
        "ret_std": float(np.std(ret)) if len(ret) > 2 else float("nan"),
        "ret_kurtosis": kurt(ret),
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
