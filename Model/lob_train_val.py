"""lob_train_val.py

Training + evaluation utilities for L2 LOB generators.

This file contains:
- train_loop(): trains BiMeanFlowLOB or BiFlowLOB on a WindowedLOBParamsDataset
- generate_continuation(): autoregressive continuation in normalized space
- eval_one_window(): decodes raw L2 and prints metrics real vs generated
- eval_many_windows(): evaluates metrics over many random windows (recommended for paper figs)

Models: `lob_model.py`
Datasets/feature map/metrics: `lob_datasets.py`
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from lob_model import LOBConfig, BiMeanFlowLOB, BiFlowLOB
from lob_datasets import L2FeatureMap, WindowedLOBParamsDataset, compute_basic_l2_metrics


def seed_all(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(ds: WindowedLOBParamsDataset, batch_size: int, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)


def _parse_batch(batch, cfg: LOBConfig):
    """Supports these dataset return formats:
      - (hist, tgt, meta)
      - (hist, tgt, fut, meta)
      - (hist, tgt, cond, meta)
      - (hist, tgt, fut, cond, meta)
    """
    cond = None
    fut = None

    if cfg.rollout_K > 0:
        if cfg.cond_dim > 0:
            hist, tgt, fut, cond, _meta = batch
        else:
            hist, tgt, fut, _meta = batch
    else:
        if cfg.cond_dim > 0:
            hist, tgt, cond, _meta = batch
        else:
            hist, tgt, _meta = batch

    return hist, tgt, fut, cond


def train_loop(
    ds: WindowedLOBParamsDataset,
    cfg: LOBConfig,
    model_name: str = "bimean",
    steps: int = 3000,
    log_every: int = 200,
) -> torch.nn.Module:
    """
    model_name:
      - "bimean"  -> BiMeanFlowLOB (1-step sampling)
      - "biflow"  -> BiFlowLOB (multi-step sampling baseline)
    """
    device = cfg.device
    loader = make_loader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    it = iter(loader)

    if model_name == "bimean":
        model = BiMeanFlowLOB(cfg).to(device)
        if ds.is_standardized:
            model.set_scaler(ds.params_mean, ds.params_std)
    elif model_name == "biflow":
        model = BiFlowLOB(cfg).to(device)
    else:
        raise ValueError("model_name must be 'bimean' or 'biflow'")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    model.train()
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        hist, tgt, fut, cond = _parse_batch(batch, cfg)

        hist = hist.to(device).float()
        tgt = tgt.to(device).float()
        if fut is not None:
            fut = fut.to(device).float()
        if cond is not None:
            cond = cond.to(device).float()

        opt.zero_grad(set_to_none=True)

        if model_name == "bimean":
            loss, logs = model.loss(tgt, hist, cond=cond, x_future=fut)
        else:
            loss = model.fm_loss(tgt, hist, cond=cond)
            logs = {"loss_total": float(loss.detach().cpu())}

        loss.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if (step % log_every) == 0 or step == 1:
            if model_name == "bimean":
                print(
                    f"step {step:5d} | total {logs['loss_total']:.4f} | mean {logs['loss_mean']:.4f} | "
                    f"xrec {logs['loss_xrec']:.4f} | zcyc {logs['loss_zcycle']:.4f} | prior {logs['loss_prior']:.4f} | "
                    f"imb {logs['loss_imb']:.4f} | roll {logs['loss_roll']:.4f}"
                )
            else:
                print(f"step {step:5d} | fm_loss {logs['loss_total']:.4f}")

    return model


@torch.no_grad()
def generate_continuation(
    model: torch.nn.Module,
    ds: WindowedLOBParamsDataset,
    cfg: LOBConfig,
    idx: int,
    horizon: int = 200,
    model_name: str = "bimean",
    ode_steps: int = 32,
):
    """Autoregressive continuation of length `horizon` from ds[idx] history.

    Returns:
      hist_norm: [H,D] normalized
      gen_norm:  [horizon,D] normalized
      meta: dict (contains init_mid_for_window)
    """
    device = next(model.parameters()).device
    D = ds.params.shape[1]

    item = ds[idx]
    # Unpack (history, target, [future], [cond], meta)
    if ds.future_horizon > 0:
        if len(item) == 4:
            hist, _tgt, _fut, meta = item
            cond = None
        else:
            hist, _tgt, _fut, cond, meta = item
    else:
        if len(item) == 3:
            hist, _tgt, meta = item
            cond = None
        else:
            hist, _tgt, cond, meta = item

    ctx = hist.unsqueeze(0).to(device).float()

    # cond is per-step; expand to batch if provided
    cond_b = cond.unsqueeze(0).to(device).float() if cond is not None else None

    gen = []
    for _ in range(horizon):
        if model_name == "bimean":
            x_next = model.sample(ctx, cond=cond_b)  # [1,D]
        else:
            x_next = model.sample(ctx, cond=cond_b, steps=ode_steps)  # [1,D]
        gen.append(x_next.squeeze(0).detach().cpu())
        ctx = torch.cat([ctx[:, 1:, :], x_next.unsqueeze(1)], dim=1)

    gen = torch.stack(gen, dim=0).numpy().astype(np.float32)   # [horizon,D] normalized
    hist_np = hist.numpy().astype(np.float32)                  # [H,D] normalized
    return hist_np, gen, meta


def eval_one_window(
    ds: WindowedLOBParamsDataset,
    model: torch.nn.Module,
    cfg: LOBConfig,
    horizon: int = 200,
    model_name: str = "bimean",
    ode_steps: int = 32,
    idx: Optional[int] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Print real-vs-generated metrics for one window."""
    fm = L2FeatureMap(levels=cfg.levels)

    # choose an idx that has enough room for horizon in the raw sequence
    if idx is None:
        valid = np.where(ds.start_indices + horizon < len(ds.params))[0]
        idx = int(np.random.choice(valid))

    t = int(ds.start_indices[idx])
    H = ds.H

    # Real: history + future horizon in normalized, then denorm to raw
    real_norm = ds.params[t - H : t + horizon].astype(np.float32)
    real_raw = ds.denorm(real_norm)

    # Generated continuation
    hist_norm, gen_norm, meta = generate_continuation(
        model, ds, cfg, idx=idx, horizon=horizon, model_name=model_name, ode_steps=ode_steps
    )
    hist_raw = ds.denorm(hist_norm)
    gen_raw = ds.denorm(gen_norm)

    init_mid = float(meta["init_mid_for_window"])

    ask_p_r, ask_v_r, bid_p_r, bid_v_r, _ = fm.decode_sequence(real_raw, init_mid=init_mid)
    ask_p_g, ask_v_g, bid_p_g, bid_v_g, _ = fm.decode_sequence(np.concatenate([hist_raw, gen_raw], axis=0), init_mid=init_mid)

    mr = compute_basic_l2_metrics(ask_p_r, ask_v_r, bid_p_r, bid_v_r)
    mg = compute_basic_l2_metrics(ask_p_g, ask_v_g, bid_p_g, bid_v_g)

    print("\n--- Real (history+future) metrics ---")
    for k, v in mr.items():
        print(f"{k:>14s}: {v:.6g}")

    print("\n--- Generated (history+gen) metrics ---")
    for k, v in mg.items():
        print(f"{k:>14s}: {v:.6g}")

    return mr, mg


def eval_many_windows(
    ds: WindowedLOBParamsDataset,
    model: torch.nn.Module,
    cfg: LOBConfig,
    horizon: int = 200,
    model_name: str = "bimean",
    ode_steps: int = 32,
    n_windows: int = 100,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Evaluate mean±std of metrics over many random windows (recommended for paper plots)."""
    rng = np.random.default_rng(seed)
    fm = L2FeatureMap(levels=cfg.levels)

    valid = np.where(ds.start_indices + horizon < len(ds.params))[0]
    if len(valid) == 0:
        raise ValueError("No valid windows for the requested horizon.")
    picks = rng.choice(valid, size=min(n_windows, len(valid)), replace=False)

    real_rows = []
    gen_rows = []

    for idx in picks:
        idx = int(idx)
        t = int(ds.start_indices[idx])
        H = ds.H

        real_norm = ds.params[t - H : t + horizon].astype(np.float32)
        real_raw = ds.denorm(real_norm)

        hist_norm, gen_norm, meta = generate_continuation(
            model, ds, cfg, idx=idx, horizon=horizon, model_name=model_name, ode_steps=ode_steps
        )
        hist_raw = ds.denorm(hist_norm)
        gen_raw = ds.denorm(gen_norm)

        init_mid = float(meta["init_mid_for_window"])
        ask_p_r, ask_v_r, bid_p_r, bid_v_r, _ = fm.decode_sequence(real_raw, init_mid=init_mid)
        ask_p_g, ask_v_g, bid_p_g, bid_v_g, _ = fm.decode_sequence(np.concatenate([hist_raw, gen_raw], axis=0), init_mid=init_mid)

        real_rows.append(compute_basic_l2_metrics(ask_p_r, ask_v_r, bid_p_r, bid_v_r))
        gen_rows.append(compute_basic_l2_metrics(ask_p_g, ask_v_g, bid_p_g, bid_v_g))

    def summarize(rows):
        keys = rows[0].keys()
        out = {}
        for k in keys:
            vals = np.array([r[k] for r in rows], dtype=np.float64)
            out[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
        return out

    summary = {"real": summarize(real_rows), "gen": summarize(gen_rows)}
    print("\n=== Summary over", len(picks), "windows (horizon =", horizon, ") ===")
    for k in summary["real"].keys():
        r = summary["real"][k]
        g = summary["gen"][k]
        print(f"{k:>14s} | real {r['mean']:.6g}±{r['std']:.3g} | gen {g['mean']:.6g}±{g['std']:.3g}")
    return summary
