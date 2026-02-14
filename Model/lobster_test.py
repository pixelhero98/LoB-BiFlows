"""lobster_test_bimean.py

Train & evaluate L2 LOB generators on free datasets:
  1) FI-2010-like arrays (public benchmark; 10 levels, 40 raw features)
  2) ABIDES L2 snapshots (npz with bids/asks arrays)
  3) Synthetic L2 generator (built-in; quick sanity checks)

Models:
  - bimean    : BiFlow + MeanFlow hybrid (paired x<->z + 1-step sampling)
  - rectified : rectified-flow / flow-matching baseline (multi-step sampling)

Examples
--------
# FI-2010 (array with at least 40 columns)
python lobster_test_bimean.py --dataset fi2010 --path /path/to/fi2010.npy --layout auto --model bimean

# Synthetic (no file needed)
python lobster_test_bimean.py --dataset synthetic --model bimean --synthetic_len 200000

# Baseline rectified flow on FI-2010 (compare vs bimean)
python lobster_test_bimean.py --dataset fi2010 --path /path/to/fi2010.npy --model rectified --ode_steps 32

This script prints quick real-vs-generated metrics for one random window.
For a paper, you'd average metrics over many windows / multiple seeds.
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from demo_model_bimean import (
    L2FeatureMap,
    LOBConfig,
    BiFlowLOB,
    BiMeanFlowLOB,
    build_dataset_from_abides,
    build_dataset_from_fi2010,
    build_dataset_synthetic,
    compute_basic_l2_metrics,
)


def seed_all(seed: int = 0):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    cfg: LOBConfig,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
) -> torch.nn.Module:
    if model_name == "rectified":
        model = BiFlowLOB(cfg).to(device)
    elif model_name == "bimean":
        model = BiMeanFlowLOB(cfg).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    step = 0
    model.train()
    while step < cfg.steps:
        for batch in loader:
            if cfg.cond_dim == 0:
                hist, tgt, _meta = batch
                cond = None
            else:
                hist, tgt, cond, _meta = batch
                cond = cond.to(device)

            hist = hist.to(device).float()
            tgt = tgt.to(device).float()

            if model_name == "rectified":
                loss = model.fm_loss(tgt, hist, cond=cond)
                logs = {"loss_total": float(loss.detach().cpu())}
            else:
                loss, logs = model.loss(tgt, hist, cond=cond)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            step += 1
            if step % 200 == 0:
                if model_name == "rectified":
                    print(f"step {step:6d} | fm_loss {logs['loss_total']:.4f}")
                else:
                    print(
                        f"step {step:6d} | total {logs['loss_total']:.4f} | "
                        f"mean {logs['loss_mean']:.4f} | xrec {logs['loss_xrec']:.4f} | "
                        f"zcyc {logs['loss_zcycle']:.4f} | prior {logs['loss_prior']:.4f}"
                    )

            if step >= cfg.steps:
                break

    return model


@torch.no_grad()
def generate_one_sequence(
    model: torch.nn.Module,
    model_name: str,
    history: torch.Tensor,
    gen_len: int,
    device: torch.device,
    ode_steps: int = 32,
    guidance_scale: float = 1.0,
) -> np.ndarray:
    """Autoregressive generation in parameter space.

    Returns params sequence [gen_len, D].
    """
    model.eval()
    hist = history.clone().to(device).float()  # [H, D]
    out = []

    for _ in range(gen_len):
        ctx = hist.unsqueeze(0)  # [1, H, D]
        if model_name == "rectified":
            x_next = model.sample(ctx, cond=None, steps=ode_steps, guidance_scale=guidance_scale).squeeze(0)
        else:
            x_next = model.sample(ctx, cond=None, guidance_scale=guidance_scale).squeeze(0)
        out.append(x_next.detach().cpu().numpy())
        hist = torch.cat([hist[1:], x_next.unsqueeze(0)], dim=0)

    return np.stack(out, axis=0).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fi2010", "abides", "synthetic"], required=True)
    p.add_argument("--path", type=str, default="")
    p.add_argument("--layout", type=str, default="auto", choices=["auto", "interleaved", "blocks"])

    p.add_argument("--model", choices=["bimean", "rectified"], default="bimean")
    p.add_argument("--levels", type=int, default=10)
    p.add_argument("--history_len", type=int, default=50)

    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)

    # evaluation / generation
    p.add_argument("--eval_gen_len", type=int, default=2000)
    p.add_argument("--ode_steps", type=int, default=32, help="Only used for rectified baseline")
    p.add_argument("--guidance_scale", type=float, default=1.0)

    # synthetic data
    p.add_argument("--synthetic_len", type=int, default=200000)
    args = p.parse_args()

    seed_all(args.seed)

    cfg = LOBConfig(
        levels=args.levels,
        history_len=args.history_len,
        steps=args.steps,
        batch_size=args.batch_size,
    )
    device = cfg.device

    if args.dataset in ("fi2010", "abides") and not args.path:
        raise SystemExit("--path is required for dataset fi2010/abides")

    if args.dataset == "fi2010":
        ds = build_dataset_from_fi2010(args.path, cfg, layout=args.layout)
    elif args.dataset == "abides":
        ds = build_dataset_from_abides(args.path, cfg)
    else:
        ds = build_dataset_synthetic(cfg, length=args.synthetic_len, seed=args.seed)

    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    print(f"Dataset windows: {len(ds):,} | levels={cfg.levels} | state_dim={cfg.state_dim}")
    print(f"Training model: {args.model}")

    model = train(cfg, loader, device, model_name=args.model)

    # ---------- quick evaluation (single-window) ----------
    fm = L2FeatureMap(cfg.levels, cfg.eps)

    idx = np.random.randint(0, len(ds))
    if cfg.cond_dim == 0:
        hist, _tgt, meta = ds[idx]
    else:
        hist, _tgt, _cond, meta = ds[idx]

    hist_np = hist.numpy()
    init_mid = meta["init_mid_for_window"]
    ask_p_r, ask_v_r, bid_p_r, bid_v_r, _mids_r = fm.decode_sequence(hist_np, init_mid=init_mid)
    real_metrics = compute_basic_l2_metrics(ask_p_r, ask_v_r, bid_p_r, bid_v_r)

    gen_params = generate_one_sequence(
        model=model,
        model_name=args.model,
        history=hist,
        gen_len=args.eval_gen_len,
        device=device,
        ode_steps=args.ode_steps,
        guidance_scale=args.guidance_scale,
    )

    init_mid_gen = meta["mid_prev"]
    ask_p_g, ask_v_g, bid_p_g, bid_v_g, _mids_g = fm.decode_sequence(gen_params, init_mid=init_mid_gen)
    gen_metrics = compute_basic_l2_metrics(ask_p_g, ask_v_g, bid_p_g, bid_v_g)

    print("\n--- Real (history window) metrics ---")
    for k, v in real_metrics.items():
        print(f"{k:>14s}: {v:.6g}")

    print("\n--- Generated (continuation) metrics ---")
    for k, v in gen_metrics.items():
        print(f"{k:>14s}: {v:.6g}")

    if args.model == "rectified":
        print(f"\n(Note) rectified baseline used ode_steps={args.ode_steps}")
    else:
        print("\n(Note) bimean model uses 1-step sampling by default.")


if __name__ == "__main__":
    main()
