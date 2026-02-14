"""lob_colab_quickrun_std.py

Colab-friendly quick run (NO argparse).

What you get
------------
- run_synthetic(): train & evaluate on built-in synthetic L2 generator
- run_fi_like():   train & evaluate on FI-2010/FI-2020-like arrays (.npy/.npz/.csv)
- run_abides_npz(): train & evaluate on ABIDES-exported L2 npz (bids/asks arrays)

Notes
-----
- Uses automatic standardization from demo_model_bimean_std.py (Mode A).
- Always denormalizes before decoding to prices/sizes for metrics.
- Designed for quick sanity checks (not full paper training).

"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from demo_model_bimean_std import (
    LOBConfig,
    L2FeatureMap,
    BiMeanFlowLOB,
    BiFlowLOB,
    build_dataset_synthetic,
    build_dataset_from_fi2010,
    build_dataset_from_abides,
    compute_basic_l2_metrics,
)


def _seed_all(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_loader(ds, batch_size: int, shuffle: bool = True, num_workers: int = 0):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)


def train_model(ds, cfg: LOBConfig, model_name: str = "bimean", steps: int = 3000, log_every: int = 200):
    """
    model_name:
      - "bimean"  -> BiMeanFlowLOB (1-step sampling)
      - "biflow"  -> BiFlowLOB (multi-step sampling baseline)
    """
    device = cfg.device
    loader = _make_loader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    it = iter(loader)

    if model_name == "bimean":
        model = BiMeanFlowLOB(cfg).to(device)
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

        if len(batch) == 3:
            hist, tgt, _meta = batch
            cond = None
        else:
            hist, tgt, cond, _meta = batch

        hist = hist.to(device).float()
        tgt = tgt.to(device).float()
        if cond is not None:
            cond = cond.to(device).float()

        opt.zero_grad(set_to_none=True)

        if model_name == "bimean":
            loss, logs = model.loss(tgt, hist, cond=cond)
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
                    f"xrec {logs['loss_xrec']:.4f} | zcyc {logs['loss_zcycle']:.4f} | prior {logs['loss_prior']:.4f}"
                )
            else:
                print(f"step {step:5d} | fm_loss {logs['loss_total']:.4f}")

    return model


@torch.no_grad()
def generate_continuation(model, ds, idx: int, horizon: int = 200, model_name: str = "bimean", ode_steps: int = 32):
    """
    Generates an autoregressive continuation of length `horizon` starting from ds[idx] history.
    Works on *normalized* params; caller should denorm before decoding.
    """
    device = next(model.parameters()).device
    H = ds.H
    D = ds.params.shape[1]

    hist, tgt, meta = ds[idx]  # tensors + dict
    # Build initial context window [1, H, D] (normalized)
    ctx = hist.unsqueeze(0).to(device).float()

    gen = []
    for _ in range(horizon):
        if model_name == "bimean":
            x_next = model.sample(ctx)  # [1, D]
        else:
            x_next = model.sample(ctx, steps=ode_steps)  # [1, D]
        gen.append(x_next.squeeze(0).detach().cpu())

        # shift context
        x_next_ctx = x_next.unsqueeze(1)  # [1,1,D]
        ctx = torch.cat([ctx[:, 1:, :], x_next_ctx], dim=1)

    gen = torch.stack(gen, dim=0).numpy().astype(np.float32)  # [T, D] normalized
    hist_np = hist.numpy().astype(np.float32)                # [H, D] normalized
    return hist_np, gen, meta


def eval_one_window(ds, model, model_name: str = "bimean", horizon: int = 200, ode_steps: int = 32, idx: int = None):
    """
    Prints basic real-vs-generated L2 metrics for one random window.
    """
    if idx is None:
        idx = int(np.random.randint(0, len(ds)))

    fm = L2FeatureMap(levels=ds.params.shape[1] // 4)

    # Real window: history + horizon
    t = int(ds.start_indices[idx])
    H = ds.H
    raw_real_norm = ds.params[t - H : t + horizon].astype(np.float32)
    raw_real = ds.denorm(raw_real_norm)

    # Generated continuation
    hist_norm, gen_norm, meta = generate_continuation(model, ds, idx=idx, horizon=horizon, model_name=model_name, ode_steps=ode_steps)
    hist_raw = ds.denorm(hist_norm)
    gen_raw = ds.denorm(gen_norm)

    # Decode both sequences from the SAME init_mid (raw mid)
    init_mid = float(meta["init_mid_for_window"])
    ask_p_r, ask_v_r, bid_p_r, bid_v_r, _ = fm.decode_sequence(raw_real, init_mid=init_mid)
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


def run_synthetic(
    seed: int = 0,
    steps: int = 3000,
    horizon: int = 200,
    model_name: str = "bimean",
    levels: int = 10,
    history_len: int = 50,
):
    _seed_all(seed)

    cfg = LOBConfig(
        levels=levels,
        history_len=history_len,
        hidden_dim=128,
        dropout=0.1,
        batch_size=64,
        lr=2e-4,
        weight_decay=1e-4,
        grad_clip=1.0,
        standardize=True,  # Mode A
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    ds = build_dataset_synthetic(cfg, length=200_000, seed=seed, stride=1)
    print("Dataset:", len(ds), "windows | standardized:", ds.is_standardized)

    model = train_model(ds, cfg, model_name=model_name, steps=steps, log_every=200)
    model.eval()

    eval_one_window(ds, model, model_name=model_name, horizon=horizon)
    return model, cfg, ds


def run_fi_like(
    path: str,
    layout: str = "auto",
    seed: int = 0,
    steps: int = 3000,
    horizon: int = 200,
    model_name: str = "bimean",
    levels: int = 10,
    history_len: int = 50,
):
    _seed_all(seed)

    cfg = LOBConfig(
        levels=levels,
        history_len=history_len,
        hidden_dim=128,
        dropout=0.1,
        batch_size=64,
        lr=2e-4,
        weight_decay=1e-4,
        grad_clip=1.0,
        standardize=True,  # Mode A
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    ds = build_dataset_from_fi2010(path, cfg, layout=layout, stride=1)
    print("Dataset:", len(ds), "windows | standardized:", ds.is_standardized)

    model = train_model(ds, cfg, model_name=model_name, steps=steps, log_every=200)
    model.eval()

    eval_one_window(ds, model, model_name=model_name, horizon=horizon)
    return model, cfg, ds


def run_abides_npz(
    path_npz: str,
    seed: int = 0,
    steps: int = 3000,
    horizon: int = 200,
    model_name: str = "bimean",
    levels: int = 10,
    history_len: int = 50,
):
    _seed_all(seed)

    cfg = LOBConfig(
        levels=levels,
        history_len=history_len,
        hidden_dim=128,
        dropout=0.1,
        batch_size=64,
        lr=2e-4,
        weight_decay=1e-4,
        grad_clip=1.0,
        standardize=True,  # Mode A
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    ds = build_dataset_from_abides(path_npz, cfg, stride=1)
    print("Dataset:", len(ds), "windows | standardized:", ds.is_standardized)

    model = train_model(ds, cfg, model_name=model_name, steps=steps, log_every=200)
    model.eval()

    eval_one_window(ds, model, model_name=model_name, horizon=horizon)
    return model, cfg, ds
