"""
train_pair_correction_v4.py

Train the bounded, velocity-aware Zone 3 scalar correction NN for SIMON.

Why v4 exists
-------------
The v3 model learned the 6-input Zone 3 target well on held-out data, but it
could still produce unsafe near-zero c values on fixed-state probe cases.
That is possible because v3 directly regressed log(c_opt) with an unbounded
linear output.

v4 keeps the same scientific target and the same 6 input features, but changes
the model output to a bounded c range:

    c_min_train <= c_pred <= c_max_train

by predicting a raw scalar z and mapping it through a sigmoid:

    log(c_pred) = log(c_min_train) +
                  (log(c_max_train)-log(c_min_train)) * sigmoid(z)

Default bounds are c in [0.25, 3.0], chosen to cover the observed augmented
Zone 3 target range while preventing the pathological near-zero collapse.

Features:
    [log(r_soft), log(m_i), log(m_j), log(dt), v_rad_norm, v_tan_norm]
Target:
    log(c_opt)

Additional v4 safeguards
------------------------
1. Weighted training loss increases pressure on the strong-approach, large-dt
   target window that exposed the v3 failure.
2. A data-derived fixed-probe anchoring loss is added. For each fixed probe,
   the target is the median c_opt of the nearest real training samples at the
   same dt in normalised 6D feature space. This discourages artificial local
   valleys while keeping the target grounded in generated data.
3. Training diagnostics print the same fixed-state probe table so the collapse
   is visible immediately after training.

Important compatibility note
----------------------------
This v4 model is NOT compatible with the old diagnostic_v3.py model class,
because v4's forward pass includes the bounded sigmoid transform. Use a v4-aware
diagnostic/evaluator that defines the same PairCorrectionNN class below.

Example:
    python -B train_pair_correction_v4.py --data encounter_data_zone3_v3_augmented.npz --epochs 5000 --hidden 64 --output pair_correction_nn_v4_bounded.pt
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    plt = None


REQUIRED_FIELDS = [
    "r_AU", "r_soft", "log_mi", "log_mj", "log_dt",
    "v_rad_norm", "v_tan_norm",
    "c_opt", "log_c_opt", "c_ana", "log_c_ana", "improvement",
]
FEATURE_NAMES = ["log_r_soft", "log_mi", "log_mj", "log_dt", "v_rad_norm", "v_tan_norm"]
DT_LIST = [0.005, 0.010, 0.020, 0.040, 0.050, 0.060, 0.080, 0.100]
EPS = 3e-4


class PairCorrectionNN(nn.Module):
    """
    6 -> H -> H -> H -> 1 with SiLU, followed by bounded log(c) transform.

    Inputs:
        [log(r_soft), log(m_i), log(m_j), log(dt), v_rad_norm, v_tan_norm]
    Output:
        bounded log(c_pred), where c_pred is constrained to [c_min, c_max].
    """
    def __init__(self, hidden: int = 64, c_min: float = 0.25, c_max: float = 3.0):
        super().__init__()
        if not (0.0 < c_min < c_max):
            raise ValueError(f"Invalid bounds: c_min={c_min}, c_max={c_max}")

        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("input_mean", torch.zeros(6))
        self.register_buffer("input_std", torch.ones(6))
        self.register_buffer("log_c_min", torch.tensor(float(np.log(c_min)), dtype=torch.float32))
        self.register_buffer("log_c_max", torch.tensor(float(np.log(c_max)), dtype=torch.float32))

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.net(x_norm).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.forward_raw(x)
        s = torch.sigmoid(raw)
        return self.log_c_min + (self.log_c_max - self.log_c_min) * s


@dataclass
class DatasetBundle:
    features: np.ndarray
    targets: np.ndarray
    raw: Dict[str, np.ndarray]


@dataclass
class ProbeBundle:
    names: List[str]
    features: np.ndarray
    targets: np.ndarray
    nn_dist: np.ndarray
    nn_c_med: np.ndarray


def _safe_std(x: np.ndarray) -> np.ndarray:
    s = x.std(axis=0).astype(np.float32)
    s[s < 1e-8] = 1.0
    return s


def _round_dt(log_dt: np.ndarray) -> np.ndarray:
    dt = np.exp(log_dt.astype(np.float64))
    return np.array([round(float(x), 4) for x in dt])


def load_zone3_v3_data(npz_path: str, c_min: float, c_max: float) -> DatasetBundle:
    data = np.load(npz_path)
    missing = [k for k in REQUIRED_FIELDS if k not in data.files]
    if missing:
        raise KeyError(
            f"{npz_path} is missing required v3 fields: {missing}\n"
            "The v4 model requires v_rad_norm and v_tan_norm."
        )

    raw = {k: data[k].astype(np.float32) for k in REQUIRED_FIELDS}
    n = len(raw["r_AU"])
    for k, arr in raw.items():
        if arr.ndim != 1:
            raise ValueError(f"Field {k} must be 1D, got shape {arr.shape}")
        if len(arr) != n:
            raise ValueError(f"Field {k} length {len(arr)} does not match r_AU length {n}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Field {k} contains non-finite values")

    if not np.all(raw["c_opt"] > 0):
        raise ValueError("c_opt must be strictly positive")

    log_r_soft = np.log(raw["r_soft"].astype(np.float64) + 1e-30).astype(np.float32)
    features = np.stack([
        log_r_soft,
        raw["log_mi"],
        raw["log_mj"],
        raw["log_dt"],
        raw["v_rad_norm"],
        raw["v_tan_norm"],
    ], axis=1).astype(np.float32)

    targets_unclipped = raw["log_c_opt"].astype(np.float32)
    lo = float(np.log(c_min))
    hi = float(np.log(c_max))
    targets = np.clip(targets_unclipped, lo, hi).astype(np.float32)
    n_clipped = int(np.sum(np.abs(targets - targets_unclipped) > 1e-7))

    print(f"[data] Loaded {npz_path}")
    print(f"[data] samples={n}")
    print(f"[data] v4 output bounds: c in [{c_min:.4g}, {c_max:.4g}]")
    print(f"[data] targets clipped to bounds: {n_clipped} ({n_clipped/max(n,1):.2%})")
    print("[data] feature ranges:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"       {name:12s} min={features[:, i].min(): .4f}  "
              f"med={np.median(features[:, i]): .4f}  max={features[:, i].max(): .4f}")
    print(f"[data] target c_opt: min={raw['c_opt'].min():.5f}  med={np.median(raw['c_opt']):.5f}  max={raw['c_opt'].max():.5f}")
    print(f"[data] target log_c_opt after clipping: mean={targets.mean():.5f}  std={targets.std():.5f}  "
          f"min={targets.min():.5f}  max={targets.max():.5f}")

    dt_rounded = _round_dt(raw["log_dt"])
    c = raw["c_opt"].astype(np.float64)
    vr = raw["v_rad_norm"].astype(np.float64)
    print("[data] per-dt target medians:")
    for dt in sorted(set(dt_rounded)):
        mask = dt_rounded == dt
        app = mask & (vr < 0)
        rec = mask & (vr >= 0)
        print(f"       dt={dt:0.4f}  n={int(mask.sum()):5d}  "
              f"c_med={np.median(c[mask]):.5f}  "
              f"app_med={np.median(c[app]) if np.any(app) else np.nan:.5f}  "
              f"rec_med={np.median(c[rec]) if np.any(rec) else np.nan:.5f}")

    return DatasetBundle(features=features, targets=targets, raw=raw)


def make_splits(n: int, seed: int, train_frac: float = 0.70, val_frac: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    if len(test_idx) == 0:
        raise ValueError("Test split is empty; use a larger dataset")
    return train_idx, val_idx, test_idx


def make_sample_weights(raw: Dict[str, np.ndarray], indices: np.ndarray,
                        strong_weight: float, approach_weight: float,
                        identity_weight: float, low_c_weight: float) -> np.ndarray:
    """
    Construct non-negative sample weights for the MSE loss.

    The weights are deliberately mild. They do not change the target; they only
    make the optimiser pay more attention to rare regions that matter for safety.
    """
    r = raw["r_AU"][indices].astype(np.float64)
    log_dt = raw["log_dt"][indices].astype(np.float64)
    dt = np.exp(log_dt)
    vr = raw["v_rad_norm"][indices].astype(np.float64)
    vt = raw["v_tan_norm"][indices].astype(np.float64)
    c = raw["c_opt"][indices].astype(np.float64)

    w = np.ones(len(indices), dtype=np.float32)

    large_dt = dt >= 0.039
    approach = vr < 0
    strong_target = large_dt & (vr < -0.6) & (r >= 0.08) & (r <= 0.12) & (vt >= 0.4) & (vt <= 0.9)
    identity = np.abs(c - 1.0) < 1e-7
    low_c = c < 0.45

    w += float(approach_weight) * (large_dt & approach).astype(np.float32)
    w += float(strong_weight) * strong_target.astype(np.float32)
    w += float(identity_weight) * identity.astype(np.float32)
    w += float(low_c_weight) * low_c.astype(np.float32)

    return w.astype(np.float32)


def evaluate_arrays(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    y_true_log = y_true_log.astype(np.float64)
    y_pred_log = y_pred_log.astype(np.float64)
    err_log = y_pred_log - y_true_log
    c_true = np.exp(y_true_log)
    c_pred = np.exp(y_pred_log)
    err_c = c_pred - c_true
    return {
        "mse_log": float(np.mean(err_log ** 2)),
        "rmse_log": float(np.sqrt(np.mean(err_log ** 2))),
        "mae_log": float(np.mean(np.abs(err_log))),
        "mae_c": float(np.mean(np.abs(err_c))),
        "median_abs_c": float(np.median(np.abs(err_c))),
        "p95_abs_c": float(np.percentile(np.abs(err_c), 95)),
        "corr_c": float(np.corrcoef(c_true, c_pred)[0, 1]) if len(c_true) > 2 else float("nan"),
        "pred_c_min": float(np.min(c_pred)),
        "pred_c_med": float(np.median(c_pred)),
        "pred_c_max": float(np.max(c_pred)),
    }


def predict_numpy(model: PairCorrectionNN, features: np.ndarray, device: str, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            xb = torch.tensor(features[i:i + batch_size], dtype=torch.float32, device=device)
            out.append(model(xb).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(out, axis=0)


def build_probe_inputs() -> Tuple[List[str], np.ndarray]:
    names: List[str] = []
    rows: List[List[float]] = []
    r_soft = float(np.sqrt(0.10 ** 2 + EPS ** 2))

    for d in DT_LIST:
        for label, vr in [("approach", -0.8), ("side", 0.0), ("recede", +0.8)]:
            names.append(f"A_{label}_dt{d:.4f}")
            rows.append([np.log(r_soft), np.log(1.0), np.log(0.01), np.log(d), vr, 0.65])

    for vt in [0.1, 0.3, 0.6, 0.9, 1.2, 1.6]:
        names.append(f"B_recede_dt0.0400_vt{vt:.1f}")
        rows.append([np.log(r_soft), np.log(1.0), np.log(0.01), np.log(0.04), +0.8, vt])

    return names, np.array(rows, dtype=np.float32)


def make_probe_bundle(features: np.ndarray, targets: np.ndarray, train_idx: np.ndarray,
                      feat_mean: np.ndarray, feat_std: np.ndarray, k: int = 16) -> ProbeBundle:
    """
    Build fixed-probe targets from nearest real training samples at the same dt.

    The target for each probe is median c_opt of the k nearest training samples
    restricted to the same dt value. This is not a new physical target; it is a
    local data-derived anchor that prevents pathological interpolation valleys.
    """
    names, X_probe = build_probe_inputs()
    X_train = features[train_idx].astype(np.float64)
    y_train = targets[train_idx].astype(np.float64)
    dt_train = _round_dt(X_train[:, 3].astype(np.float64))

    mean = feat_mean.astype(np.float64)
    std = feat_std.astype(np.float64) + 1e-8
    Xn_train = (X_train - mean) / std
    Xn_probe = (X_probe.astype(np.float64) - mean) / std

    probe_targets = []
    probe_dists = []
    probe_c_meds = []

    for name, xp, xnp in zip(names, X_probe, Xn_probe):
        d_probe = round(float(np.exp(float(xp[3]))), 4)
        same_dt = np.where(dt_train == d_probe)[0]
        if len(same_dt) == 0:
            # Should not happen for the standard dt list, but keep safe.
            same_dt = np.arange(len(Xn_train))
        diff = Xn_train[same_dt] - xnp
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        order = np.argsort(dist)[:max(1, min(k, len(dist)))]
        nn_local = same_dt[order]
        c_vals = np.exp(y_train[nn_local])
        c_med = float(np.median(c_vals))
        probe_targets.append(float(np.log(c_med)))
        probe_dists.append(float(np.min(dist)))
        probe_c_meds.append(c_med)

    print("\n[probe-anchor] Data-derived fixed-probe targets")
    print(f"  {'probe':>26} | {'nn_dist':>8} | {'target c_med':>12}")
    print("  " + "-" * 53)
    for name, d, cm in zip(names, probe_dists, probe_c_meds):
        print(f"  {name:>26s} | {d:8.3f} | {cm:12.5f}")

    return ProbeBundle(
        names=names,
        features=X_probe.astype(np.float32),
        targets=np.array(probe_targets, dtype=np.float32),
        nn_dist=np.array(probe_dists, dtype=np.float32),
        nn_c_med=np.array(probe_c_meds, dtype=np.float32),
    )


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.sum(weight * (pred - target) ** 2) / torch.sum(weight.clamp_min(1e-8))


def train_model(bundle: DatasetBundle, hidden: int, epochs: int, lr: float, seed: int,
                device: str, batch_size: int, output_prefix: str,
                c_min: float, c_max: float,
                strong_weight: float, approach_weight: float,
                identity_weight: float, low_c_weight: float,
                probe_weight: float, probe_k: int) -> Tuple[PairCorrectionNN, Dict[str, np.ndarray], Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    features = bundle.features
    targets = bundle.targets
    n = len(features)
    train_idx, val_idx, test_idx = make_splits(n, seed=seed)

    feat_mean = features[train_idx].mean(axis=0).astype(np.float32)
    feat_std = _safe_std(features[train_idx])

    train_weights = make_sample_weights(
        bundle.raw, train_idx,
        strong_weight=strong_weight,
        approach_weight=approach_weight,
        identity_weight=identity_weight,
        low_c_weight=low_c_weight,
    )

    print("\n[weights] training sample weights")
    print(f"  min/med/max = {train_weights.min():.3f} / {np.median(train_weights):.3f} / {train_weights.max():.3f}")
    print(f"  mean        = {train_weights.mean():.3f}")

    probe_bundle = make_probe_bundle(features, targets, train_idx, feat_mean, feat_std, k=probe_k)

    model = PairCorrectionNN(hidden=hidden, c_min=c_min, c_max=c_max).to(device)
    model.input_mean = torch.tensor(feat_mean, dtype=torch.float32, device=device)
    model.input_std = torch.tensor(feat_std, dtype=torch.float32, device=device)

    X_train = torch.tensor(features[train_idx], dtype=torch.float32, device=device)
    y_train = torch.tensor(targets[train_idx], dtype=torch.float32, device=device)
    w_train = torch.tensor(train_weights, dtype=torch.float32, device=device)
    X_val = torch.tensor(features[val_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(targets[val_idx], dtype=torch.float32, device=device)
    X_probe = torch.tensor(probe_bundle.features, dtype=torch.float32, device=device)
    y_probe = torch.tensor(probe_bundle.targets, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr / 100.0)
    mse_loss = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    train_hist = []
    val_hist = []
    probe_hist = []
    t0 = time.perf_counter()

    n_train = len(train_idx)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        ep_loss_sum = 0.0
        ep_seen = 0
        for start in range(0, n_train, batch_size):
            ids = perm[start:start + batch_size]
            xb = X_train[ids]
            yb = y_train[ids]
            wb = w_train[ids]

            pred = model(xb)
            data_loss = weighted_mse(pred, yb, wb)

            if probe_weight > 0.0:
                probe_pred = model(X_probe)
                probe_loss = mse_loss(probe_pred, y_probe)
                loss = data_loss + float(probe_weight) * probe_loss
            else:
                probe_loss = torch.tensor(0.0, device=device)
                loss = data_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            ep_loss_sum += float(data_loss.item()) * len(ids)
            ep_seen += len(ids)

        sched.step()
        train_loss = ep_loss_sum / max(ep_seen, 1)

        model.eval()
        with torch.no_grad():
            val_loss = float(mse_loss(model(X_val), y_val).item())
            probe_loss_val = float(mse_loss(model(X_probe), y_probe).item())
        train_hist.append(train_loss)
        val_hist.append(val_loss)
        probe_hist.append(probe_loss_val)

        # Select by validation loss plus a small probe-safety component.
        selection_score = val_loss + float(probe_weight) * probe_loss_val
        if selection_score < best_val:
            best_val = selection_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 0 or (ep + 1) % max(1, epochs // 10) == 0 or ep + 1 == epochs:
            print(f"  ep {ep+1:5d}/{epochs}  train={train_loss:.7f}  val={val_loss:.7f}  probe={probe_loss_val:.7f}")

    elapsed = time.perf_counter() - t0
    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    np.savez_compressed(
        f"{output_prefix}_split.npz",
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
        feature_names=np.array(FEATURE_NAMES),
        input_mean=feat_mean,
        input_std=feat_std,
        c_min=np.array([c_min], dtype=np.float32),
        c_max=np.array([c_max], dtype=np.float32),
        seed=np.array([seed], dtype=np.int64),
        probe_names=np.array(probe_bundle.names),
        probe_features=probe_bundle.features,
        probe_targets=probe_bundle.targets,
        probe_nn_dist=probe_bundle.nn_dist,
        probe_nn_c_med=probe_bundle.nn_c_med,
    )

    pred_train = predict_numpy(model, features[train_idx], device)
    pred_val = predict_numpy(model, features[val_idx], device)
    pred_test = predict_numpy(model, features[test_idx], device)

    metrics = {}
    for name, idx, pred in [
        ("train", train_idx, pred_train),
        ("val", val_idx, pred_val),
        ("test", test_idx, pred_test),
    ]:
        m = evaluate_arrays(targets[idx].astype(np.float64), pred)
        for k, v in m.items():
            metrics[f"{name}_{k}"] = v

    pred_probe = predict_numpy(model, probe_bundle.features, device)
    m_probe = evaluate_arrays(probe_bundle.targets.astype(np.float64), pred_probe)
    for k, v in m_probe.items():
        metrics[f"probe_{k}"] = v
    metrics["probe_hard_gate_frac"] = float(np.mean((np.exp(pred_probe) < 0.2) | (np.exp(pred_probe) > 5.0)))
    metrics["probe_conservative_gate_frac"] = float(np.mean((np.exp(pred_probe) < 0.4) | (np.exp(pred_probe) > 2.5)))

    metrics["best_selection_score"] = float(best_val)
    metrics["elapsed_sec"] = float(elapsed)
    metrics["n_train"] = float(len(train_idx))
    metrics["n_val"] = float(len(val_idx))
    metrics["n_test"] = float(len(test_idx))
    metrics["c_min_bound"] = float(c_min)
    metrics["c_max_bound"] = float(c_max)

    histories = {
        "train_loss": np.array(train_hist, dtype=np.float64),
        "val_loss": np.array(val_hist, dtype=np.float64),
        "probe_loss": np.array(probe_hist, dtype=np.float64),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "probe_features": probe_bundle.features,
        "probe_targets": probe_bundle.targets,
        "probe_names": np.array(probe_bundle.names),
    }
    return model, histories, metrics


def print_test_diagnostics(bundle: DatasetBundle, model: PairCorrectionNN, indices: np.ndarray, device: str,
                           probe_features: np.ndarray, probe_targets: np.ndarray, probe_names: np.ndarray) -> None:
    features = bundle.features[indices]
    targets = bundle.targets[indices].astype(np.float64)
    pred_log = predict_numpy(model, features, device)
    c_true = np.exp(targets)
    c_pred = np.exp(pred_log)
    raw = bundle.raw
    dt = np.exp(raw["log_dt"][indices].astype(np.float64))
    rounded = np.array([round(float(x), 4) for x in dt])
    vr = raw["v_rad_norm"][indices].astype(np.float64)
    impr = raw["improvement"][indices].astype(np.float64)

    print("\n[test] Overall held-out metrics")
    m = evaluate_arrays(targets, pred_log)
    for k in ["mse_log", "rmse_log", "mae_log", "mae_c", "median_abs_c", "p95_abs_c", "corr_c"]:
        print(f"       {k:14s}: {m[k]:.6g}")
    print(f"       pred c range : {c_pred.min():.4f} / {np.median(c_pred):.4f} / {c_pred.max():.4f}")
    print(f"       true c range : {c_true.min():.4f} / {np.median(c_true):.4f} / {c_true.max():.4f}")
    print(f"       out-of-gate predicted c (<0.2 or >5): {np.mean((c_pred < 0.2) | (c_pred > 5.0)):.2%}")
    print(f"       conservative warnings (<0.4 or >2.5): {np.mean((c_pred < 0.4) | (c_pred > 2.5)):.2%}")

    print("\n[test] Per-dt medians on held-out test set")
    print(f"  {'dt':>7} | {'n':>5} | {'true_med':>9} | {'pred_med':>9} | {'MAE_c':>9} | {'impr_med':>9}")
    print("  " + "-" * 61)
    for d in sorted(set(rounded)):
        mask = rounded == d
        print(f"  {d:7.4f} | {int(mask.sum()):5d} | {np.median(c_true[mask]):9.4f} | "
              f"{np.median(c_pred[mask]):9.4f} | {np.mean(np.abs(c_pred[mask]-c_true[mask])):9.4f} | "
              f"{np.median(impr[mask]):9.2%}")

    print("\n[test] Approaching vs receding medians on held-out test set")
    print(f"  {'dt':>7} | {'app_n':>5} | {'app_true':>9} | {'app_pred':>9} | {'rec_n':>5} | {'rec_true':>9} | {'rec_pred':>9}")
    print("  " + "-" * 85)
    signs_ok = 0
    signs_total = 0
    for d in sorted(set(rounded)):
        base = rounded == d
        app = base & (vr < 0)
        rec = base & (vr >= 0)
        def med_or_nan(a, mask):
            return float(np.median(a[mask])) if np.any(mask) else float("nan")
        app_t, app_p = med_or_nan(c_true, app), med_or_nan(c_pred, app)
        rec_t, rec_p = med_or_nan(c_true, rec), med_or_nan(c_pred, rec)
        if np.isfinite(app_p) and np.isfinite(rec_p):
            signs_total += 1
            signs_ok += int(app_p > rec_p)
        print(f"  {d:7.4f} | {int(app.sum()):5d} | {app_t:9.4f} | {app_p:9.4f} | "
              f"{int(rec.sum()):5d} | {rec_t:9.4f} | {rec_p:9.4f}")
    print(f"\n[test] Velocity-sign ordering: app_pred > rec_pred for {signs_ok}/{signs_total} dt groups")

    identity = np.abs(c_true - 1.0) < 1e-6
    if np.any(identity):
        print("\n[test] Near-identity targets on held-out test set")
        print(f"       n identity={int(identity.sum())}")
        print(f"       predicted c median={np.median(c_pred[identity]):.5f}")
        print(f"       within [0.90, 1.10]={np.mean((c_pred[identity] >= 0.90) & (c_pred[identity] <= 1.10)):.2%}")

    print("\n[probe] Fixed-state velocity-direction check")
    print("        r=0.10 AU, m_i=1, m_j=0.01, v_tan_norm=0.65")
    print(f"  {'dt':>7} | {'approach(vr=-0.8)':>18} | {'side(vr=0)':>12} | {'recede(vr=+0.8)':>17}")
    print("  " + "-" * 66)
    names, rows = build_probe_inputs()
    probe_pred = np.exp(predict_numpy(model, rows, device))
    name_to_pred = dict(zip(names, probe_pred))
    for d in DT_LIST:
        ca = name_to_pred[f"A_approach_dt{d:.4f}"]
        cs = name_to_pred[f"A_side_dt{d:.4f}"]
        cr = name_to_pred[f"A_recede_dt{d:.4f}"]
        flag = ""
        if ca < 0.2 or cs < 0.2 or cr < 0.2:
            flag = "  <-- HARD-GATE WARNING"
        print(f"  {d:7.4f} | {ca:18.5f} | {cs:12.5f} | {cr:17.5f}{flag}")

    print("\n[probe] Data-derived probe anchors vs prediction")
    pred_log = predict_numpy(model, probe_features, device)
    pred_c = np.exp(pred_log)
    target_c = np.exp(probe_targets.astype(np.float64))
    print(f"  {'probe':>26} | {'target_c':>9} | {'pred_c':>9} | {'abs_err':>9}")
    print("  " + "-" * 61)
    for name, tc, pc in zip(probe_names, target_c, pred_c):
        print(f"  {str(name):>26s} | {tc:9.4f} | {pc:9.4f} | {abs(pc-tc):9.4f}")


def maybe_plot_training(histories: Dict[str, np.ndarray], output_prefix: str) -> None:
    if not HAS_MPL:
        print("[plot] matplotlib not available; skipping training_log_v4.png")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(histories["train_loss"], label="weighted train")
    ax.plot(histories["val_loss"], label="val")
    ax.plot(histories["probe_loss"], label="probe anchor")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss on log(c_opt)")
    ax.set_title("Zone 3 v4 bounded 6-input NN training curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = "training_log_v4.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[plot] Saved {out}")


def write_summary(path: str, args: argparse.Namespace, metrics: Dict[str, float]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Zone 3 v4 bounded training summary\n")
        f.write("=" * 72 + "\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write("\nMetrics\n")
        f.write("-" * 72 + "\n")
        for k in sorted(metrics):
            f.write(f"{k}: {metrics[k]}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train SIMON Zone 3 v4 bounded 6-input trajectory-correction NN.")
    ap.add_argument("--data", default="encounter_data_zone3_v3_augmented.npz")
    ap.add_argument("--hidden", type=int, default=64,
                    help="Hidden width. 64 is default for the 6-input target.")
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="pair_correction_nn_v4_bounded.pt")
    ap.add_argument("--c-min", type=float, default=0.25,
                    help="Lower bound for c_pred in the bounded output transform.")
    ap.add_argument("--c-max", type=float, default=3.0,
                    help="Upper bound for c_pred in the bounded output transform.")
    ap.add_argument("--strong-weight", type=float, default=2.0,
                    help="Extra sample weight for large-dt strong-approach target-window samples.")
    ap.add_argument("--approach-weight", type=float, default=0.35,
                    help="Extra sample weight for large-dt approaching samples.")
    ap.add_argument("--identity-weight", type=float, default=0.35,
                    help="Extra sample weight for c_opt=1 identity samples.")
    ap.add_argument("--low-c-weight", type=float, default=0.50,
                    help="Extra sample weight for low-c samples, so bounded output still learns legitimate small c.")
    ap.add_argument("--probe-weight", type=float, default=0.20,
                    help="Loss weight for data-derived fixed-probe anchors.")
    ap.add_argument("--probe-k", type=int, default=16,
                    help="Number of same-dt nearest neighbours used to define each probe anchor target.")
    args = ap.parse_args()

    output_prefix = os.path.splitext(args.output)[0]
    print(f"[config] data={args.data}")
    print(f"[config] output={args.output}")
    print(f"[config] hidden={args.hidden} epochs={args.epochs} lr={args.lr} batch_size={args.batch_size} seed={args.seed}")
    print(f"[config] device={args.device}")
    print(f"[config] bounded output c in [{args.c_min}, {args.c_max}]")
    print(f"[config] weights: strong={args.strong_weight} approach={args.approach_weight} "
          f"identity={args.identity_weight} low_c={args.low_c_weight} probe={args.probe_weight}")

    bundle = load_zone3_v3_data(args.data, c_min=args.c_min, c_max=args.c_max)
    model, histories, metrics = train_model(
        bundle=bundle,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        output_prefix=output_prefix,
        c_min=args.c_min,
        c_max=args.c_max,
        strong_weight=args.strong_weight,
        approach_weight=args.approach_weight,
        identity_weight=args.identity_weight,
        low_c_weight=args.low_c_weight,
        probe_weight=args.probe_weight,
        probe_k=args.probe_k,
    )

    torch.save(model.cpu().state_dict(), args.output)
    print(f"\n[save] Saved model: {args.output} ({os.path.getsize(args.output)/1024:.1f} KB)")
    print(f"[save] Saved split: {output_prefix}_split.npz")

    # Metrics and sanity diagnostics use CPU now that the model has been saved to CPU.
    model_cpu = PairCorrectionNN(hidden=args.hidden, c_min=args.c_min, c_max=args.c_max)
    model_cpu.load_state_dict(torch.load(args.output, map_location="cpu"))
    model_cpu.eval()
    print_test_diagnostics(
        bundle,
        model_cpu,
        histories["test_idx"],
        device="cpu",
        probe_features=histories["probe_features"],
        probe_targets=histories["probe_targets"],
        probe_names=histories["probe_names"],
    )

    np.savez_compressed(
        f"{output_prefix}_metrics.npz",
        **{k: np.array([v], dtype=np.float64) for k, v in metrics.items()},
        train_loss=histories["train_loss"],
        val_loss=histories["val_loss"],
        probe_loss=histories["probe_loss"],
    )
    print(f"[save] Saved metrics: {output_prefix}_metrics.npz")

    maybe_plot_training(histories, output_prefix)
    summary_path = "train_pair_correction_v4_summary.txt"
    write_summary(summary_path, args, metrics)
    print(f"[save] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
