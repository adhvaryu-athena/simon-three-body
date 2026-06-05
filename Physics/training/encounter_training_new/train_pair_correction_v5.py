"""
train_pair_correction_v5.py  --  Phase 3 Zone 3 NN trainer

Phase 3 improvements over v4:
----------------------------------------------
1. dt < 0.02 EXCLUDED at data load.
   Fine-dt samples (0.005, 0.01) have near-identity targets and add chaos noise
   in deployment. They are not generated in Phase 3, but this guard handles
   any accidental mixing with old data.

2. Wider architecture: 6 -> 128 -> 128 -> 128 -> 1  (was 64).
   The p95 absolute error in c was 0.146 in v4. For encounters at dt=0.08
   where c_true~0.73, that is a 20% relative miss. Wider hidden layers improve
   representational capacity for the 6D input space.

3. Residual connection: skip path from input directly to output.
   raw_output = main_MLP_output(x) + skip_linear(x)
   This lets the network learn "c = baseline(inputs) + correction" rather than
   composing identity from three hidden layers. Stabilises the near-identity
   regime and improves convergence.

4. Asymmetric approach/recede loss.
   v4 treated approach and recede samples identically. But recede correction is
   harder to predict (larger magnitude, steeper gradient with dt) and was
   systematically biased in v4 (pred ~0.829 vs true ~0.794 at dt=0.04).
   v5 weights the recede MSE by recede_weight (default 1.5x) relative to approach.

5. Improvement-threshold weighting revised.
   Samples with improvement < 0.15 (near-identity) get weight 0.05
   (stronger downweight than v4's 0.35). Samples with improvement > 0.50 get
   weight 3.0. This shifts gradient toward high-payoff corrections.

Output: pair_correction_nn_v5.pt
        Same deployment interface as v4 -- evaluator needs no changes.

Example:
    python -B train_pair_correction_v5.py \\
        --data encounter_data_zone3_phase3.npz \\
        --epochs 5000 --hidden 128 \\
        --output pair_correction_nn_v5.pt
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

# Phase 3 valid dt values -- 0.005 and 0.01 excluded
VALID_DTS_P3 = {0.020, 0.040, 0.050, 0.060, 0.080, 0.100}
# Probe check uses the same dt list as Phase 2 for backward compatibility
DT_LIST = [0.020, 0.040, 0.050, 0.060, 0.080, 0.100]
EPS = 3e-4


# =============================================================================
# Model -- v5 with residual connection and wider hidden
# =============================================================================
class PairCorrectionNN(nn.Module):
    """
    6 -> H -> H -> H -> 1  (main path)
    6 -> 1                 (skip / residual path)
    raw = main + skip
    log(c) = log_c_min + (log_c_max - log_c_min) * sigmoid(raw)
    c in [c_min, c_max]

    The skip connection lets the network learn a linear baseline directly,
    which stabilises the near-identity regime and speeds convergence.
    Compatible with the v4 evaluator: the state_dict has the same key structure
    except for the added 'skip.weight' and 'skip.bias' keys.

    IMPORTANT: this model is NOT backwards compatible with v4 -- the evaluator
    must load v5 weights and use the v5 forward pass. The log_gate evaluator
    (pair_eval_v4_zone3_frozen_timeavg_v3_with_log_gate.py) will need a small
    update to accept v5 weights. See the model class update instructions below.
    """
    def __init__(self, hidden: int = 128, c_min: float = 0.25, c_max: float = 3.0):
        super().__init__()
        if not (0.0 < c_min < c_max):
            raise ValueError(f"Invalid bounds: c_min={c_min}, c_max={c_max}")

        # Main deep path
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        # Residual / skip path: direct linear mapping from input to output
        # Initialised to near-zero so early training starts from the main path
        self.skip = nn.Linear(6, 1, bias=True)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

        self.register_buffer("input_mean", torch.zeros(6))
        self.register_buffer("input_std",  torch.ones(6))
        self.register_buffer("log_c_min",
                             torch.tensor(float(np.log(c_min)), dtype=torch.float32))
        self.register_buffer("log_c_max",
                             torch.tensor(float(np.log(c_max)), dtype=torch.float32))

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.net(x_norm).squeeze(-1) + self.skip(x_norm).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.forward_raw(x)
        s   = torch.sigmoid(raw)
        return self.log_c_min + (self.log_c_max - self.log_c_min) * s


@dataclass
class DatasetBundle:
    features: np.ndarray
    targets:  np.ndarray
    raw:      Dict[str, np.ndarray]


@dataclass
class ProbeBundle:
    names:    List[str]
    features: np.ndarray
    targets:  np.ndarray
    nn_dist:  np.ndarray
    nn_c_med: np.ndarray


def _safe_std(x: np.ndarray) -> np.ndarray:
    s = x.std(axis=0).astype(np.float32)
    s[s < 1e-8] = 1.0
    return s


def _round_dt(log_dt: np.ndarray) -> np.ndarray:
    dt = np.exp(log_dt.astype(np.float64))
    return np.array([round(float(x), 4) for x in dt])


# =============================================================================
# Data loading with dt < 0.02 exclusion
# =============================================================================
def load_zone3_data(npz_path: str, c_min: float, c_max: float) -> DatasetBundle:
    data = np.load(npz_path)
    missing = [k for k in REQUIRED_FIELDS if k not in data.files]
    if missing:
        raise KeyError(f"{npz_path} is missing fields: {missing}")

    raw = {k: data[k].astype(np.float32) for k in REQUIRED_FIELDS}
    n_raw = len(raw["r_AU"])

    # --- Phase 3: drop dt < 0.02 ---
    dt_vals = np.exp(raw["log_dt"].astype(np.float64))
    keep    = dt_vals >= 0.019   # 0.02 with small float tolerance
    n_drop  = int(np.sum(~keep))
    if n_drop > 0:
        print(f"[data] Dropping {n_drop}/{n_raw} samples with dt < 0.02 "
              f"(no meaningful signal at fine dt; Phase 3 policy).")
        raw = {k: v[keep] for k, v in raw.items()}

    n = len(raw["r_AU"])
    if n == 0:
        raise ValueError("Dataset is empty after dt filtering.")

    for k, arr in raw.items():
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
    lo = float(np.log(c_min)); hi = float(np.log(c_max))
    targets   = np.clip(targets_unclipped, lo, hi).astype(np.float32)
    n_clipped = int(np.sum(np.abs(targets - targets_unclipped) > 1e-7))

    print(f"[data] Loaded {npz_path}")
    print(f"[data] samples after dt filter: {n}  (dropped {n_drop})")
    print(f"[data] c bounds: [{c_min:.4g}, {c_max:.4g}]  clipped: {n_clipped} ({n_clipped/max(n,1):.2%})")
    print("[data] feature ranges:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"       {name:12s} min={features[:, i].min(): .4f}  "
              f"med={np.median(features[:, i]): .4f}  max={features[:, i].max(): .4f}")
    print(f"[data] c_opt: min={raw['c_opt'].min():.5f}  "
          f"med={np.median(raw['c_opt']):.5f}  max={raw['c_opt'].max():.5f}")

    # Per-dt breakdown
    dt_arr    = np.exp(raw["log_dt"].astype(np.float64))
    dt_rounded = np.array([round(float(x), 4) for x in dt_arr])
    c_arr     = raw["c_opt"].astype(np.float64)
    vr_arr    = raw["v_rad_norm"].astype(np.float64)
    print("[data] per-dt target medians:")
    for d in sorted(set(dt_rounded)):
        mask = dt_rounded == d
        app  = mask & (vr_arr < 0)
        rec  = mask & (vr_arr >= 0)
        print(f"       dt={d:0.4f}  n={int(mask.sum()):5d}  "
              f"c_med={np.median(c_arr[mask]):.5f}  "
              f"app_med={np.median(c_arr[app]) if np.any(app) else float('nan'):.5f}  "
              f"rec_med={np.median(c_arr[rec]) if np.any(rec) else float('nan'):.5f}")

    return DatasetBundle(features=features, targets=targets, raw=raw)


# =============================================================================
# Train/val/test split
# =============================================================================
def make_splits(n: int, seed: int,
                train_frac: float = 0.70,
                val_frac:   float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng   = np.random.RandomState(seed)
    idx   = rng.permutation(n)
    nt    = int(round(train_frac * n))
    nv    = int(round(val_frac * n))
    tr    = idx[:nt]
    va    = idx[nt:nt + nv]
    te    = idx[nt + nv:]
    if len(te) == 0:
        raise ValueError("Test split empty; use a larger dataset.")
    return tr, va, te


# =============================================================================
# Sample weights -- Phase 3 revised thresholds
# =============================================================================
def make_sample_weights(raw: Dict[str, np.ndarray], indices: np.ndarray,
                        strong_weight:    float,
                        approach_weight:  float,
                        low_impr_weight:  float,
                        high_impr_weight: float,
                        low_c_weight:     float) -> np.ndarray:
    """
    Phase 3 weighting scheme:
      - near-identity (imp < 0.15): weight = low_impr_weight (default 0.05)
        Strong downweight. NN should pass through; heavy fitting causes noise.
      - high-improvement (imp > 0.50): weight += high_impr_weight (default 3.0)
        These are the cases where the NN genuinely matters.
      - strong approach at large dt: extra weight (probe-fix region)
      - recede is handled by asymmetric loss in train_model, not here
    """
    r     = raw["r_AU"][indices].astype(np.float64)
    dt    = np.exp(raw["log_dt"][indices].astype(np.float64))
    vr    = raw["v_rad_norm"][indices].astype(np.float64)
    vt    = raw["v_tan_norm"][indices].astype(np.float64)
    c     = raw["c_opt"][indices].astype(np.float64)
    impr  = raw["improvement"][indices].astype(np.float64)

    w = np.ones(len(indices), dtype=np.float32)

    near_identity = impr < 0.15
    high_impr     = impr > 0.50
    large_dt      = dt >= 0.039
    approach      = vr < 0
    probe_region  = large_dt & approach & (r >= 0.088) & (r <= 0.112) & (vt >= 0.40) & (vt <= 0.90)
    low_c         = c < 0.45

    # Near-identity: reduce to low_impr_weight (Phase 3: 0.05, was 0.35 in v4)
    w[near_identity] = float(low_impr_weight)
    # High-improvement: boost
    w[high_impr] += float(high_impr_weight)
    # Strong approach in probe-fix region: boost (fixes OFF_MANIFOLD probe)
    w += float(approach_weight) * (large_dt & approach).astype(np.float32)
    w += float(strong_weight)   * probe_region.astype(np.float32)
    # Low-c samples: protect from collapse
    w += float(low_c_weight) * low_c.astype(np.float32)

    return w.astype(np.float32)


# =============================================================================
# Evaluation helpers
# =============================================================================
def evaluate_arrays(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    y_true_log = y_true_log.astype(np.float64)
    y_pred_log = y_pred_log.astype(np.float64)
    err_log    = y_pred_log - y_true_log
    c_true     = np.exp(y_true_log)
    c_pred     = np.exp(y_pred_log)
    err_c      = c_pred - c_true
    return {
        "mse_log":        float(np.mean(err_log ** 2)),
        "rmse_log":       float(np.sqrt(np.mean(err_log ** 2))),
        "mae_log":        float(np.mean(np.abs(err_log))),
        "mae_c":          float(np.mean(np.abs(err_c))),
        "median_abs_c":   float(np.median(np.abs(err_c))),
        "p90_abs_c":      float(np.percentile(np.abs(err_c), 90)),
        "p95_abs_c":      float(np.percentile(np.abs(err_c), 95)),
        "corr_c":         float(np.corrcoef(c_true, c_pred)[0, 1]) if len(c_true) > 2 else float("nan"),
        "pred_c_min":     float(np.min(c_pred)),
        "pred_c_med":     float(np.median(c_pred)),
        "pred_c_max":     float(np.max(c_pred)),
    }


def predict_numpy(model: PairCorrectionNN, features: np.ndarray,
                  device: str, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            xb = torch.tensor(features[i:i + batch_size], dtype=torch.float32, device=device)
            out.append(model(xb).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(out, axis=0)


# =============================================================================
# Probe bundle (same logic as v4, adapted for Phase 3 dt list)
# =============================================================================
def build_probe_inputs() -> Tuple[List[str], np.ndarray]:
    names: List[str] = []
    rows:  List[List[float]] = []
    r_soft = float(np.sqrt(0.10 ** 2 + EPS ** 2))
    for d in DT_LIST:
        for label, vr in [("approach", -0.8), ("side", 0.0), ("recede", +0.8)]:
            names.append(f"A_{label}_dt{d:.4f}")
            rows.append([np.log(r_soft), np.log(1.0), np.log(0.01), np.log(d), vr, 0.65])
    for vt in [0.1, 0.3, 0.6, 0.9, 1.2, 1.6]:
        names.append(f"B_recede_dt0.0400_vt{vt:.1f}")
        rows.append([np.log(r_soft), np.log(1.0), np.log(0.01), np.log(0.04), +0.8, vt])
    return names, np.array(rows, dtype=np.float32)


def make_probe_bundle(features: np.ndarray, targets: np.ndarray,
                      train_idx: np.ndarray,
                      feat_mean: np.ndarray, feat_std: np.ndarray,
                      k: int = 16) -> ProbeBundle:
    names, X_probe = build_probe_inputs()
    X_train   = features[train_idx].astype(np.float64)
    y_train   = targets[train_idx].astype(np.float64)
    dt_train  = _round_dt(X_train[:, 3])
    mean = feat_mean.astype(np.float64)
    std  = feat_std.astype(np.float64) + 1e-8
    Xn_train = (X_train - mean) / std
    Xn_probe = (X_probe.astype(np.float64) - mean) / std

    probe_targets = []
    probe_dists   = []
    probe_c_meds  = []
    for xp, xnp in zip(X_probe, Xn_probe):
        d_probe = round(float(np.exp(float(xp[3]))), 4)
        same_dt = np.where(dt_train == d_probe)[0]
        if len(same_dt) == 0:
            same_dt = np.arange(len(Xn_train))
        diff    = Xn_train[same_dt] - xnp
        dist    = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        order   = np.argsort(dist)[:max(1, min(k, len(dist)))]
        nn_local = same_dt[order]
        c_vals  = np.exp(y_train[nn_local])
        c_med   = float(np.median(c_vals))
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


# =============================================================================
# Loss helpers
# =============================================================================
def weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                 weight: torch.Tensor) -> torch.Tensor:
    return torch.sum(weight * (pred - target) ** 2) / torch.sum(weight.clamp_min(1e-8))


def asymmetric_mse(pred:         torch.Tensor,
                   target:        torch.Tensor,
                   weight:        torch.Tensor,
                   vr_batch:      torch.Tensor,
                   recede_weight: float) -> torch.Tensor:
    """
    Compute weighted MSE with extra pressure on recede samples.

    Approach (vr < 0) and recede (vr >= 0) contributions are computed
    separately, then combined:
        loss = L_approach + recede_weight * L_recede

    This directly targets the systematic under-prediction of the recede
    correction that was observed in Phase 2 (pred ~0.829 vs true ~0.794).
    """
    sq = weight * (pred - target) ** 2
    w_sum_all = torch.sum(weight).clamp_min(1e-8)

    approach_mask = (vr_batch < 0)
    recede_mask   = ~approach_mask

    if approach_mask.any():
        L_app = torch.sum(sq[approach_mask]) / torch.sum(weight[approach_mask]).clamp_min(1e-8)
    else:
        L_app = torch.tensor(0.0, device=pred.device)

    if recede_mask.any():
        L_rec = torch.sum(sq[recede_mask]) / torch.sum(weight[recede_mask]).clamp_min(1e-8)
    else:
        L_rec = torch.tensor(0.0, device=pred.device)

    # Normalise the combined loss to keep total magnitude comparable to
    # vanilla MSE (avoids needing to re-tune lr)
    n_app = float(approach_mask.sum().item())
    n_rec = float(recede_mask.sum().item())
    total = n_app + float(recede_weight) * n_rec
    if total < 1e-8:
        return torch.sum(sq) / w_sum_all

    return (n_app * L_app + float(recede_weight) * n_rec * L_rec) / total


# =============================================================================
# Training loop
# =============================================================================
def train_model(bundle:           DatasetBundle,
                hidden:           int,
                epochs:           int,
                lr:               float,
                seed:             int,
                device:           str,
                batch_size:       int,
                output_prefix:    str,
                c_min:            float,
                c_max:            float,
                strong_weight:    float,
                approach_weight:  float,
                low_impr_weight:  float,
                high_impr_weight: float,
                low_c_weight:     float,
                recede_weight:    float,
                probe_weight:     float,
                probe_k:          int) -> Tuple[PairCorrectionNN, Dict, Dict]:

    torch.manual_seed(seed)
    np.random.seed(seed)

    features  = bundle.features
    targets   = bundle.targets
    n         = len(features)
    train_idx, val_idx, test_idx = make_splits(n, seed=seed)

    feat_mean = features[train_idx].mean(axis=0).astype(np.float32)
    feat_std  = _safe_std(features[train_idx])

    train_weights = make_sample_weights(
        bundle.raw, train_idx,
        strong_weight=strong_weight,
        approach_weight=approach_weight,
        low_impr_weight=low_impr_weight,
        high_impr_weight=high_impr_weight,
        low_c_weight=low_c_weight,
    )

    print("\n[weights] training sample weights")
    print(f"  min/med/max = {train_weights.min():.3f} / {np.median(train_weights):.3f} / {train_weights.max():.3f}")
    print(f"  mean        = {train_weights.mean():.3f}")

    probe_bundle = make_probe_bundle(features, targets, train_idx, feat_mean, feat_std, k=probe_k)

    model = PairCorrectionNN(hidden=hidden, c_min=c_min, c_max=c_max).to(device)
    model.input_mean = torch.tensor(feat_mean, dtype=torch.float32, device=device)
    model.input_std  = torch.tensor(feat_std,  dtype=torch.float32, device=device)

    X_train = torch.tensor(features[train_idx],   dtype=torch.float32, device=device)
    y_train = torch.tensor(targets[train_idx],    dtype=torch.float32, device=device)
    w_train = torch.tensor(train_weights,         dtype=torch.float32, device=device)
    vr_train = torch.tensor(
        bundle.raw["v_rad_norm"][train_idx].astype(np.float32),
        dtype=torch.float32, device=device
    )
    X_val   = torch.tensor(features[val_idx],     dtype=torch.float32, device=device)
    y_val   = torch.tensor(targets[val_idx],      dtype=torch.float32, device=device)
    X_probe = torch.tensor(probe_bundle.features, dtype=torch.float32, device=device)
    y_probe = torch.tensor(probe_bundle.targets,  dtype=torch.float32, device=device)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr / 100.0)
    mse_plain = nn.MSELoss()

    best_val    = float("inf")
    best_state  = None
    train_hist  = []
    val_hist    = []
    probe_hist  = []
    t0 = time.perf_counter()

    n_train = len(train_idx)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        ep_loss_sum = 0.0
        ep_seen     = 0

        for start in range(0, n_train, batch_size):
            ids  = perm[start:start + batch_size]
            xb   = X_train[ids]
            yb   = y_train[ids]
            wb   = w_train[ids]
            vrb  = vr_train[ids]

            pred = model(xb)

            # Asymmetric approach/recede loss
            data_loss = asymmetric_mse(pred, yb, wb, vrb, recede_weight)

            if probe_weight > 0.0:
                probe_pred = model(X_probe)
                probe_loss = mse_plain(probe_pred, y_probe)
                loss = data_loss + float(probe_weight) * probe_loss
            else:
                probe_loss = torch.tensor(0.0, device=device)
                loss = data_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            ep_loss_sum += float(data_loss.item()) * len(ids)
            ep_seen     += len(ids)

        sched.step()
        train_loss = ep_loss_sum / max(ep_seen, 1)

        model.eval()
        with torch.no_grad():
            val_loss       = float(mse_plain(model(X_val), y_val).item())
            probe_loss_val = float(mse_plain(model(X_probe), y_probe).item())

        train_hist.append(train_loss)
        val_hist.append(val_loss)
        probe_hist.append(probe_loss_val)

        selection_score = val_loss + float(probe_weight) * probe_loss_val
        if selection_score < best_val:
            best_val   = selection_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep == 0 or (ep + 1) % max(1, epochs // 10) == 0 or ep + 1 == epochs:
            print(f"  ep {ep+1:5d}/{epochs}  "
                  f"train={train_loss:.7f}  val={val_loss:.7f}  probe={probe_loss_val:.7f}")

    elapsed = time.perf_counter() - t0
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

    pred_test = predict_numpy(model, features[test_idx], device)
    metrics   = {}
    for nm, idx, pred in [
        ("train", train_idx, predict_numpy(model, features[train_idx], device)),
        ("val",   val_idx,   predict_numpy(model, features[val_idx],   device)),
        ("test",  test_idx,  pred_test),
    ]:
        for k, v in evaluate_arrays(targets[idx].astype(np.float64), pred).items():
            metrics[f"{nm}_{k}"] = v

    pred_probe = predict_numpy(model, probe_bundle.features, device)
    for k, v in evaluate_arrays(probe_bundle.targets.astype(np.float64), pred_probe).items():
        metrics[f"probe_{k}"] = v
    metrics["probe_hard_gate_frac"]        = float(np.mean((np.exp(pred_probe) < 0.2) | (np.exp(pred_probe) > 5.0)))
    metrics["probe_conservative_gate_frac"]= float(np.mean((np.exp(pred_probe) < 0.4) | (np.exp(pred_probe) > 2.5)))
    metrics["best_selection_score"]        = float(best_val)
    metrics["elapsed_sec"]                 = float(elapsed)
    metrics["n_train"]                     = float(len(train_idx))
    metrics["n_val"]                       = float(len(val_idx))
    metrics["n_test"]                      = float(len(test_idx))

    histories = {
        "train_loss":     np.array(train_hist,  dtype=np.float64),
        "val_loss":       np.array(val_hist,    dtype=np.float64),
        "probe_loss":     np.array(probe_hist,  dtype=np.float64),
        "train_idx":      train_idx,
        "val_idx":        val_idx,
        "test_idx":       test_idx,
        "probe_features": probe_bundle.features,
        "probe_targets":  probe_bundle.targets,
        "probe_names":    np.array(probe_bundle.names),
    }
    return model, histories, metrics


# =============================================================================
# Post-training diagnostics (same structure as v4 for familiar output)
# =============================================================================
def print_test_diagnostics(bundle:         DatasetBundle,
                           model:          PairCorrectionNN,
                           test_indices:   np.ndarray,
                           device:         str,
                           probe_features: np.ndarray,
                           probe_targets:  np.ndarray,
                           probe_names:    np.ndarray) -> None:
    features = bundle.features[test_indices]
    targets  = bundle.targets[test_indices].astype(np.float64)
    pred_log = predict_numpy(model, features, device)
    c_true   = np.exp(targets)
    c_pred   = np.exp(pred_log)
    raw      = bundle.raw
    dt_vals  = np.exp(raw["log_dt"][test_indices].astype(np.float64))
    rounded  = np.array([round(float(x), 4) for x in dt_vals])
    vr       = raw["v_rad_norm"][test_indices].astype(np.float64)
    impr     = raw["improvement"][test_indices].astype(np.float64)

    print("\n[test] Overall held-out metrics")
    m = evaluate_arrays(targets, pred_log)
    for k in ["mse_log", "rmse_log", "mae_log", "mae_c", "median_abs_c", "p95_abs_c", "corr_c"]:
        print(f"       {k:14s}: {m[k]:.6g}")
    print(f"       pred c range  : {c_pred.min():.4f} / {np.median(c_pred):.4f} / {c_pred.max():.4f}")
    print(f"       true c range  : {c_true.min():.4f} / {np.median(c_true):.4f} / {c_true.max():.4f}")
    print(f"       out-of-gate (<0.2 or >5): {np.mean((c_pred < 0.2) | (c_pred > 5.0)):.2%}")
    print(f"       conservative  (<0.4 or >2.5): {np.mean((c_pred < 0.4) | (c_pred > 2.5)):.2%}")

    print("\n[test] Per-dt medians on held-out test set")
    print(f"  {'dt':>7} | {'n':>5} | {'true_med':>9} | {'pred_med':>9} | {'MAE_c':>9} | {'impr_med':>9}")
    print("  " + "-" * 61)
    for d in sorted(set(rounded)):
        mask = rounded == d
        print(f"  {d:7.4f} | {int(mask.sum()):5d} | {np.median(c_true[mask]):9.4f} | "
              f"{np.median(c_pred[mask]):9.4f} | {np.mean(np.abs(c_pred[mask]-c_true[mask])):9.4f} | "
              f"{np.median(impr[mask]):9.2%}")

    print("\n[test] Approaching vs receding medians (KEY CHECK for Phase 3)")
    print(f"  {'dt':>7} | {'app_n':>5} | {'app_true':>9} | {'app_pred':>9} | {'rec_n':>5} | {'rec_true':>9} | {'rec_pred':>9}")
    print("  " + "-" * 85)
    signs_ok = 0; signs_total = 0
    for d in sorted(set(rounded)):
        base = rounded == d
        app  = base & (vr < 0)
        rec  = base & (vr >= 0)
        def med(a, mask): return float(np.median(a[mask])) if np.any(mask) else float("nan")
        app_t, app_p = med(c_true, app), med(c_pred, app)
        rec_t, rec_p = med(c_true, rec), med(c_pred, rec)
        if np.isfinite(app_p) and np.isfinite(rec_p):
            signs_total += 1
            signs_ok    += int(app_p > rec_p)
        bias_rec = rec_p - rec_t if (np.isfinite(rec_p) and np.isfinite(rec_t)) else float("nan")
        print(f"  {d:7.4f} | {int(app.sum()):5d} | {app_t:9.4f} | {app_p:9.4f} | "
              f"{int(rec.sum()):5d} | {rec_t:9.4f} | {rec_p:9.4f}  bias_rec={bias_rec:+.4f}")
    print(f"\n[test] app_pred > rec_pred: {signs_ok}/{signs_total} dt groups")
    print("  [Phase 3 target: recede bias |rec_pred - rec_true| < 0.03 at all dts]")

    # Fixed-state probe table
    print("\n[probe] Fixed-state velocity-direction check")
    print("        r=0.10 AU, m_i=1, m_j=0.01, v_tan_norm=0.65")
    print(f"  {'dt':>7} | {'approach(vr=-0.8)':>18} | {'side(vr=0)':>12} | {'recede(vr=+0.8)':>17}")
    print("  " + "-" * 66)
    names_p, rows_p = build_probe_inputs()
    probe_pred_all  = np.exp(predict_numpy(model, rows_p, device))
    n2p = dict(zip(names_p, probe_pred_all))
    for d in DT_LIST:
        ca = n2p.get(f"A_approach_dt{d:.4f}", float("nan"))
        cs = n2p.get(f"A_side_dt{d:.4f}",    float("nan"))
        cr = n2p.get(f"A_recede_dt{d:.4f}",  float("nan"))
        flag = ""
        if any(v < 0.2 for v in [ca, cs, cr] if np.isfinite(v)):
            flag = "  <-- HARD-GATE WARNING"
        elif any(v < 0.4 for v in [ca, cs, cr] if np.isfinite(v)):
            flag = "  <-- conservative warning"
        print(f"  {d:7.4f} | {ca:18.5f} | {cs:12.5f} | {cr:17.5f}{flag}")

    print("\n[probe] Data-derived probe anchors vs prediction")
    pred_log_p = predict_numpy(model, probe_features, device)
    pred_c_p   = np.exp(pred_log_p)
    target_c_p = np.exp(probe_targets.astype(np.float64))
    print(f"  {'probe':>26} | {'target_c':>9} | {'pred_c':>9} | {'abs_err':>9}")
    print("  " + "-" * 61)
    for nm, tc, pc in zip(probe_names, target_c_p, pred_c_p):
        flag = " <-- OFF" if abs(pc - tc) > 0.10 else ""
        print(f"  {str(nm):>26s} | {tc:9.4f} | {pc:9.4f} | {abs(pc-tc):9.4f}{flag}")


# =============================================================================
# Plotting and summary
# =============================================================================
def maybe_plot_training(histories: Dict, output_prefix: str) -> None:
    if not HAS_MPL:
        print("[plot] matplotlib not available; skipping.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(histories["train_loss"], label="weighted train (asym)")
    ax.plot(histories["val_loss"],   label="val")
    ax.plot(histories["probe_loss"], label="probe anchor")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log scale)")
    ax.set_title("Zone 3 v5 residual 6-input NN training curve")
    ax.grid(True, alpha=0.25); ax.legend()
    fig.tight_layout()
    out = "training_log_v5.png"
    fig.savefig(out, dpi=180); plt.close(fig)
    print(f"[plot] Saved {out}")


def write_summary(path: str, args: argparse.Namespace, metrics: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Zone 3 v5 Phase 3 training summary\n")
        f.write("=" * 72 + "\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write("\nMetrics\n" + "-" * 72 + "\n")
        for k in sorted(metrics):
            f.write(f"{k}: {metrics[k]}\n")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train SIMON Zone 3 v5 Phase 3 bounded residual NN.")
    ap.add_argument("--data", default="encounter_data_zone3_phase3.npz")
    ap.add_argument("--hidden", type=int, default=128,
                    help="Hidden width. Default 128 (was 64 in v4).")
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="pair_correction_nn_v5.pt")
    ap.add_argument("--c-min", type=float, default=0.25)
    ap.add_argument("--c-max", type=float, default=3.0)
    # Phase 3 weighting
    ap.add_argument("--strong-weight",    type=float, default=2.0,
                    help="Extra weight for approach samples in probe-fix region.")
    ap.add_argument("--approach-weight",  type=float, default=0.35,
                    help="Extra weight for large-dt approaching samples.")
    ap.add_argument("--low-impr-weight",  type=float, default=0.05,
                    help="Weight for near-identity samples (imp<0.15). Phase 3: 0.05 (was 0.35 in v4).")
    ap.add_argument("--high-impr-weight", type=float, default=3.0,
                    help="Extra weight for high-improvement samples (imp>0.50).")
    ap.add_argument("--low-c-weight",     type=float, default=0.50,
                    help="Extra weight for low-c samples.")
    ap.add_argument("--recede-weight",    type=float, default=1.5,
                    help="Multiplier on recede MSE vs approach MSE. Phase 3 fix for recede bias.")
    ap.add_argument("--probe-weight",     type=float, default=0.20)
    ap.add_argument("--probe-k",          type=int,   default=16)
    args = ap.parse_args()

    output_prefix = os.path.splitext(args.output)[0]
    print(f"[config] data={args.data}")
    print(f"[config] output={args.output}")
    print(f"[config] hidden={args.hidden} epochs={args.epochs} lr={args.lr} "
          f"batch_size={args.batch_size} seed={args.seed}")
    print(f"[config] device={args.device}")
    print(f"[config] c bounds=[{args.c_min}, {args.c_max}]")
    print(f"[config] dt<0.02 samples excluded at load (Phase 3 policy)")
    print(f"[config] recede_weight={args.recede_weight} (asymmetric loss)")
    print(f"[config] low_impr_weight={args.low_impr_weight} | high_impr_weight={args.high_impr_weight}")

    bundle = load_zone3_data(args.data, c_min=args.c_min, c_max=args.c_max)
    model, histories, metrics = train_model(
        bundle=bundle, hidden=args.hidden, epochs=args.epochs, lr=args.lr,
        seed=args.seed, device=args.device, batch_size=args.batch_size,
        output_prefix=output_prefix, c_min=args.c_min, c_max=args.c_max,
        strong_weight=args.strong_weight, approach_weight=args.approach_weight,
        low_impr_weight=args.low_impr_weight, high_impr_weight=args.high_impr_weight,
        low_c_weight=args.low_c_weight, recede_weight=args.recede_weight,
        probe_weight=args.probe_weight, probe_k=args.probe_k,
    )

    torch.save(model.cpu().state_dict(), args.output)
    print(f"\n[save] Saved model: {args.output} ({os.path.getsize(args.output)/1024:.1f} KB)")
    print(f"[save] Saved split: {output_prefix}_split.npz")

    model_cpu = PairCorrectionNN(hidden=args.hidden, c_min=args.c_min, c_max=args.c_max)
    model_cpu.load_state_dict(torch.load(args.output, map_location="cpu"))
    model_cpu.eval()
    print_test_diagnostics(
        bundle, model_cpu, histories["test_idx"], device="cpu",
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
    summary_path = "train_pair_correction_v5_summary.txt"
    write_summary(summary_path, args, metrics)
    print(f"[save] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
