"""
train_encounter_surrogate_v2_velocitysafe.py

Velocity-safe encounter-level residual surrogate for SIMON Zone-3 encounter windows.

Purpose
-------
This is the v2 trainer for the encounter-level surrogate.  It keeps the same
scientific target as v1:

    input  X = X_rel18  (relative 3-body state at Zone-3 encounter start)
    target Y = [residual_x, residual_v] flattened to 18 values
             = [IAS15_final - revised_noNN_final]

At inference:

    corrected_final_state = noNN_final_state + NN_predicted_residual

Why v2 exists
-------------
The v1 trainer used

    Loss = L_pos + 0.1 * L_vel

and selected the best checkpoint mostly by position error.  This produced strong
position improvement but weaker velocity safety.  v2 is designed to make the
surrogate dynamically safer before any full-rollout deployment:

  1. velocity loss weight defaults to 1.0, not 0.1;
  2. validation checkpoint selection penalizes velocity error and unsafe cases;
  3. optional physical consistency penalties discourage artificial COM/momentum
     drift relative to the IAS15 encounter-exit state;
  4. the summary reports position, velocity, and combined unsafe fractions.

Expected data format
--------------------
Use data produced by generate_encounter_surrogate_data_v1.py and merged by
merge_encounter_surrogate_shards_v1.py. Required fields include:

    X_rel18, residual_x, residual_v,
    x_ias_final, v_ias_final, x_nonn_final, v_nonn_final,
    m, r_pair, v_rad_norm, v_tan_norm, relE_ias, relE_nonn,
    pos_residual_rms, vel_residual_rms, category_id

Example
-------
    python -B train_encounter_surrogate_v2_velocitysafe.py \
        --data encounter_surrogate_v1_large1.npz \
        --epochs 3000 \
        --hidden 128 \
        --output encounter_surrogate_v2_velocitysafe_large1.pt

Outputs
-------
    <output>.pt                         trained PyTorch model state_dict
    <output stem>_summary.txt           training/evaluation summary
    <output stem>_splits.npz            train/val/test indices
    <output stem>_predictions_test.npz  held-out predictions and errors
    <output stem>_loss.png              loss curve if matplotlib is available
    <output stem>_scatter_pos.png       noNN vs corrected position error
    <output stem>_scatter_vel.png       noNN vs corrected velocity error
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
# DataLoader/TensorDataset intentionally not used in v2.1.
# Manual batching avoids a rare Windows/PyTorch tuple-shadow crash.

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    plt = None


REQUIRED_FIELDS = [
    "X_rel18",
    "residual_x", "residual_v",
    "x_ias_final", "v_ias_final", "x_nonn_final", "v_nonn_final",
    "m",
    "r_pair", "v_rad_norm", "v_tan_norm",
    "relE_ias", "relE_nonn",
    "pos_residual_rms", "vel_residual_rms",
    "category_id",
]

CATEGORY_NAMES = {
    0: "strong_approach",
    1: "weak_side",
    2: "recede",
    3: "broad",
}


# =============================================================================
# Model
# =============================================================================
class EncounterResidualMLP(nn.Module):
    """
    Fixed-size MLP for a 3-body, fixed-pair encounter surrogate.

    Input:
        X_rel18 normalized by input_mean/input_std.
    Output:
        normalized 18D residual state:
        [residual_x_flat9, residual_v_flat9].
    """

    def __init__(self, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        if hidden < 16:
            raise ValueError("hidden should be at least 16")
        if not (0.0 <= dropout < 0.5):
            raise ValueError("dropout must be in [0, 0.5)")

        mid = max(64, hidden // 2)
        self.net = nn.Sequential(
            nn.Linear(18, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, mid), nn.SiLU(),
            nn.Linear(mid, 18),
        )

        self.register_buffer("input_mean", torch.zeros(18, dtype=torch.float32))
        self.register_buffer("input_std", torch.ones(18, dtype=torch.float32))
        self.register_buffer("target_mean", torch.zeros(18, dtype=torch.float32))
        self.register_buffer("target_std", torch.ones(18, dtype=torch.float32))

    def forward_norm(self, x_raw: torch.Tensor) -> torch.Tensor:
        x = (x_raw - self.input_mean) / (self.input_std + 1e-8)
        return self.net(x)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Return residual prediction in physical/original units."""
        y_norm = self.forward_norm(x_raw)
        return y_norm * (self.target_std + 1e-8) + self.target_mean


@dataclass
class SurrogateData:
    X: np.ndarray              # (n,18)
    Y: np.ndarray              # (n,18), [resid_x9, resid_v9]
    raw: Dict[str, np.ndarray]


# =============================================================================
# Data loading and checks
# =============================================================================
def _safe_std(a: np.ndarray, axis: int = 0) -> np.ndarray:
    s = np.std(a, axis=axis).astype(np.float32)
    s[~np.isfinite(s)] = 1.0
    s[s < 1e-8] = 1.0
    return s


def _rms_state_pos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return np.sqrt(np.mean(np.sum(d * d, axis=2), axis=1))


def _rms_state_vel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return np.sqrt(np.mean(np.sum(d * d, axis=2), axis=1))


def load_data(path: str) -> SurrogateData:
    data = np.load(path)
    missing = [k for k in REQUIRED_FIELDS if k not in data.files]
    if missing:
        raise KeyError(f"{path} is missing required fields: {missing}")

    raw: Dict[str, np.ndarray] = {k: data[k] for k in data.files}

    X = np.asarray(raw["X_rel18"], dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 18:
        raise ValueError(f"X_rel18 must have shape (n,18), got {X.shape}")

    rx = np.asarray(raw["residual_x"], dtype=np.float32)
    rv = np.asarray(raw["residual_v"], dtype=np.float32)
    if rx.ndim != 3 or rx.shape[1:] != (3, 3):
        raise ValueError(f"residual_x must have shape (n,3,3), got {rx.shape}")
    if rv.ndim != 3 or rv.shape[1:] != (3, 3):
        raise ValueError(f"residual_v must have shape (n,3,3), got {rv.shape}")

    n = X.shape[0]
    for k in REQUIRED_FIELDS:
        if len(raw[k]) != n:
            raise ValueError(f"Field {k} length {len(raw[k])} does not match X length {n}")
        if not np.all(np.isfinite(raw[k])):
            raise ValueError(f"Field {k} contains non-finite values")

    m = np.asarray(raw["m"], dtype=np.float32)
    if m.ndim != 2 or m.shape[1] != 3:
        raise ValueError(f"m must have shape (n,3), got {m.shape}")
    if np.any(m <= 0):
        raise ValueError("All masses must be positive")

    Y = np.concatenate([rx.reshape(n, 9), rv.reshape(n, 9)], axis=1).astype(np.float32)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("X or Y contains non-finite values")

    # Verify residual fields exactly match IAS15 - noNN within float tolerance.
    x_ias = np.asarray(raw["x_ias_final"], dtype=np.float64)
    x_no = np.asarray(raw["x_nonn_final"], dtype=np.float64)
    v_ias = np.asarray(raw["v_ias_final"], dtype=np.float64)
    v_no = np.asarray(raw["v_nonn_final"], dtype=np.float64)
    if not np.allclose(rx.astype(np.float64), x_ias - x_no, rtol=2e-5, atol=2e-8):
        raise ValueError("residual_x does not match x_ias_final - x_nonn_final")
    if not np.allclose(rv.astype(np.float64), v_ias - v_no, rtol=2e-5, atol=2e-8):
        raise ValueError("residual_v does not match v_ias_final - v_nonn_final")

    return SurrogateData(X=X, Y=Y, raw=raw)


def make_stratified_splits(raw: Dict[str, np.ndarray], seed: int,
                           train_frac: float = 0.70,
                           val_frac: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratify by category_id so strong/weak/recede appear in all splits."""
    cat = np.asarray(raw["category_id"], dtype=np.int64)
    rng = np.random.RandomState(seed)

    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []

    for c in sorted(set(cat.tolist())):
        idx = np.where(cat == c)[0]
        rng.shuffle(idx)
        if len(idx) < 3:
            train_parts.append(idx)
            continue
        n_train = int(round(train_frac * len(idx)))
        n_val = int(round(val_frac * len(idx)))
        if n_train + n_val >= len(idx):
            n_val = max(1, len(idx) - n_train - 1)
        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train:n_train + n_val])
        test_parts.append(idx[n_train + n_val:])

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"Bad split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    if len(set(train_idx) & set(val_idx)) or len(set(train_idx) & set(test_idx)) or len(set(val_idx) & set(test_idx)):
        raise ValueError("Split overlap detected")

    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


# =============================================================================
# Metrics and physics helpers
# =============================================================================
def _com_position(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    M = np.sum(m, axis=1, keepdims=True)
    return np.sum(x * m[:, :, None], axis=1) / (M + 1e-30)


def _com_velocity(v: np.ndarray, m: np.ndarray) -> np.ndarray:
    M = np.sum(m, axis=1, keepdims=True)
    return np.sum(v * m[:, :, None], axis=1) / (M + 1e-30)


def _torch_com_position(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    M = torch.sum(m, dim=1, keepdim=True)
    return torch.sum(x * m[:, :, None], dim=1) / (M + 1e-30)


def _torch_com_velocity(v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    M = torch.sum(m, dim=1, keepdim=True)
    return torch.sum(v * m[:, :, None], dim=1) / (M + 1e-30)


def _torch_energy(x: torch.Tensor, v: torch.Tensor, m: torch.Tensor, eps: float = 3e-4) -> torch.Tensor:
    """Softened total energy per sample; used only if energy penalty is enabled."""
    ke = 0.5 * torch.sum(m * torch.sum(v * v, dim=2), dim=1)
    pe = torch.zeros_like(ke)
    for i in range(3):
        for j in range(i + 1, 3):
            rij = x[:, i, :] - x[:, j, :]
            r_soft = torch.sqrt(torch.sum(rij * rij, dim=1) + eps * eps)
            pe = pe - m[:, i] * m[:, j] / (r_soft + 1e-30)
    return ke + pe


def compute_metrics(raw: Dict[str, np.ndarray], indices: np.ndarray,
                    pred_resid: np.ndarray, unsafe_factor: float = 2.0) -> Dict[str, float]:
    """
    Evaluate corrected_final = noNN_final + predicted residual against IAS15.
    pred_resid shape: (n_subset,18), original units.
    """
    idx = indices
    n = len(idx)
    prx = pred_resid[:, :9].reshape(n, 3, 3).astype(np.float64)
    prv = pred_resid[:, 9:].reshape(n, 3, 3).astype(np.float64)

    x_ias = np.asarray(raw["x_ias_final"], dtype=np.float64)[idx]
    v_ias = np.asarray(raw["v_ias_final"], dtype=np.float64)[idx]
    x_no = np.asarray(raw["x_nonn_final"], dtype=np.float64)[idx]
    v_no = np.asarray(raw["v_nonn_final"], dtype=np.float64)[idx]
    m = np.asarray(raw["m"], dtype=np.float64)[idx]

    x_corr = x_no + prx
    v_corr = v_no + prv

    base_pos = _rms_state_pos(x_no, x_ias)
    corr_pos = _rms_state_pos(x_corr, x_ias)
    base_vel = _rms_state_vel(v_no, v_ias)
    corr_vel = _rms_state_vel(v_corr, v_ias)

    pos_impr = (base_pos - corr_pos) / (base_pos + 1e-30)
    vel_impr = (base_vel - corr_vel) / (base_vel + 1e-30)

    true_resid = np.concatenate([
        (x_ias - x_no).reshape(n, 9),
        (v_ias - v_no).reshape(n, 9),
    ], axis=1)
    resid_err = pred_resid.astype(np.float64) - true_resid
    resid_pos_err = np.sqrt(np.mean(np.sum(resid_err[:, :9].reshape(n, 3, 3) ** 2, axis=2), axis=1))
    resid_vel_err = np.sqrt(np.mean(np.sum(resid_err[:, 9:].reshape(n, 3, 3) ** 2, axis=2), axis=1))

    unsafe_pos = corr_pos > (unsafe_factor * base_pos + 1e-12)
    unsafe_vel = corr_vel > (unsafe_factor * base_vel + 1e-12)
    unsafe_any = unsafe_pos | unsafe_vel

    com_x_err = np.sqrt(np.sum((_com_position(x_corr, m) - _com_position(x_ias, m)) ** 2, axis=1))
    com_v_err = np.sqrt(np.sum((_com_velocity(v_corr, m) - _com_velocity(v_ias, m)) ** 2, axis=1))

    return {
        "n": float(n),
        "base_pos_mean": float(np.mean(base_pos)),
        "base_pos_median": float(np.median(base_pos)),
        "base_pos_p95": float(np.percentile(base_pos, 95)),
        "corr_pos_mean": float(np.mean(corr_pos)),
        "corr_pos_median": float(np.median(corr_pos)),
        "corr_pos_p95": float(np.percentile(corr_pos, 95)),
        "pos_improvement_mean_pct": float(100.0 * np.mean(pos_impr)),
        "pos_improvement_median_pct": float(100.0 * np.median(pos_impr)),
        "pos_success_frac": float(np.mean(corr_pos < base_pos)),
        "base_vel_mean": float(np.mean(base_vel)),
        "base_vel_median": float(np.median(base_vel)),
        "base_vel_p95": float(np.percentile(base_vel, 95)),
        "corr_vel_mean": float(np.mean(corr_vel)),
        "corr_vel_median": float(np.median(corr_vel)),
        "corr_vel_p95": float(np.percentile(corr_vel, 95)),
        "vel_improvement_mean_pct": float(100.0 * np.mean(vel_impr)),
        "vel_improvement_median_pct": float(100.0 * np.median(vel_impr)),
        "vel_success_frac": float(np.mean(corr_vel < base_vel)),
        "resid_pos_err_mean": float(np.mean(resid_pos_err)),
        "resid_pos_err_median": float(np.median(resid_pos_err)),
        "resid_vel_err_mean": float(np.mean(resid_vel_err)),
        "resid_vel_err_median": float(np.median(resid_vel_err)),
        "unsafe_pos_frac": float(np.mean(unsafe_pos)),
        "unsafe_vel_frac": float(np.mean(unsafe_vel)),
        "unsafe_any_frac": float(np.mean(unsafe_any)),
        "com_pos_err_median": float(np.median(com_x_err)),
        "com_vel_err_median": float(np.median(com_v_err)),
        "com_pos_err_p95": float(np.percentile(com_x_err, 95)),
        "com_vel_err_p95": float(np.percentile(com_v_err, 95)),
    }


def predict_residual(model: EncounterResidualMLP, X: np.ndarray,
                     device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            yb = model(xb).detach().cpu().numpy().astype(np.float32)
            outs.append(yb)
    return np.concatenate(outs, axis=0)


def category_table(raw: Dict[str, np.ndarray], indices: np.ndarray,
                   pred_resid: np.ndarray, unsafe_factor: float) -> List[str]:
    lines: List[str] = []
    cat = np.asarray(raw["category_id"], dtype=np.int64)[indices]
    for c in sorted(set(cat.tolist())):
        mask = cat == c
        idx_c = indices[mask]
        pred_c = pred_resid[mask]
        m = compute_metrics(raw, idx_c, pred_c, unsafe_factor=unsafe_factor)
        name = CATEGORY_NAMES.get(int(c), f"category_{int(c)}")
        lines.append(
            f"  {name:16s} n={int(m['n']):5d} "
            f"pos_med {m['base_pos_median']:.4e}->{m['corr_pos_median']:.4e} "
            f"vel_med {m['base_vel_median']:.4e}->{m['corr_vel_median']:.4e} "
            f"pos_impr={m['pos_improvement_median_pct']:+7.2f}% "
            f"vel_impr={m['vel_improvement_median_pct']:+7.2f}% "
            f"unsafe_pos={m['unsafe_pos_frac']:.1%} unsafe_vel={m['unsafe_vel_frac']:.1%}"
        )
    return lines


# =============================================================================
# Training
# =============================================================================
def train_one_model(args) -> None:
    t0 = time.perf_counter()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data = load_data(args.data)
    X, Y, raw = data.X, data.Y, data.raw
    n = len(X)
    if n < 100:
        raise ValueError(f"Dataset is too small for training: n={n}")

    train_idx, val_idx, test_idx = make_stratified_splits(raw, args.seed)

    x_mean = X[train_idx].mean(axis=0).astype(np.float32)
    x_std = _safe_std(X[train_idx], axis=0)
    y_mean = Y[train_idx].mean(axis=0).astype(np.float32)
    y_std = _safe_std(Y[train_idx], axis=0)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = EncounterResidualMLP(hidden=args.hidden, dropout=args.dropout).to(device)
    with torch.no_grad():
        model.input_mean.copy_(torch.tensor(x_mean, dtype=torch.float32, device=device))
        model.input_std.copy_(torch.tensor(x_std, dtype=torch.float32, device=device))
        model.target_mean.copy_(torch.tensor(y_mean, dtype=torch.float32, device=device))
        model.target_std.copy_(torch.tensor(y_std, dtype=torch.float32, device=device))

    # Training tensors.  Keep physical tensors too for COM/momentum penalties.
    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    Y_train_norm = torch.tensor((Y[train_idx] - y_mean) / (y_std + 1e-8), dtype=torch.float32)
    m_train = torch.tensor(np.asarray(raw["m"], dtype=np.float32)[train_idx], dtype=torch.float32)
    x_no_train = torch.tensor(np.asarray(raw["x_nonn_final"], dtype=np.float32)[train_idx], dtype=torch.float32)
    v_no_train = torch.tensor(np.asarray(raw["v_nonn_final"], dtype=np.float32)[train_idx], dtype=torch.float32)
    x_ias_train = torch.tensor(np.asarray(raw["x_ias_final"], dtype=np.float32)[train_idx], dtype=torch.float32)
    v_ias_train = torch.tensor(np.asarray(raw["v_ias_final"], dtype=np.float32)[train_idx], dtype=torch.float32)

    # Use manual batching instead of torch.utils.data.DataLoader/TensorDataset.
    # This avoids a rare Windows/PyTorch failure where TensorDataset.__getitem__
    # raises: TypeError: 'tuple' object is not callable.  The training logic is
    # unchanged: each epoch shuffles train rows and slices the same tensors.
    n_train = int(X_train.shape[0])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    huber = nn.SmoothL1Loss(reduction="none", beta=args.huber_beta)

    best_score = float("inf")
    best_state = None
    best_epoch = -1
    bad_epochs = 0
    train_losses: List[float] = []
    val_scores: List[float] = []

    pos_w = float(args.pos_weight)
    vel_w = float(args.vel_weight)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        perm = torch.randperm(n_train)
        for start in range(0, n_train, int(args.batch_size)):
            bidx = perm[start:start + int(args.batch_size)]
            xb = X_train[bidx].to(device, non_blocking=True)
            yb = Y_train_norm[bidx].to(device, non_blocking=True)
            mb = m_train[bidx].to(device, non_blocking=True)
            xno = x_no_train[bidx].to(device, non_blocking=True)
            vno = v_no_train[bidx].to(device, non_blocking=True)
            xias = x_ias_train[bidx].to(device, non_blocking=True)
            vias = v_ias_train[bidx].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred_norm = model.forward_norm(xb)
            elem = huber(pred_norm, yb)
            loss_pos = elem[:, :9].mean()
            loss_vel = elem[:, 9:].mean()
            loss = pos_w * loss_pos + vel_w * loss_vel

            # Optional physical consistency terms in physical/original units.
            if args.com_position_weight > 0 or args.com_velocity_weight > 0 or args.energy_weight > 0:
                pred_phys = pred_norm * (model.target_std + 1e-8) + model.target_mean
                prx = pred_phys[:, :9].reshape(-1, 3, 3)
                prv = pred_phys[:, 9:].reshape(-1, 3, 3)
                xcorr = xno + prx
                vcorr = vno + prv
                if args.com_position_weight > 0:
                    cpos_err = _torch_com_position(xcorr, mb) - _torch_com_position(xias, mb)
                    loss = loss + float(args.com_position_weight) * torch.mean(torch.sum(cpos_err * cpos_err, dim=1))
                if args.com_velocity_weight > 0:
                    cvel_err = _torch_com_velocity(vcorr, mb) - _torch_com_velocity(vias, mb)
                    loss = loss + float(args.com_velocity_weight) * torch.mean(torch.sum(cvel_err * cvel_err, dim=1))
                if args.energy_weight > 0:
                    e_corr = _torch_energy(xcorr, vcorr, mb, eps=args.energy_eps)
                    e_ias = _torch_energy(xias, vias, mb, eps=args.energy_eps)
                    rel = (e_corr - e_ias) / (torch.abs(e_ias) + 1e-8)
                    loss = loss + float(args.energy_weight) * torch.mean(torch.clamp(rel * rel, max=args.energy_rel_clip ** 2))

            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            batch_n = int(xb.shape[0])
            loss_sum += float(loss.detach().cpu().item()) * batch_n
            n_seen += batch_n

        train_loss = loss_sum / max(n_seen, 1)
        train_losses.append(train_loss)

        # Validation score in physical units, with explicit velocity and unsafe penalties.
        pred_val = predict_residual(model, X[val_idx], device=device, batch_size=args.eval_batch_size)
        vm = compute_metrics(raw, val_idx, pred_val, unsafe_factor=args.unsafe_factor)
        val_score = (
            vm["corr_pos_median"]
            + args.val_vel_weight * vm["corr_vel_median"]
            + args.val_pos_p95_weight * vm["corr_pos_p95"]
            + args.val_vel_p95_weight * vm["corr_vel_p95"]
            + args.unsafe_pos_penalty * vm["unsafe_pos_frac"]
            + args.unsafe_vel_penalty * vm["unsafe_vel_frac"]
        )
        val_scores.append(float(val_score))

        if val_score < best_score:
            best_score = float(val_score)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:5d}/{args.epochs} | train_loss={train_loss:.6e} "
                f"val_score={val_score:.6e} "
                f"pos_med={vm['corr_pos_median']:.3e}/{vm['base_pos_median']:.3e} "
                f"vel_med={vm['corr_vel_median']:.3e}/{vm['base_vel_median']:.3e} "
                f"unsafe_pos={vm['unsafe_pos_frac']:.1%} unsafe_vel={vm['unsafe_vel_frac']:.1%} "
                f"best_epoch={best_epoch}"
            )

        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"[early-stop] no validation improvement for {args.patience} epochs; stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training failed to produce a best_state")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    pred_train = predict_residual(model, X[train_idx], device=device, batch_size=args.eval_batch_size)
    pred_val = predict_residual(model, X[val_idx], device=device, batch_size=args.eval_batch_size)
    pred_test = predict_residual(model, X[test_idx], device=device, batch_size=args.eval_batch_size)

    metrics_train = compute_metrics(raw, train_idx, pred_train, unsafe_factor=args.unsafe_factor)
    metrics_val = compute_metrics(raw, val_idx, pred_val, unsafe_factor=args.unsafe_factor)
    metrics_test = compute_metrics(raw, test_idx, pred_test, unsafe_factor=args.unsafe_factor)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.output)
    stem = os.path.splitext(args.output)[0]
    np.savez_compressed(stem + "_splits.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    # Save held-out predictions in the same format as v1, plus useful errors.
    ntest = len(test_idx)
    prx = pred_test[:, :9].reshape(ntest, 3, 3).astype(np.float32)
    prv = pred_test[:, 9:].reshape(ntest, 3, 3).astype(np.float32)
    x_no_test = np.asarray(raw["x_nonn_final"], dtype=np.float32)[test_idx]
    v_no_test = np.asarray(raw["v_nonn_final"], dtype=np.float32)[test_idx]
    x_ias_test = np.asarray(raw["x_ias_final"], dtype=np.float32)[test_idx]
    v_ias_test = np.asarray(raw["v_ias_final"], dtype=np.float32)[test_idx]
    x_corr = x_no_test + prx
    v_corr = v_no_test + prv
    base_pos = _rms_state_pos(x_no_test.astype(np.float64), x_ias_test.astype(np.float64))
    corr_pos = _rms_state_pos(x_corr.astype(np.float64), x_ias_test.astype(np.float64))
    base_vel = _rms_state_vel(v_no_test.astype(np.float64), v_ias_test.astype(np.float64))
    corr_vel = _rms_state_vel(v_corr.astype(np.float64), v_ias_test.astype(np.float64))

    np.savez_compressed(
        stem + "_predictions_test.npz",
        test_idx=test_idx,
        pred_residual_x=prx,
        pred_residual_v=prv,
        x_corrected_final=x_corr,
        v_corrected_final=v_corr,
        x_ias_final=x_ias_test,
        v_ias_final=v_ias_test,
        x_nonn_final=x_no_test,
        v_nonn_final=v_no_test,
        base_pos_error=base_pos.astype(np.float32),
        corr_pos_error=corr_pos.astype(np.float32),
        base_vel_error=base_vel.astype(np.float32),
        corr_vel_error=corr_vel.astype(np.float32),
    )

    elapsed = time.perf_counter() - t0
    summary_path = stem + "_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        def w(line: str = ""):
            print(line)
            f.write(line + "\n")

        w("Encounter-level surrogate v2 velocity-safe training summary")
        w("=" * 88)
        w(f"data: {args.data}")
        w(f"output: {args.output}")
        w(f"samples: {n}")
        w(f"train/val/test: {len(train_idx)} / {len(val_idx)} / {len(test_idx)}")
        w(f"model: MLP 18 -> hidden={args.hidden} -> 18")
        w(f"epochs requested: {args.epochs}")
        w(f"best_epoch: {best_epoch}")
        w(f"best_selection_score: {best_score:.8e}")
        w(f"lr: {args.lr}")
        w(f"batch_size: {args.batch_size}")
        w(f"loss: SmoothL1/Huber on normalized residuals, pos_weight={args.pos_weight}, vel_weight={args.vel_weight}")
        w(f"physical penalties: com_position_weight={args.com_position_weight}, com_velocity_weight={args.com_velocity_weight}, energy_weight={args.energy_weight}")
        w(f"validation score: pos_med + {args.val_vel_weight}*vel_med + {args.val_pos_p95_weight}*pos_p95 + {args.val_vel_p95_weight}*vel_p95 + {args.unsafe_pos_penalty}*unsafe_pos + {args.unsafe_vel_penalty}*unsafe_vel")
        w(f"unsafe_factor: {args.unsafe_factor}x baseline error")
        w(f"device: {device}")
        w(f"elapsed_sec: {elapsed:.3f}")
        w("")

        w("Dataset checks")
        w("-" * 88)
        r = np.asarray(raw["r_pair"], dtype=np.float64)
        vr = np.asarray(raw["v_rad_norm"], dtype=np.float64)
        vt = np.asarray(raw["v_tan_norm"], dtype=np.float64)
        relE_ias = np.asarray(raw["relE_ias"], dtype=np.float64)
        relE_no = np.asarray(raw["relE_nonn"], dtype=np.float64)
        cat = np.asarray(raw["category_id"], dtype=np.int64)
        w(f"r_pair: min={r.min():.6e} med={np.median(r):.6e} max={r.max():.6e}")
        w(f"v_rad_norm: min={vr.min():+.4f} med={np.median(vr):+.4f} max={vr.max():+.4f}")
        w(f"v_tan_norm: min={vt.min():.4f} med={np.median(vt):.4f} max={vt.max():.4f}")
        w(f"relE_ias: max={np.max(relE_ias):.3e} med={np.median(relE_ias):.3e}")
        w(f"relE_nonn: max={np.max(relE_no):.3e} med={np.median(relE_no):.3e} p95={np.percentile(relE_no,95):.3e}")
        w("category counts:")
        for c in sorted(set(cat.tolist())):
            w(f"  {CATEGORY_NAMES.get(int(c), 'category_'+str(int(c))):16s}: {int(np.sum(cat == c))}")
        w("")

        def print_metric_block(name: str, m: Dict[str, float]):
            w(f"{name} metrics")
            w("-" * 88)
            for k in [
                "base_pos_mean", "corr_pos_mean", "base_pos_median", "corr_pos_median",
                "base_pos_p95", "corr_pos_p95", "pos_improvement_mean_pct",
                "pos_improvement_median_pct", "pos_success_frac", "unsafe_pos_frac",
                "base_vel_mean", "corr_vel_mean", "base_vel_median", "corr_vel_median",
                "base_vel_p95", "corr_vel_p95", "vel_improvement_mean_pct",
                "vel_improvement_median_pct", "vel_success_frac", "unsafe_vel_frac",
                "unsafe_any_frac", "resid_pos_err_median", "resid_vel_err_median",
                "com_pos_err_median", "com_vel_err_median", "com_pos_err_p95", "com_vel_err_p95",
            ]:
                w(f"  {k:34s}: {m[k]:.8g}")
            w("")

        print_metric_block("TRAIN", metrics_train)
        print_metric_block("VAL", metrics_val)
        print_metric_block("TEST", metrics_test)

        w("TEST category breakdown")
        w("-" * 88)
        for line in category_table(raw, test_idx, pred_test, unsafe_factor=args.unsafe_factor):
            w(line)
        w("")

        w("Verdict")
        w("-" * 88)
        if (metrics_test["pos_improvement_median_pct"] > 0.0 and
            metrics_test["vel_improvement_median_pct"] > 0.0 and
            metrics_test["unsafe_vel_frac"] <= args.max_acceptable_unsafe_vel and
            metrics_test["unsafe_pos_frac"] <= args.max_acceptable_unsafe_pos):
            w("PROCEED: velocity-safe surrogate improves held-out position and velocity medians with acceptable unsafe fractions.")
        elif metrics_test["pos_improvement_median_pct"] > 0.0 and metrics_test["vel_improvement_median_pct"] > 0.0:
            w("PROCEED WITH CAUTION: position and velocity medians improve, but unsafe fractions need gating or further tuning.")
        elif metrics_test["pos_improvement_median_pct"] > 0.0:
            w("PARTIAL: position improves, but velocity remains unresolved.")
        else:
            w("STOP/REASSESS: surrogate does not improve held-out encounter-exit position error.")

    if HAS_MPL:
        # Loss curve.
        plt.figure(figsize=(7, 4))
        plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label="train loss")
        plt.plot(np.arange(1, len(val_scores) + 1), val_scores, label="val score")
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("loss / score")
        plt.title("Velocity-safe encounter surrogate training curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(stem + "_loss.png")
        plt.close()

        def _scatter(base, corr, path, title, xlabel, ylabel):
            lo = max(1e-12, min(float(base.min()), float(corr.min())) * 0.8)
            hi = max(float(base.max()), float(corr.max())) * 1.2
            plt.figure(figsize=(5.5, 5))
            plt.scatter(base, corr, s=18, alpha=0.7)
            plt.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="no change")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(path)
            plt.close()

        _scatter(
            base_pos, corr_pos,
            stem + "_scatter_pos.png",
            "Held-out position correction",
            "noNN position error vs IAS15",
            "corrected position error vs IAS15",
        )
        _scatter(
            base_vel, corr_vel,
            stem + "_scatter_vel.png",
            "Held-out velocity correction",
            "noNN velocity error vs IAS15",
            "corrected velocity error vs IAS15",
        )

    print(f"\nSaved model: {args.output}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved splits: {stem}_splits.npz")
    print(f"Saved test predictions: {stem}_predictions_test.npz")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Train velocity-safe encounter-level residual surrogate v2.")
    ap.add_argument("--data", default="encounter_surrogate_v1_large1.npz",
                    help="Merged encounter surrogate dataset .npz")
    ap.add_argument("--output", default="encounter_surrogate_v2_velocitysafe_large1.pt",
                    help="Output model .pt path")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--huber-beta", type=float, default=0.5,
                    help="SmoothL1 beta in normalized target units")
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--vel-weight", type=float, default=1.0,
                    help="v2 default is 1.0; v1 used 0.1")
    ap.add_argument("--com-position-weight", type=float, default=0.05)
    ap.add_argument("--com-velocity-weight", type=float, default=0.05)
    ap.add_argument("--energy-weight", type=float, default=0.0,
                    help="Optional softened-energy consistency penalty; default off for stability")
    ap.add_argument("--energy-eps", type=float, default=3e-4)
    ap.add_argument("--energy-rel-clip", type=float, default=10.0)
    ap.add_argument("--val-vel-weight", type=float, default=0.5,
                    help="Validation score includes this * corrected velocity median")
    ap.add_argument("--val-pos-p95-weight", type=float, default=0.05)
    ap.add_argument("--val-vel-p95-weight", type=float, default=0.05)
    ap.add_argument("--unsafe-pos-penalty", type=float, default=0.05)
    ap.add_argument("--unsafe-vel-penalty", type=float, default=0.10)
    ap.add_argument("--unsafe-factor", type=float, default=2.0,
                    help="Unsafe if corrected error > factor * noNN baseline error")
    ap.add_argument("--max-acceptable-unsafe-pos", type=float, default=0.10)
    ap.add_argument("--max-acceptable-unsafe-vel", type=float, default=0.15)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=700,
                    help="Early stopping patience; use 0 to disable")
    ap.add_argument("--print-every", type=int, default=100)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.unsafe_factor <= 1.0:
        raise ValueError("--unsafe-factor should be > 1.0")
    if args.vel_weight <= 0:
        raise ValueError("--vel-weight must be positive")
    return args


if __name__ == "__main__":
    train_one_model(parse_args())
