"""
diagnostic_v3.py

Comprehensive diagnostics for the SIMON Zone 3 v3 6-input correction model.

Checks performed:
  1. Dataset structural checks and feature/target ranges.
  2. Model architecture compatibility: input_mean/std must be length 6.
  3. Held-out test metrics using the saved split from training if available.
  4. Per-dt target-vs-predicted medians.
  5. Approaching vs receding separation: verifies v_rad_norm is being used.
  6. Identity-target behaviour: verifies c≈1 cases are not over-corrected.
  7. Velocity probe: fixed r,m,dt with vr<0 vs vr>0.
  8. Gate-safety check: fraction of predicted c outside [0.2,5.0].
  9. Nearest-neighbour check for fixed probes to identify off-manifold failures.
 10. Optional plots if matplotlib is available.

Example:
  python -B diagnostic_v3.py --data encounter_data_zone3_v3.npz --model pair_correction_nn_v3.pt
"""

import argparse
import os
from typing import Dict, Tuple, List

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
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("input_mean", torch.zeros(6))
        self.register_buffer("input_std", torch.ones(6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.input_mean) / (self.input_std + 1e-8)).squeeze(-1)


def infer_hidden_from_state_dict(sd: Dict[str, torch.Tensor]) -> int:
    if "net.0.weight" not in sd:
        raise KeyError("Model state_dict does not contain net.0.weight")
    w = sd["net.0.weight"]
    if w.ndim != 2 or w.shape[1] != 6:
        raise ValueError(f"Expected first layer shape (hidden, 6), got {tuple(w.shape)}")
    return int(w.shape[0])


def load_model(model_path: str) -> PairCorrectionNN:
    sd = torch.load(model_path, map_location="cpu")
    hidden = infer_hidden_from_state_dict(sd)
    model = PairCorrectionNN(hidden=hidden)
    model.load_state_dict(sd)
    model.eval()
    if model.input_mean.numel() != 6 or model.input_std.numel() != 6:
        raise ValueError("Model normalisation buffers are not length 6; this is not a v3 6-input model.")
    print(f"[model] Loaded {model_path}")
    print(f"[model] hidden={hidden}, parameters={sum(p.numel() for p in model.parameters())}")
    print(f"[model] input_mean={model.input_mean.numpy()}")
    print(f"[model] input_std ={model.input_std.numpy()}")
    return model


def load_data(path: str) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    data = np.load(path)
    missing = [k for k in REQUIRED_FIELDS if k not in data.files]
    if missing:
        raise KeyError(f"Missing required v3 fields: {missing}")
    raw = {k: data[k].astype(np.float32) for k in REQUIRED_FIELDS}
    n = len(raw["r_AU"])
    for k, arr in raw.items():
        if len(arr) != n:
            raise ValueError(f"Field {k} length mismatch")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Field {k} contains non-finite values")

    log_r_soft = np.log(raw["r_soft"].astype(np.float64) + 1e-30).astype(np.float32)
    X = np.stack([
        log_r_soft,
        raw["log_mi"],
        raw["log_mj"],
        raw["log_dt"],
        raw["v_rad_norm"],
        raw["v_tan_norm"],
    ], axis=1).astype(np.float32)
    y = raw["log_c_opt"].astype(np.float32)
    print(f"[data] Loaded {path}: n={n}")
    return raw, X, y


def load_or_make_test_indices(n: int, split_path: str, seed: int) -> np.ndarray:
    if split_path and os.path.exists(split_path):
        s = np.load(split_path)
        if "test_idx" not in s.files:
            raise KeyError(f"{split_path} does not contain test_idx")
        idx = s["test_idx"].astype(np.int64)
        print(f"[split] Loaded test_idx from {split_path}: n={len(idx)}")
        return idx
    rng = np.random.RandomState(seed)
    idx_all = rng.permutation(n)
    n_train = int(round(0.70 * n))
    n_val = int(round(0.15 * n))
    idx = idx_all[n_train + n_val:]
    print(f"[split] No split file found; recreated deterministic test split with seed={seed}: n={len(idx)}")
    return idx.astype(np.int64)


def predict(model: PairCorrectionNN, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32)
            outs.append(model(xb).cpu().numpy().astype(np.float64))
    return np.concatenate(outs, axis=0)


def metrics(y_log: np.ndarray, pred_log: np.ndarray) -> Dict[str, float]:
    y_log = y_log.astype(np.float64)
    pred_log = pred_log.astype(np.float64)
    err_log = pred_log - y_log
    c_true = np.exp(y_log)
    c_pred = np.exp(pred_log)
    err_c = c_pred - c_true
    return {
        "mse_log": float(np.mean(err_log ** 2)),
        "rmse_log": float(np.sqrt(np.mean(err_log ** 2))),
        "mae_log": float(np.mean(np.abs(err_log))),
        "mae_c": float(np.mean(np.abs(err_c))),
        "median_abs_c": float(np.median(np.abs(err_c))),
        "p90_abs_c": float(np.percentile(np.abs(err_c), 90)),
        "p95_abs_c": float(np.percentile(np.abs(err_c), 95)),
        "corr_c": float(np.corrcoef(c_true, c_pred)[0, 1]) if len(c_true) > 2 else float("nan"),
        "c_pred_min": float(np.min(c_pred)),
        "c_pred_med": float(np.median(c_pred)),
        "c_pred_max": float(np.max(c_pred)),
    }


def print_dataset_checks(raw: Dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> None:
    print("\nDATASET CHECKS")
    print("=" * 72)
    print(f"required fields present: {all(k in raw for k in REQUIRED_FIELDS)}")
    print(f"finite all: {all(np.all(np.isfinite(raw[k])) for k in REQUIRED_FIELDS)}")
    print(f"r in Zone 3: {bool(np.all((raw['r_AU'] > 0.052) & (raw['r_AU'] < 0.148)))}")
    print(f"v_tan_norm >= 0: {bool(np.all(raw['v_tan_norm'] >= 0))}")
    print(f"log_c matches log(c): {bool(np.allclose(raw['log_c_opt'], np.log(raw['c_opt'] + 1e-30), rtol=2e-5, atol=2e-5))}")
    print("\nFeature ranges:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:12s} min={X[:,i].min(): .4f} med={np.median(X[:,i]): .4f} max={X[:,i].max(): .4f}")
    print(f"Target log(c): min={y.min():.4f} med={np.median(y):.4f} max={y.max():.4f}")
    print(f"Target c     : min={raw['c_opt'].min():.4f} med={np.median(raw['c_opt']):.4f} max={raw['c_opt'].max():.4f}")
    print(f"Identity targets c=1: {int(np.sum(np.abs(raw['c_opt'] - 1.0) < 1e-7))} ({np.mean(np.abs(raw['c_opt'] - 1.0) < 1e-7):.2%})")


def grouped_tables(raw: Dict[str, np.ndarray], idx: np.ndarray, y_log: np.ndarray, pred_log: np.ndarray) -> None:
    c_true = np.exp(y_log)
    c_pred = np.exp(pred_log)
    dt = np.exp(raw["log_dt"][idx].astype(np.float64))
    rounded = np.array([round(float(x), 4) for x in dt])
    vr = raw["v_rad_norm"][idx].astype(np.float64)
    vt = raw["v_tan_norm"][idx].astype(np.float64)
    impr = raw["improvement"][idx].astype(np.float64)

    print("\nHELD-OUT TEST METRICS")
    print("=" * 72)
    m = metrics(y_log, pred_log)
    for k, v in m.items():
        print(f"  {k:14s}: {v:.6g}")
    print(f"  out-of-gate c_pred (<0.2 or >5): {np.mean((c_pred < 0.2) | (c_pred > 5.0)):.2%}")

    print("\nPer-dt held-out table")
    print(f"  {'dt':>7} | {'n':>5} | {'true_med':>9} | {'pred_med':>9} | {'MAE_c':>9} | {'true_q25':>9} | {'true_q75':>9}")
    print("  " + "-" * 78)
    dt_true_meds = []
    dt_pred_meds = []
    for d in sorted(set(rounded)):
        mask = rounded == d
        dt_true_meds.append(float(np.median(c_true[mask])))
        dt_pred_meds.append(float(np.median(c_pred[mask])))
        print(f"  {d:7.4f} | {int(mask.sum()):5d} | {np.median(c_true[mask]):9.4f} | "
              f"{np.median(c_pred[mask]):9.4f} | {np.mean(np.abs(c_pred[mask]-c_true[mask])):9.4f} | "
              f"{np.percentile(c_true[mask],25):9.4f} | {np.percentile(c_true[mask],75):9.4f}")

    print("\nApproach vs recede held-out table")
    print(f"  {'dt':>7} | {'app_n':>5} | {'app_true':>9} | {'app_pred':>9} | {'rec_n':>5} | {'rec_true':>9} | {'rec_pred':>9}")
    print("  " + "-" * 85)
    signs_ok_count = 0
    signs_total = 0
    for d in sorted(set(rounded)):
        base = rounded == d
        app = base & (vr < 0)
        rec = base & (vr >= 0)
        def med(a, mask):
            return float(np.median(a[mask])) if np.any(mask) else float("nan")
        app_t, app_p = med(c_true, app), med(c_pred, app)
        rec_t, rec_p = med(c_true, rec), med(c_pred, rec)
        if np.isfinite(app_p) and np.isfinite(rec_p):
            signs_total += 1
            if app_p > rec_p:
                signs_ok_count += 1
        print(f"  {d:7.4f} | {int(app.sum()):5d} | {app_t:9.4f} | {app_p:9.4f} | "
              f"{int(rec.sum()):5d} | {rec_t:9.4f} | {rec_p:9.4f}")
    print(f"\nVelocity-sign ordering check: app_pred > rec_pred for {signs_ok_count}/{signs_total} dt groups")

    identity = np.abs(c_true - 1.0) < 1e-7
    print("\nIdentity-target behaviour")
    if np.any(identity):
        print(f"  n_identity={int(identity.sum())}")
        print(f"  pred_median={np.median(c_pred[identity]):.5f}")
        print(f"  pred_q25={np.percentile(c_pred[identity],25):.5f} pred_q75={np.percentile(c_pred[identity],75):.5f}")
        print(f"  within [0.90,1.10]={np.mean((c_pred[identity] >= 0.90) & (c_pred[identity] <= 1.10)):.2%}")
    else:
        print("  no identity targets in held-out split")

    print("\nVelocity-feature coverage in held-out split")
    print(f"  v_rad_norm: min={vr.min():.3f} med={np.median(vr):.3f} max={vr.max():.3f}")
    print(f"  v_tan_norm: min={vt.min():.3f} med={np.median(vt):.3f} max={vt.max():.3f}")
    print(f"  approaching fraction: {np.mean(vr < 0):.2%}")
    print(f"  improvement median: {np.median(impr):.2%}")


def velocity_probe(model: PairCorrectionNN) -> Dict[str, float]:
    """
    Fixed-state sanity probes.

    Returns a small report dictionary so final_verdict can warn if the model
    behaves badly on controlled probe states.
    """
    print("\nFIXED-STATE PROBES")
    print("=" * 72)

    hard_min, hard_max = 0.2, 5.0
    conservative_min, conservative_max = 0.4, 2.5

    all_probe_c = []
    approach_c = []
    recede_c = []

    print("Probe A: Same r/m/vtan, flip radial velocity sign")
    print("  r=0.10 AU, m_i=1.0, m_j=0.01, v_tan_norm=0.65")
    print(f"  {'dt':>7} | {'approach vr=-0.8':>17} | {'side vr=0':>10} | {'recede vr=+0.8':>16}")
    print("  " + "-" * 63)

    r_soft = float(np.sqrt(0.10 ** 2 + EPS ** 2))

    for d in DT_LIST:
        rows = np.array([
            [np.log(r_soft), np.log(1.0), np.log(0.01), np.log(d), -0.8, 0.65],
            [np.log(r_soft), np.log(1.0), np.log(0.01), np.log(d),  0.0, 0.65],
            [np.log(r_soft), np.log(1.0), np.log(0.01), np.log(d), +0.8, 0.65],
        ], dtype=np.float32)

        cp = np.exp(predict(model, rows))
        all_probe_c.extend([float(x) for x in cp])
        approach_c.append(float(cp[0]))
        recede_c.append(float(cp[2]))

        flag = ""
        if np.any((cp < hard_min) | (cp > hard_max) | ~np.isfinite(cp)):
            flag = "  <-- HARD-GATE WARNING"
        elif np.any((cp < conservative_min) | (cp > conservative_max)):
            flag = "  <-- conservative warning"

        print(f"  {d:7.4f} | {cp[0]:17.5f} | {cp[1]:10.5f} | {cp[2]:16.5f}{flag}")

    print("\nProbe B: Same r/m/vr, increase tangential velocity")
    print("  r=0.10 AU, m_i=1.0, m_j=0.01, dt=0.04, v_rad_norm=+0.8")
    print(f"  {'v_tan_norm':>10} | {'c_pred':>9}")
    print("  " + "-" * 24)

    for vt in [0.1, 0.3, 0.6, 0.9, 1.2, 1.6]:
        row = np.array([[np.log(r_soft), np.log(1.0), np.log(0.01), np.log(0.04), +0.8, vt]], dtype=np.float32)
        cp = float(np.exp(predict(model, row))[0])
        all_probe_c.append(cp)

        flag = ""
        if (not np.isfinite(cp)) or cp < hard_min or cp > hard_max:
            flag = "  <-- HARD-GATE WARNING"
        elif cp < conservative_min or cp > conservative_max:
            flag = "  <-- conservative warning"

        print(f"  {vt:10.3f} | {cp:9.5f}{flag}")

    all_probe_c = np.array(all_probe_c, dtype=np.float64)
    approach_c = np.array(approach_c, dtype=np.float64)
    recede_c = np.array(recede_c, dtype=np.float64)

    report = {
        "probe_c_min": float(np.nanmin(all_probe_c)),
        "probe_c_max": float(np.nanmax(all_probe_c)),
        "probe_hard_gate_frac": float(np.mean((all_probe_c < hard_min) | (all_probe_c > hard_max) | ~np.isfinite(all_probe_c))),
        "probe_conservative_gate_frac": float(np.mean((all_probe_c < conservative_min) | (all_probe_c > conservative_max) | ~np.isfinite(all_probe_c))),
        "probe_approach_gt_recede_frac": float(np.mean(approach_c > recede_c)),
    }

    print("\nProbe summary")
    print(f"  c range                         : {report['probe_c_min']:.5g} to {report['probe_c_max']:.5g}")
    print(f"  hard gate failure fraction       : {report['probe_hard_gate_frac']:.2%}")
    print(f"  conservative warning fraction    : {report['probe_conservative_gate_frac']:.2%}")
    print(f"  approach > recede fraction       : {report['probe_approach_gt_recede_frac']:.2%}")

    return report



def build_probe_inputs() -> Tuple[List[str], np.ndarray]:
    """
    Build the same fixed-state probe inputs used by velocity_probe().

    Keeping this in one helper lets the nearest-neighbour diagnostic inspect
    exactly the same controlled states that triggered the hard-gate warnings.
    """
    names: List[str] = []
    rows: List[List[float]] = []
    r_soft = float(np.sqrt(0.10 ** 2 + EPS ** 2))

    # Probe A: same r/m/vtan, flip radial velocity sign for every dt.
    for d in DT_LIST:
        for label, vr in [("approach", -0.8), ("side", 0.0), ("recede", +0.8)]:
            names.append(f"A_{label}_dt{d:.4f}")
            rows.append([np.log(r_soft), np.log(1.0), np.log(0.01), np.log(d), vr, 0.65])

    # Probe B: same r/m/vr/dt, increase tangential velocity.
    for vt in [0.1, 0.3, 0.6, 0.9, 1.2, 1.6]:
        names.append(f"B_recede_dt0.0400_vt{vt:.1f}")
        rows.append([np.log(r_soft), np.log(1.0), np.log(0.01), np.log(0.04), +0.8, vt])

    return names, np.array(rows, dtype=np.float32)


def _nearest_reference_radius(X_norm: np.ndarray, max_sample: int = 750) -> Dict[str, float]:
    """
    Estimate the typical nearest-neighbour spacing of the training dataset in
    model-normalised feature space. This gives a scale for deciding whether a
    fixed probe is inside or outside the data manifold.
    """
    n = len(X_norm)
    if n <= 1:
        return {"p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan")}

    if n <= max_sample:
        sample_idx = np.arange(n)
    else:
        # Deterministic spread across the file; no randomness in diagnostics.
        sample_idx = np.linspace(0, n - 1, max_sample).round().astype(np.int64)
        sample_idx = np.unique(sample_idx)

    mins = []
    for idx in sample_idx:
        diff = X_norm - X_norm[idx]
        d2 = np.einsum("ij,ij->i", diff, diff)
        d2[idx] = np.inf
        mins.append(float(np.sqrt(np.min(d2))))

    mins = np.array(mins, dtype=np.float64)
    return {
        "p50": float(np.percentile(mins, 50)),
        "p90": float(np.percentile(mins, 90)),
        "p95": float(np.percentile(mins, 95)),
        "p99": float(np.percentile(mins, 99)),
    }


def nearest_neighbor_probe_diagnostic(
    raw: Dict[str, np.ndarray],
    X: np.ndarray,
    model: PairCorrectionNN,
    k: int = 8,
) -> Dict[str, float]:
    """
    For each fixed probe, find nearby real generated samples in the SAME dt group.

    This answers the central question raised by the probe failure:
      - Is the failing probe near real training/test data, meaning the model is
        wrong in an important region?
      - Or is the failing probe far from the accepted data manifold, meaning it is
        mainly an extrapolation warning that should be guarded by gates?
    """
    print("\nNEAREST-NEIGHBOUR CHECK FOR FIXED PROBES")
    print("=" * 72)

    names, X_probe = build_probe_inputs()
    c_probe = np.exp(predict(model, X_probe))

    mean = model.input_mean.detach().cpu().numpy().astype(np.float64)
    std = model.input_std.detach().cpu().numpy().astype(np.float64) + 1e-8
    X_norm = (X.astype(np.float64) - mean) / std
    Xp_norm = (X_probe.astype(np.float64) - mean) / std

    ref = _nearest_reference_radius(X_norm)
    # p95 is a useful practical cutoff: beyond this, the probe is farther from
    # real samples than 95% of real samples are from their own nearest neighbour.
    off_cut = ref["p95"]
    if not np.isfinite(off_cut) or off_cut <= 0:
        off_cut = 1.0

    print("Reference nearest-neighbour spacing in normalised 6D feature space:")
    print(f"  p50={ref['p50']:.3f}  p90={ref['p90']:.3f}  p95={ref['p95']:.3f}  p99={ref['p99']:.3f}")
    print(f"  off-manifold warning cutoff: nearest distance > {off_cut:.3f}")

    c_true_all = raw["c_opt"].astype(np.float64)
    dt_all = np.exp(raw["log_dt"].astype(np.float64))

    hard_min, hard_max = 0.2, 5.0
    conservative_min, conservative_max = 0.4, 2.5

    rows = []
    hard_fail_count = 0
    hard_fail_off = 0
    hard_fail_near = 0
    off_count = 0

    print("\nProbe nearest-neighbour table, restricted to same dt")
    print(f"  {'probe':>24} | {'c_pred':>9} | {'nn_dist':>8} | {'off?':>5} | {'nn_c_med':>9} | {'nn_c_min':>9} | {'nn_c_max':>9} | {'flags':>18}")
    print("  " + "-" * 111)

    for name, xp, xpn, cp in zip(names, X_probe, Xp_norm, c_probe):
        probe_dt = float(np.exp(float(xp[3])))
        same_dt = np.abs(dt_all - probe_dt) < 1e-5
        if not np.any(same_dt):
            same_dt = np.ones(len(X_norm), dtype=bool)

        global_ids = np.nonzero(same_dt)[0]
        diff = X_norm[global_ids] - xpn
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        order = np.argsort(dist)[:max(1, int(k))]
        nn_ids = global_ids[order]
        nn_dist = float(dist[order[0]])
        nn_c = c_true_all[nn_ids]

        hard_fail = (not np.isfinite(cp)) or cp < hard_min or cp > hard_max
        conservative_warn = (not np.isfinite(cp)) or cp < conservative_min or cp > conservative_max
        off = bool(nn_dist > off_cut)

        if off:
            off_count += 1
        if hard_fail:
            hard_fail_count += 1
            if off:
                hard_fail_off += 1
            else:
                hard_fail_near += 1

        flags = []
        if hard_fail:
            flags.append("HARD_GATE")
        elif conservative_warn:
            flags.append("CONSERVE")
        if off:
            flags.append("OFF_MANIFOLD")
        flag_text = ",".join(flags) if flags else "-"

        print(f"  {name:>24} | {float(cp):9.5f} | {nn_dist:8.3f} | {str(off):>5} | "
              f"{np.median(nn_c):9.4f} | {np.min(nn_c):9.4f} | {np.max(nn_c):9.4f} | {flag_text:>18}")

        rows.append((name, float(cp), nn_dist, off, float(np.median(nn_c)), hard_fail, conservative_warn))

    total = max(1, len(rows))
    report = {
        "nn_offmanifold_frac": float(off_count / total),
        "nn_hard_fail_count": float(hard_fail_count),
        "nn_hard_fail_offmanifold_count": float(hard_fail_off),
        "nn_hard_fail_near_count": float(hard_fail_near),
        "nn_reference_p95": float(off_cut),
    }

    print("\nNearest-neighbour probe summary")
    print(f"  off-manifold probe fraction              : {report['nn_offmanifold_frac']:.2%}")
    print(f"  hard-gate failing probes                 : {int(report['nn_hard_fail_count'])}")
    print(f"  hard-gate failures that are off-manifold : {int(report['nn_hard_fail_offmanifold_count'])}")
    print(f"  hard-gate failures near real data        : {int(report['nn_hard_fail_near_count'])}")

    if hard_fail_near > 0:
        print("  Interpretation: at least one unsafe probe is near accepted data; retraining or a bounded-output model may be needed.")
    elif hard_fail_off > 0:
        print("  Interpretation: unsafe fixed probes appear off-manifold; strict rollout gating is still required.")
    else:
        print("  Interpretation: fixed probes are locally supported and do not trigger hard-gate failures.")

    return report

def maybe_make_plots(raw: Dict[str, np.ndarray], idx: np.ndarray, y_log: np.ndarray, pred_log: np.ndarray, out_dir: str) -> None:
    if not HAS_MPL:
        print("\n[plots] matplotlib not available; skipping plots.")
        return
    os.makedirs(out_dir, exist_ok=True)
    c_true = np.exp(y_log)
    c_pred = np.exp(pred_log)
    dt = np.exp(raw["log_dt"][idx].astype(np.float64))
    rounded = np.array([round(float(x), 4) for x in dt])
    vr = raw["v_rad_norm"][idx].astype(np.float64)
    vt = raw["v_tan_norm"][idx].astype(np.float64)

    # 1. predicted vs true
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(c_true, c_pred, s=5, alpha=0.30)
    lo = min(float(c_true.min()), float(c_pred.min()))
    hi = max(float(c_true.max()), float(c_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("True c_opt")
    ax.set_ylabel("Predicted c")
    ax.set_title("Zone 3 v3 held-out predicted vs true c")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "diag_v3_pred_vs_true.png"), dpi=180)
    plt.close(fig)

    # 2. per dt medians
    dts = sorted(set(rounded))
    true_med = [np.median(c_true[rounded == d]) for d in dts]
    pred_med = [np.median(c_pred[rounded == d]) for d in dts]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dts, true_med, "o-", label="true median")
    ax.plot(dts, pred_med, "s--", label="pred median")
    ax.axhline(1.0, color="k", lw=1, ls=":")
    ax.set_xlabel("dt (yr)")
    ax.set_ylabel("median c")
    ax.set_title("Held-out median c by dt")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "diag_v3_median_by_dt.png"), dpi=180)
    plt.close(fig)

    # 3. residual vs radial velocity
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(vr, c_pred - c_true, c=dt, s=5, alpha=0.35, cmap="viridis")
    ax.axhline(0.0, color="k", lw=1, ls="--")
    ax.axvline(0.0, color="k", lw=1, ls=":")
    ax.set_xlabel("v_rad_norm")
    ax.set_ylabel("predicted c - true c")
    ax.set_title("Held-out residual vs radial velocity")
    fig.colorbar(sc, ax=ax, label="dt")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "diag_v3_residual_vs_vrad.png"), dpi=180)
    plt.close(fig)

    # 4. velocity space predicted c
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(vr, vt, c=c_pred, s=5, alpha=0.35, cmap="viridis")
    ax.axvline(0.0, color="k", lw=1, ls="--")
    ax.set_xlabel("v_rad_norm")
    ax.set_ylabel("v_tan_norm")
    ax.set_title("Held-out velocity feature space coloured by predicted c")
    fig.colorbar(sc, ax=ax, label="predicted c")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "diag_v3_velocity_space_pred.png"), dpi=180)
    plt.close(fig)

    print(f"\n[plots] Saved diagnostics to {out_dir}")


def final_verdict(
    raw: Dict[str, np.ndarray],
    idx: np.ndarray,
    y_log: np.ndarray,
    pred_log: np.ndarray,
    probe_report: Dict[str, float],
    nn_report: Dict[str, float],
) -> str:
    c_true = np.exp(y_log)
    c_pred = np.exp(pred_log)
    dt = np.exp(raw["log_dt"][idx].astype(np.float64))
    rounded = np.array([round(float(x), 4) for x in dt])
    vr = raw["v_rad_norm"][idx].astype(np.float64)

    m = metrics(y_log, pred_log)

    # Held-out fit checks.
    mae_ok = m["mae_c"] < 0.05
    corr_ok = np.isfinite(m["corr_c"]) and m["corr_c"] > 0.90

    # Deployment safety checks.
    hard_gate_frac = np.mean((c_pred < 0.2) | (c_pred > 5.0) | ~np.isfinite(c_pred))
    conservative_gate_frac = np.mean((c_pred < 0.4) | (c_pred > 2.5) | ~np.isfinite(c_pred))

    hard_gate_ok = hard_gate_frac < 0.005
    conservative_gate_ok = conservative_gate_frac < 0.03

    # Check that the velocity sign effect is learned.
    sign_ok = 0
    sign_total = 0
    for d in sorted(set(rounded)):
        base = rounded == d
        app = base & (vr < 0)
        rec = base & (vr >= 0)
        if np.any(app) and np.any(rec):
            sign_total += 1
            if np.median(c_pred[app]) > np.median(c_pred[rec]):
                sign_ok += 1
    sign_order_ok = sign_total > 0 and sign_ok == sign_total

    # Identity target behaviour.
    identity = np.abs(c_true - 1.0) < 1e-7
    if np.any(identity):
        identity_within = float(np.mean((c_pred[identity] >= 0.90) & (c_pred[identity] <= 1.10)))
    else:
        identity_within = 1.0
    identity_ok = identity_within >= 0.80

    # Probe safety.
    probe_hard_ok = probe_report.get("probe_hard_gate_frac", 1.0) == 0.0
    probe_conservative_ok = probe_report.get("probe_conservative_gate_frac", 1.0) <= 0.10
    probe_sign_ok = probe_report.get("probe_approach_gt_recede_frac", 0.0) >= 0.75

    problems = []
    warnings = []

    if not mae_ok:
        problems.append(f"held-out MAE_c too high ({m['mae_c']:.4f})")
    if not corr_ok:
        problems.append(f"held-out corr_c too low ({m['corr_c']:.4f})")
    if not hard_gate_ok:
        problems.append(f"held-out hard gate failures {hard_gate_frac:.2%}")
    if not sign_order_ok:
        problems.append(f"approach/recede ordering only {sign_ok}/{sign_total} dt groups")

    if not conservative_gate_ok:
        warnings.append(f"held-out conservative gate warnings {conservative_gate_frac:.2%}")
    if not identity_ok:
        warnings.append(f"identity targets only {identity_within:.2%} within [0.90,1.10]")
    if not probe_hard_ok:
        problems.append(f"fixed-state probe hard gate failures {probe_report.get('probe_hard_gate_frac', 1.0):.2%}")
    if not probe_conservative_ok:
        warnings.append(f"fixed-state probe conservative warnings {probe_report.get('probe_conservative_gate_frac', 1.0):.2%}")
    if not probe_sign_ok:
        problems.append("fixed-state probe does not preserve approach > recede pattern")

    nn_hard_near = int(nn_report.get("nn_hard_fail_near_count", 0))
    nn_hard_off = int(nn_report.get("nn_hard_fail_offmanifold_count", 0))
    if nn_hard_near > 0:
        problems.append(f"{nn_hard_near} fixed-probe hard failures are near real data")
    elif nn_hard_off > 0:
        warnings.append(f"{nn_hard_off} fixed-probe hard failures appear off-manifold; keep strict gate")

    if problems:
        return "CHECK BEFORE DEPLOYMENT: " + "; ".join(problems + warnings)

    if warnings:
        return "PROCEED TO ROLLOUT TESTS WITH STRICT GATING: " + "; ".join(warnings)

    return "PROCEED TO ROLLOUT TESTS: held-out and probe diagnostics support the v3 model."


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose SIMON Zone 3 v3 6-input correction model.")
    ap.add_argument("--data", default="encounter_data_zone3_v3.npz")
    ap.add_argument("--model", default="pair_correction_nn_v3.pt")
    ap.add_argument("--split", default="pair_correction_nn_v3_split.npz")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="diagnostic_v3_out")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--nn-k", type=int, default=8,
                    help="Number of same-dt nearest neighbours to report for each fixed probe")
    args = ap.parse_args()

    raw, X, y = load_data(args.data)
    print_dataset_checks(raw, X, y)
    model = load_model(args.model)

    idx = load_or_make_test_indices(len(X), args.split, args.seed)
    y_test = y[idx].astype(np.float64)
    pred_test = predict(model, X[idx])

    grouped_tables(raw, idx, y_test, pred_test)
    probe_report = velocity_probe(model)
    nn_report = nearest_neighbor_probe_diagnostic(raw, X, model, k=args.nn_k)
    if not args.no_plots:
        maybe_make_plots(raw, idx, y_test, pred_test, args.out_dir)

    verdict = final_verdict(raw, idx, y_test, pred_test, probe_report, nn_report)
    
    print("\nFINAL VERDICT")
    print("=" * 72)
    print(verdict)


if __name__ == "__main__":
    main()
