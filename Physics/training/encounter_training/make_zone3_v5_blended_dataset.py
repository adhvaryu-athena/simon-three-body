"""
make_zone3_v5_blended_dataset.py

Create a blended Zone 3 training dataset for the v5/window-target experiment.

Inputs
------
1. Old broad one-step Zone 3 dataset, e.g.
       encounter_data_zone3_v3_augmented.npz
   where c_opt/log_c_opt mean one-step optimum.

2. New short-window Zone 3 dataset, e.g.
       encounter_data_zone3_window_v2_full.npz
   where c_opt/log_c_opt mean short-window optimum.

Output
------
A single .npz containing all original required v3 fields plus metadata fields:
    source_id      : 0 = old one-step, 1 = new window-target
    target_type_id : 0 = one-step target, 1 = window target
    sample_weight  : recommended training weight per sample

The required fields are kept unchanged so the existing inspector can still load
and inspect the output. The new weighted trainer should use sample_weight.

Example
-------
python -B make_zone3_v5_blended_dataset.py \
  --old encounter_data_zone3_v3_augmented.npz \
  --window encounter_data_zone3_window_v2_full.npz \
  --out encounter_data_zone3_v5_blended_weighted.npz \
  --old-weight 1.0 \
  --window-weight 6.0 \
  --overwrite
"""

import argparse
import os
import numpy as np

REQUIRED_FIELDS = [
    "r_AU", "r_soft", "log_mi", "log_mj", "log_dt",
    "v_rad_norm", "v_tan_norm",
    "c_opt", "log_c_opt", "c_ana", "log_c_ana", "improvement",
]

EPS = 3e-4
Z3_R_MIN = 0.052
Z3_R_MAX = 0.148


def parse_args():
    ap = argparse.ArgumentParser(description="Blend old one-step and new window-target Zone 3 datasets.")
    ap.add_argument("--old", required=True, help="Old one-step Zone 3 dataset .npz")
    ap.add_argument("--window", required=True, help="New short-window Zone 3 dataset .npz")
    ap.add_argument("--out", default="encounter_data_zone3_v5_blended_weighted.npz", help="Output blended .npz")
    ap.add_argument("--old-weight", type=float, default=1.0, help="Training weight for old one-step samples")
    ap.add_argument("--window-weight", type=float, default=6.0, help="Training weight for new window-target samples")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    return ap.parse_args()


def _load_checked(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    missing = [k for k in REQUIRED_FIELDS if k not in data.files]
    if missing:
        raise KeyError(f"{label} dataset {path} missing fields: {missing}")

    raw = {k: np.asarray(data[k], dtype=np.float32) for k in REQUIRED_FIELDS}
    n = len(raw["r_AU"])
    if n == 0:
        raise ValueError(f"{label} dataset has zero rows: {path}")
    for k, arr in raw.items():
        if arr.ndim != 1:
            raise ValueError(f"{label}:{k} must be 1D, got {arr.shape}")
        if len(arr) != n:
            raise ValueError(f"{label}:{k} length {len(arr)} != {n}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{label}:{k} contains non-finite values")

    r = raw["r_AU"].astype(np.float64)
    rsoft = raw["r_soft"].astype(np.float64)
    c = raw["c_opt"].astype(np.float64)
    logc = raw["log_c_opt"].astype(np.float64)
    cana = raw["c_ana"].astype(np.float64)
    logcana = raw["log_c_ana"].astype(np.float64)
    vt = raw["v_tan_norm"].astype(np.float64)

    if not np.all((r > Z3_R_MIN) & (r < Z3_R_MAX)):
        raise ValueError(f"{label}: r_AU outside Zone 3 safe interior [{Z3_R_MIN}, {Z3_R_MAX}]")
    if not np.all(vt >= 0):
        raise ValueError(f"{label}: v_tan_norm contains negative values")
    if not np.all(c > 0):
        raise ValueError(f"{label}: c_opt must be positive")
    if not np.allclose(logc, np.log(c + 1e-30), rtol=2e-5, atol=2e-5):
        raise ValueError(f"{label}: log_c_opt does not match log(c_opt)")
    if not np.allclose(rsoft, np.sqrt(r*r + EPS*EPS), rtol=2e-5, atol=2e-7):
        raise ValueError(f"{label}: r_soft check failed")
    expected_cana = (rsoft / r) ** 3
    if not np.allclose(cana, expected_cana, rtol=2e-5, atol=2e-7):
        raise ValueError(f"{label}: c_ana check failed")
    if not np.allclose(logcana, np.log(cana + 1e-30), rtol=2e-5, atol=2e-5):
        raise ValueError(f"{label}: log_c_ana check failed")

    return raw


def _dt_summary(log_dt, c, vr, source_name):
    dt = np.exp(log_dt.astype(np.float64))
    rounded = np.array([round(float(x), 4) for x in dt])
    print(f"\n{source_name} counts by dt:")
    for val in sorted(set(rounded)):
        mask = rounded == val
        app = mask & (vr < 0)
        rec = mask & (vr >= 0)
        print(
            f"  dt={val:0.4f} n={int(mask.sum()):5d} "
            f"c_med={np.median(c[mask]):.5f} "
            f"app%={np.mean(vr[mask] < 0):6.1%} "
            f"app_med={np.median(c[app]) if np.any(app) else np.nan:.5f} "
            f"rec_med={np.median(c[rec]) if np.any(rec) else np.nan:.5f}"
        )


def main():
    args = parse_args()
    if os.path.exists(args.out) and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.out}; use --overwrite to replace it")
    if args.old_weight <= 0 or args.window_weight <= 0:
        raise ValueError("Weights must be positive")

    old = _load_checked(args.old, "old_one_step")
    window = _load_checked(args.window, "window_target")

    n_old = len(old["r_AU"])
    n_win = len(window["r_AU"])
    print("=" * 78)
    print("BLENDING ZONE 3 DATASETS FOR V5 WINDOW-TARGET TRAINING")
    print(f"  old one-step : {args.old} ({n_old} samples), weight={args.old_weight}")
    print(f"  window target: {args.window} ({n_win} samples), weight={args.window_weight}")
    print(f"  output       : {args.out}")
    print("=" * 78)

    blended = {}
    for k in REQUIRED_FIELDS:
        blended[k] = np.concatenate([old[k], window[k]]).astype(np.float32)

    source_id = np.concatenate([
        np.zeros(n_old, dtype=np.int16),
        np.ones(n_win, dtype=np.int16),
    ])
    target_type_id = source_id.copy()  # 0 one-step, 1 window
    sample_weight = np.concatenate([
        np.full(n_old, float(args.old_weight), dtype=np.float32),
        np.full(n_win, float(args.window_weight), dtype=np.float32),
    ])

    blended["source_id"] = source_id
    blended["target_type_id"] = target_type_id
    blended["sample_weight"] = sample_weight

    np.savez_compressed(args.out, **blended)
    size_kb = os.path.getsize(args.out) / 1024.0

    print(f"Saved {args.out} ({n_old + n_win} samples, {size_kb:.1f} KB)")
    print("Metadata:")
    print("  source_id      : 0=old_one_step, 1=window_target")
    print("  target_type_id : 0=one_step_target, 1=window_target")
    print("  sample_weight  : per-sample training weight")

    src = blended["source_id"]
    w = blended["sample_weight"]
    c = blended["c_opt"].astype(np.float64)
    vr = blended["v_rad_norm"].astype(np.float64)
    log_dt = blended["log_dt"].astype(np.float64)
    print("\nBlend summary:")
    print(f"  old samples        : {int(np.sum(src == 0))}")
    print(f"  window samples     : {int(np.sum(src == 1))}")
    print(f"  nominal old weight : {float(args.old_weight):.3f}")
    print(f"  nominal win weight : {float(args.window_weight):.3f}")
    print(f"  effective old mass : {float(np.sum(w[src == 0])):.1f}")
    print(f"  effective win mass : {float(np.sum(w[src == 1])):.1f}")
    print(f"  effective win share: {float(np.sum(w[src == 1]) / np.sum(w)):.2%}")
    print(f"  blended c range    : min={np.min(c):.5f} med={np.median(c):.5f} max={np.max(c):.5f}")

    _dt_summary(log_dt[src == 0], c[src == 0], vr[src == 0], "Old one-step")
    _dt_summary(log_dt[src == 1], c[src == 1], vr[src == 1], "Window target")
    _dt_summary(log_dt, c, vr, "Blended total")


if __name__ == "__main__":
    main()
