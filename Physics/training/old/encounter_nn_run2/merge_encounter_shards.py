"""
merge_encounter_shards.py  --  Zone 3 v3 shard merger

Merge small Zone-3 encounter-data shard files produced by
    generate_encounter_data_OptionA_batch.py
into one compressed .npz file for inspection and training.

This v3 merger expects the new velocity-aware fields:
    v_rad_norm, v_tan_norm

Run from the encounter_training folder, for example:
    python -B merge_encounter_shards.py --shard-dir encounter_shards --out encounter_data_zone3_v3.npz

The default pattern matches both old shard names such as:
    z3_dt0p005_batch001.npz
and new v3 shard names such as:
    encounter_data_zone3_v3_dt0p005_batch001.npz
"""

import argparse
import glob
import os
import numpy as np

REQUIRED_FIELDS = [
    "r_AU", "r_soft", "log_mi", "log_mj", "log_dt",
    "v_rad_norm", "v_tan_norm",
    "c_opt", "log_c_opt", "c_ana", "log_c_ana", "improvement",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Merge velocity-aware Zone-3 encounter-data shard .npz files.")
    ap.add_argument("--shard-dir", default="encounter_shards",
                    help="Folder containing shard .npz files.")
    ap.add_argument("--pattern", default="*_dt*_batch*.npz",
                    help="Glob pattern for shard files inside shard-dir.")
    ap.add_argument("--out", default="encounter_data_zone3_v3.npz",
                    help="Merged output .npz file.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite output file if it already exists.")
    return ap.parse_args()


def _median_or_nan(x):
    return float(np.median(x)) if len(x) else float("nan")


def main():
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.shard_dir, args.pattern)))

    if not paths:
        raise FileNotFoundError(
            f"No shard files found in {args.shard_dir!r} matching {args.pattern!r}"
        )

    if os.path.exists(args.out) and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {args.out}\nUse --overwrite if you want to replace it."
        )

    buckets = {k: [] for k in REQUIRED_FIELDS}
    total = 0
    skipped_empty = 0

    print("=" * 72)
    print("MERGING VELOCITY-AWARE ENCOUNTER SHARDS")
    print(f"  shard_dir : {args.shard_dir}")
    print(f"  pattern   : {args.pattern}")
    print(f"  files     : {len(paths)}")
    print(f"  output    : {args.out}")
    print("  required  : " + ", ".join(REQUIRED_FIELDS))
    print("=" * 72)

    for path in paths:
        with np.load(path) as data:
            missing = [k for k in REQUIRED_FIELDS if k not in data.files]
            if missing:
                raise KeyError(
                    f"{path} is missing fields: {missing}\n"
                    "This merger is for the v3 velocity-aware dataset. "
                    "Regenerate this shard with the v3 generator, or use the old merger for old data."
                )

            n = int(len(data["r_AU"]))
            if n == 0:
                skipped_empty += 1
                print(f"  skip empty: {path}")
                continue

            for k in REQUIRED_FIELDS:
                arr = np.asarray(data[k], dtype=np.float32)
                if len(arr) != n:
                    raise ValueError(
                        f"{path}: field {k} has length {len(arr)}, expected {n}"
                    )
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"{path}: field {k} contains non-finite values")
                buckets[k].append(arr)

            total += n
            dt_med = float(np.exp(np.median(data["log_dt"])))
            vr_med = float(np.median(data["v_rad_norm"]))
            vt_med = float(np.median(data["v_tan_norm"]))
            c_med = float(np.median(data["c_opt"]))
            print(
                f"  add {os.path.basename(path):42s} n={n:6d} "
                f"dt≈{dt_med:.5f} c_med={c_med:.4f} "
                f"vr_med={vr_med:+.3f} vt_med={vt_med:.3f}"
            )

    if total == 0:
        raise RuntimeError("No non-empty shard files were found.")

    merged = {k: np.concatenate(v).astype(np.float32) for k, v in buckets.items()}
    np.savez_compressed(args.out, **merged)

    size_kb = os.path.getsize(args.out) / 1024
    print("=" * 72)
    print(f"Saved {args.out} ({total} samples, {size_kb:.0f} KB)")
    if skipped_empty:
        print(f"Skipped empty shards: {skipped_empty}")

    # Count by dt for quick verification.
    log_dt = merged["log_dt"].astype(np.float64)
    dt_vals = np.exp(log_dt)
    rounded = np.array([round(float(x), 4) for x in dt_vals])
    print("\nCounts by dt:")
    for dt in sorted(set(rounded)):
        mask = rounded == dt
        c_med = _median_or_nan(merged["c_opt"][mask])
        c_ana_med = _median_or_nan(merged["c_ana"][mask])
        vr_med = _median_or_nan(merged["v_rad_norm"][mask])
        vt_med = _median_or_nan(merged["v_tan_norm"][mask])
        n_app = int(np.sum(mask & (merged["v_rad_norm"] < 0)))
        n_rec = int(np.sum(mask & (merged["v_rad_norm"] >= 0)))
        print(
            f"  dt={dt:7.4f}  n={int(mask.sum()):6d}  "
            f"c_opt_med={c_med:.4f}  c_ana_med={c_ana_med:.6f}  "
            f"vr_med={vr_med:+.3f}  vt_med={vt_med:.3f}  "
            f"approach={n_app:5d}  recede={n_rec:5d}"
        )

    print("\nMerged fields:")
    print(list(merged.keys()))


if __name__ == "__main__":
    main()
