"""
merge_encounter_shards.py

Merge small Zone-3 encounter-data shard files produced by
    generate_encounter_data_OptionA_batch.py
into one encounter_data.npz file readable by inspect_encounter_data.py.

Run from the encounter_training folder, for example:
    python merge_encounter_shards.py --shard-dir encounter_shards --out encounter_data.npz
"""

import argparse
import glob
import os
import numpy as np

REQUIRED_FIELDS = [
    "r_AU", "r_soft", "log_mi", "log_mj", "log_dt",
    "c_opt", "log_c_opt", "c_ana", "log_c_ana", "improvement",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Merge encounter-data shard .npz files.")
    ap.add_argument("--shard-dir", default="encounter_shards",
                    help="Folder containing shard .npz files.")
    ap.add_argument("--pattern", default="z3_dt*_batch*.npz",
                    help="Glob pattern for shard files inside shard-dir.")
    ap.add_argument("--out", default="encounter_data.npz",
                    help="Merged output .npz file.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite output file if it already exists.")
    return ap.parse_args()


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
    print("MERGING ENCOUNTER SHARDS")
    print(f"  shard_dir : {args.shard_dir}")
    print(f"  pattern   : {args.pattern}")
    print(f"  files     : {len(paths)}")
    print(f"  output    : {args.out}")
    print("=" * 72)

    for path in paths:
        with np.load(path) as data:
            missing = [k for k in REQUIRED_FIELDS if k not in data.files]
            if missing:
                raise KeyError(f"{path} is missing fields: {missing}")

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
                buckets[k].append(arr)

            total += n
            dt_med = float(np.exp(np.median(data["log_dt"])))
            print(f"  add {os.path.basename(path):32s} n={n:6d} dt≈{dt_med:.5f}")

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
        c_med = float(np.median(merged["c_opt"][mask]))
        c_ana_med = float(np.median(merged["c_ana"][mask]))
        print(f"  dt={dt:7.4f}  n={int(mask.sum()):6d}  "
              f"c_opt_med={c_med:.4f}  c_ana_med={c_ana_med:.6f}")


if __name__ == "__main__":
    main()
