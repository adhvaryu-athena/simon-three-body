"""
merge_encounter_surrogate_shards_v1.py

Merge encounter-level surrogate full-state residual shards produced by
    generate_encounter_surrogate_data_v1.py

Example:
    python -B merge_encounter_surrogate_shards_v1.py --shard-dir encounter_surrogate_shards --out encounter_surrogate_v1_merged.npz --overwrite
"""

import argparse
import glob
import os
import numpy as np

REQUIRED_FIELDS = [
    "m", "x0", "v0", "x_ias_final", "v_ias_final", "x_nonn_final", "v_nonn_final",
    "residual_x", "residual_v", "X_raw21", "X_rel18", "dt", "window_years",
    "active_i", "active_j", "third_k", "r_pair", "r_soft_pair", "v_rad_norm", "v_tan_norm",
    "source_row_idx", "category_id", "E0", "E_ias_final", "E_nonn_final", "relE_ias",
    "relE_nonn", "Lz0", "min_r_ias", "min_r_nonn", "max_radius_ias", "max_radius_nonn",
    "nonn_steps", "nonn_substeps", "nonn_zone1", "nonn_zone2", "nonn_zone3", "nonn_zone4",
    "pos_residual_rms", "vel_residual_rms", "nonn_pos_error_rms",
]


def parse_args():
    ap = argparse.ArgumentParser(description="Merge encounter surrogate v1 shard files.")
    ap.add_argument("--shard-dir", "--shard_dir", dest="shard_dir", default="encounter_surrogate_shards")
    ap.add_argument("--pattern", default="encounter_surrogate_v*_*_dt*_w*_batch*.npz")
    ap.add_argument("--out", default="encounter_surrogate_v1_merged.npz")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.shard_dir, args.pattern)))
    if not paths:
        raise FileNotFoundError(f"No shards found in {args.shard_dir!r} matching {args.pattern!r}")
    if os.path.exists(args.out) and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.out}; use --overwrite")

    buckets = {k: [] for k in REQUIRED_FIELDS}
    total = 0
    print("=" * 84)
    print("MERGING ENCOUNTER SURROGATE SHARDS v1")
    print(f"  shard_dir : {args.shard_dir}")
    print(f"  pattern   : {args.pattern}")
    print(f"  files     : {len(paths)}")
    print(f"  output    : {args.out}")
    print("=" * 84)

    for path in paths:
        with np.load(path) as data:
            missing = [k for k in REQUIRED_FIELDS if k not in data.files]
            if missing:
                raise KeyError(f"{path} missing required fields: {missing}")
            n = len(data["r_pair"])
            if n == 0:
                print(f"  skip empty: {os.path.basename(path)}")
                continue
            for k in REQUIRED_FIELDS:
                arr = np.asarray(data[k])
                if len(arr) != n:
                    raise ValueError(f"{path}: {k} length {len(arr)} != {n}")
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"{path}: {k} contains non-finite values")
                buckets[k].append(arr)
            total += n
            print(
                f"  add {os.path.basename(path):60s} n={n:5d} "
                f"dt≈{float(np.median(data['dt'])):.4f} "
                f"r_med={float(np.median(data['r_pair'])):.4f} "
                f"vr_med={float(np.median(data['v_rad_norm'])):+.3f} "
                f"posR_med={float(np.median(data['pos_residual_rms'])):.3e}"
            )

    if total == 0:
        raise RuntimeError("No non-empty shards were found")
    merged = {k: np.concatenate(v, axis=0) for k, v in buckets.items()}
    np.savez_compressed(args.out, **merged)
    print("=" * 84)
    print(f"Saved {args.out} ({total} samples, {os.path.getsize(args.out)/1024:.1f} KB)")
    print("Counts by dt:")
    dts = np.array([round(float(x), 4) for x in merged["dt"]])
    for dt in sorted(set(dts)):
        mask = dts == dt
        vr = merged["v_rad_norm"][mask]
        cat = merged["category_id"][mask]
        print(
            f"  dt={dt:.4f} n={int(np.sum(mask)):5d} "
            f"app%={float(np.mean(vr < 0)):.1%} "
            f"strong={int(np.sum(cat==0))} weak={int(np.sum(cat==1))} rec={int(np.sum(cat==2))} "
            f"posR_med={float(np.median(merged['pos_residual_rms'][mask])):.3e}"
        )
    print("Merged fields:")
    print(list(merged.keys()))


if __name__ == "__main__":
    main()
