"""
inspect_encounter_data.py  --  Zone 3 v3 data inspection

Loads the merged velocity-aware Zone 3 dataset and checks whether it is ready
for 6-input training:

    [log(r_soft), log(m_i), log(m_j), log(dt), v_rad_norm, v_tan_norm]
        -> log(c_opt)

This v3 inspector keeps the original dt/c_opt inspection, but adds the new
velocity-field checks that are required after the data-generator revamp.

Expected v3 fields:
    r_AU, r_soft, log_mi, log_mj, log_dt,
    v_rad_norm, v_tan_norm,
    c_opt, log_c_opt, c_ana, log_c_ana, improvement

Output figures:
    fig_step3a_c_opt_vs_r.png
    fig_step3b_c_diff_vs_r.png
    fig_step3c_improvement.png
    fig_step3d_c_opt_vs_dt.png
    fig_step3e_c_opt_vs_vrad.png
    fig_step3f_velocity_space.png
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EPS = 3e-4
Z3_R_MIN = 0.052
Z3_R_MAX = 0.148

REQUIRED_FIELDS = [
    "r_AU", "r_soft", "log_mi", "log_mj", "log_dt",
    "v_rad_norm", "v_tan_norm",
    "c_opt", "log_c_opt", "c_ana", "log_c_ana", "improvement",
]

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi":     150,
    "savefig.dpi":    200,
    "savefig.bbox":   "tight",
})


def parse_args():
    ap = argparse.ArgumentParser(description="Inspect velocity-aware Zone-3 encounter data.")
    ap.add_argument("--file", default="encounter_data_zone3_v3.npz",
                    help="Merged encounter-data .npz file to inspect.")
    ap.add_argument("--out-dir", default=".",
                    help="Folder for output figures.")
    return ap.parse_args()


def _c_analytic(r):
    r_soft = np.sqrt(r**2 + EPS**2)
    return (r_soft / r) ** 3


def _range_line(name, x):
    print(f"{name:14s} min={np.min(x): .6f}  med={np.median(x): .6f}  max={np.max(x): .6f}")


def _finite_all(data):
    return all(np.all(np.isfinite(data[k])) for k in REQUIRED_FIELDS)


def _dt_groups(log_dt):
    dt_vals = np.exp(log_dt.astype(np.float64))
    rounded = np.array([round(float(v), 4) for v in dt_vals])
    return dt_vals, rounded, sorted(set(rounded))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        data = np.load(args.file)
    except FileNotFoundError:
        print(f"ERROR: {args.file} not found.")
        print("Run merge_encounter_shards.py first, or pass --file <merged_npz>.")
        sys.exit(1)

    missing = [k for k in REQUIRED_FIELDS if k not in data.files]

    print("=" * 75)
    print(f"INSPECT REPORT -- {args.file}")
    print("=" * 75)
    print("FIELDS:")
    print(list(data.files))
    print("\nMISSING:")
    print(missing)
    if missing:
        raise KeyError(
            "This file is missing required v3 fields. "
            "The new velocity-aware trainer needs v_rad_norm and v_tan_norm."
        )

    print("\nSHAPES:")
    n = len(data["r_AU"])
    for k in REQUIRED_FIELDS:
        print(f"{k:14s} {data[k].shape} {data[k].dtype}")
        if len(data[k]) != n:
            raise ValueError(f"Field {k} has length {len(data[k])}, expected {n}")

    r     = data["r_AU"].astype(np.float64)
    rsoft = data["r_soft"].astype(np.float64)
    logmi = data["log_mi"].astype(np.float64)
    logmj = data["log_mj"].astype(np.float64)
    logdt = data["log_dt"].astype(np.float64)
    vr    = data["v_rad_norm"].astype(np.float64)
    vt    = data["v_tan_norm"].astype(np.float64)
    c_opt = data["c_opt"].astype(np.float64)
    logc  = data["log_c_opt"].astype(np.float64)
    c_ana = data["c_ana"].astype(np.float64)
    impr  = data["improvement"].astype(np.float64)

    dt_vals, dt_rounded, mdt_vals = _dt_groups(logdt)

    print("\nRANGES:")
    for name, arr in [
        ("r_AU", r),
        ("r_soft", rsoft),
        ("log_mi", logmi),
        ("log_mj", logmj),
        ("log_dt", logdt),
        ("v_rad_norm", vr),
        ("v_tan_norm", vt),
        ("c_opt", c_opt),
        ("log_c_opt", logc),
        ("improvement", impr),
    ]:
        _range_line(name, arr)

    print("\nCHECKS:")
    check_z3 = bool(np.all((r > Z3_R_MIN) & (r < Z3_R_MAX)))
    check_vt = bool(np.all(vt >= 0))
    check_fin = bool(_finite_all(data))
    check_c_log = bool(np.allclose(logc, np.log(c_opt + 1e-30), rtol=2e-5, atol=2e-5))
    check_rs = bool(np.allclose(rsoft, np.sqrt(r * r + EPS * EPS), rtol=2e-5, atol=2e-7))
    check_cana = bool(np.allclose(c_ana, _c_analytic(r), rtol=2e-5, atol=2e-7))
    print("r in Zone 3:", check_z3)
    print("v_tan_norm nonnegative:", check_vt)
    print("finite all:", check_fin)
    print("log_c_opt matches log(c_opt):", check_c_log)
    print("r_soft matches sqrt(r^2+eps^2):", check_rs)
    print("c_ana matches analytic formula:", check_cana)

    print("\nVELOCITY BALANCE:")
    n_app = int(np.sum(vr < 0))
    n_rec = int(np.sum(vr >= 0))
    print(f"  approaching v_rad_norm < 0 : {n_app:7d}  ({n_app / max(n, 1):6.2%})")
    print(f"  receding/side v_rad_norm >=0: {n_rec:7d}  ({n_rec / max(n, 1):6.2%})")
    print("  Note: both signs are useful; the NN needs v_rad_norm to distinguish them.")

    print("\nNEAR-IDENTITY TARGETS:")
    identity_mask = np.isclose(c_opt, 1.0, rtol=0.0, atol=1e-7) & np.isclose(logc, 0.0, rtol=0.0, atol=1e-7)
    n_identity = int(np.sum(identity_mask))
    print(f"  c_opt = 1.0 stored samples: {n_identity:7d}  ({n_identity / max(n, 1):6.2%})")
    print("  These are low-improvement cases where the generator teaches the NN to leave Zone 3 nearly unchanged.")

    print("\n" + "─" * 75)
    print("ZONE 3 SUMMARY BY dt")
    print("─" * 75)
    
    hdr = (
        f"  {'dt':>7} | {'n':>6} | {'c_med':>8} | {'c_q25':>8} | {'c_q75':>8} | "
        f"{'vr_med':>8} | {'vt_med':>8} | {'app%':>7} | {'id%':>7} | {'impr_med':>9}"
    )

    print(hdr)
    print("  " + "-" * 91)

    c_opt_meds = []
    for dt in mdt_vals:
        mask = dt_rounded == dt
        vals = c_opt[mask]
        n_dt = int(mask.sum())
        cm = float(np.median(vals))
        q25 = float(np.percentile(vals, 25))
        q75 = float(np.percentile(vals, 75))
        vrm = float(np.median(vr[mask]))
        vtm = float(np.median(vt[mask]))
        
        app_frac = float(np.mean(vr[mask] < 0))
        id_frac = float(np.mean(identity_mask[mask]))
        im = float(np.median(impr[mask]))
        c_opt_meds.append((dt, cm))
        print(
            f"  {dt:7.4f} | {n_dt:6d} | {cm:8.4f} | {q25:8.4f} | {q75:8.4f} | "
            f"{vrm:+8.3f} | {vtm:8.3f} | {app_frac:7.1%} | {id_frac:7.1%} | {im:9.2%}"
        )

    print("\n" + "─" * 75)
    print("c_opt BY RADIAL-VELOCITY SIGN")
    print("─" * 75)
    print(f"  {'dt':>7} | {'approach n':>10} | {'approach c_med':>14} | {'recede n':>9} | {'recede c_med':>12}")
    print("  " + "-" * 66)
    for dt in mdt_vals:
        base = dt_rounded == dt
        app = base & (vr < 0)
        rec = base & (vr >= 0)
        app_med = float(np.median(c_opt[app])) if np.any(app) else float("nan")
        rec_med = float(np.median(c_opt[rec])) if np.any(rec) else float("nan")
        print(
            f"  {dt:7.4f} | {int(app.sum()):10d} | {app_med:14.4f} | "
            f"{int(rec.sum()):9d} | {rec_med:12.4f}"
        )

    # Verdict: fields/checks first, then training usefulness.
    if not (check_z3 and check_vt and check_fin and check_c_log and check_rs and check_cana):
        verdict = "STOP: one or more structural data checks failed. Fix the generator/merge first."
    elif len(c_opt_meds) < 2:
        verdict = "PILOT ONLY: one dt value found. Data format is valid, but train only after multiple dt values are merged."
    else:
        c_vals = [x[1] for x in c_opt_meds]
        c_rng = float(max(c_vals) - min(c_vals))
        if c_rng > 0.10:
            verdict = f"PROCEED: valid v3 fields and strong dt-dependence confirmed (median range={c_rng:.3f})."
        elif c_rng > 0.05:
            verdict = f"PROCEED WITH CAUTION: valid v3 fields, but only mild dt-dependence (median range={c_rng:.3f})."
        else:
            verdict = f"DATA FORMAT OK, BUT WEAK SIGNAL: median c_opt range is only {c_rng:.3f}."

    print("\nVERDICT:")
    print(" ", verdict)

    # Colours for dt groups.
    colours = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(mdt_vals), 1)))
    col_map = {dt: col for dt, col in zip(mdt_vals, colours)}

    # Figure A: c_opt vs r.
    fig, ax = plt.subplots(figsize=(9, 5))
    for dt in mdt_vals:
        mask = dt_rounded == dt
        ax.scatter(r[mask], c_opt[mask], s=2, alpha=0.20,
                   color=col_map[dt], label=f"dt={dt:.4f}")
    r_plot = np.linspace(float(r.min()), float(r.max()), 300)
    ax.plot(r_plot, _c_analytic(r_plot), "k--", lw=1.5,
            label="c_analytic = (r_soft/r)^3")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":", label="c = 1.0")
    ax.set_xlabel("Separation r (AU)")
    ax.set_ylabel("c_opt")
    ax.set_title("Zone 3: c_opt vs r, coloured by macro dt")
    ax.legend(fontsize=8, markerscale=4, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0.0, min(5.5, float(np.max(c_opt)) * 1.1 + 0.3))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_step3a_c_opt_vs_r.png"))
    plt.close(fig)

    # Figure B: c_opt - c_analytic vs r.
    fig, ax = plt.subplots(figsize=(9, 5))
    for dt in mdt_vals:
        mask = dt_rounded == dt
        ax.scatter(r[mask], (c_opt - c_ana)[mask], s=2, alpha=0.20,
                   color=col_map[dt], label=f"dt={dt:.4f}")
    ax.axhline(0.0, color="k", lw=1.2, ls="--")
    ax.set_xlabel("Separation r (AU)")
    ax.set_ylabel("c_opt - c_analytic")
    ax.set_title("Zone 3: trajectory correction relative to analytic softening correction")
    ax.legend(fontsize=8, markerscale=4, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_step3b_c_diff_vs_r.png"))
    plt.close(fig)

    # Figure C: improvement distribution.
    fig, ax = plt.subplots(figsize=(9, 5))
    for dt in mdt_vals:
        mask = dt_rounded == dt
        if int(mask.sum()) < 5:
            continue
        ax.hist(impr[mask] * 100.0, bins=40, alpha=0.45,
                color=col_map[dt], label=f"dt={dt:.4f}", density=True)
    ax.set_xlabel("One-step MSE improvement over c=1.0 (%)")
    ax.set_ylabel("Density")
    
    ax.set_title("Zone 3: stored improvement distribution per dt\n"
                 "Near-identity samples are stored with improvement = 0")

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_step3c_improvement.png"))
    plt.close(fig)

    # Figure D: c_opt vs dt median/IQR.
    fig, ax = plt.subplots(figsize=(9, 5))
    dt_arr, med_arr, q25_arr, q75_arr = [], [], [], []
    rng = np.random.RandomState(0)
    for dt in mdt_vals:
        mask = dt_rounded == dt
        vals = c_opt[mask]
        if len(vals) < 5:
            continue
        idx_sub = rng.choice(len(vals), size=min(500, len(vals)), replace=False)
        ax.scatter(np.full(len(idx_sub), dt), vals[idx_sub], s=2, alpha=0.12,
                   color=col_map[dt])
        dt_arr.append(dt)
        med_arr.append(float(np.median(vals)))
        q25_arr.append(float(np.percentile(vals, 25)))
        q75_arr.append(float(np.percentile(vals, 75)))
    if dt_arr:
        dt_arr = np.array(dt_arr)
        med_arr = np.array(med_arr)
        q25_arr = np.array(q25_arr)
        q75_arr = np.array(q75_arr)
        ax.fill_between(dt_arr, q25_arr, q75_arr, alpha=0.25,
                        color="steelblue", label="IQR")
        ax.plot(dt_arr, med_arr, "o-", color="steelblue", lw=2,
                ms=6, label="Median c_opt", zorder=5)
    ax.axhline(1.0, color="k", lw=1.2, ls="--", label="c=1.0")
    ax.set_xlabel("macro dt (yr)")
    ax.set_ylabel("c_opt")
    ax.set_title("Zone 3: c_opt vs dt")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_step3d_c_opt_vs_dt.png"))
    plt.close(fig)

    # Figure E: c_opt vs v_rad_norm.
    fig, ax = plt.subplots(figsize=(9, 5))
    for dt in mdt_vals:
        mask = dt_rounded == dt
        ax.scatter(vr[mask], c_opt[mask], s=2, alpha=0.20,
                   color=col_map[dt], label=f"dt={dt:.4f}")
    ax.axvline(0.0, color="k", lw=1.2, ls="--", label="approach/recede boundary")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("v_rad_norm  (negative = approaching)")
    ax.set_ylabel("c_opt")
    ax.set_title("Zone 3: c_opt vs radial velocity state")
    ax.legend(fontsize=8, markerscale=4, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_step3e_c_opt_vs_vrad.png"))
    plt.close(fig)

    # Figure F: velocity feature space.
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(vr, vt, c=c_opt, s=3, alpha=0.35, cmap="viridis")
    ax.axvline(0.0, color="k", lw=1.2, ls="--")
    ax.set_xlabel("v_rad_norm  (negative = approaching)")
    ax.set_ylabel("v_tan_norm")
    ax.set_title("Velocity-feature coverage coloured by c_opt")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("c_opt")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_step3f_velocity_space.png"))
    plt.close(fig)

    print("\nSaved figures:")
    for name in [
        "fig_step3a_c_opt_vs_r.png",
        "fig_step3b_c_diff_vs_r.png",
        "fig_step3c_improvement.png",
        "fig_step3d_c_opt_vs_dt.png",
        "fig_step3e_c_opt_vs_vrad.png",
        "fig_step3f_velocity_space.png",
    ]:
        print(" ", os.path.join(args.out_dir, name))

    print("\n" + "=" * 75)
    print(f"FINAL VERDICT: {verdict}")
    print("=" * 75)


if __name__ == "__main__":
    main()
