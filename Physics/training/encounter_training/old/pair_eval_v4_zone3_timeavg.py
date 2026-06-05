# pair_eval_v4_zone3_timeavg.py
#
# Controlled SIMON evaluator for the revised Zone 2 / Zone 3 method.
#
# Starting point: pair_eval_after_adaptive_timeavg.py.
# Main methodological changes:
#   1. Zone 2 (r < 0.05 AU): direct Newtonian force only; adaptive sub-stepping
#      handles time resolution. No old analytic-softening NN is used here.
#   2. Zone 3 (0.05 <= r < 0.15 AU): bounded v4 trajectory-trained 6-input NN.
#   3. Zone 4 (r >= 0.15 AU): direct Newtonian force.
#
# The v4 model input is:
#   [log(r_soft), log(m_i), log(m_j), log(dt), v_rad_norm, v_tan_norm]
# where v_rad_norm < 0 means the pair is approaching.
#
# Outputs are written to a separate folder by default:
#   v4_zone3_eval_out/
#
# Typical run from C:\Aarush\Physics\training\encounter_training:
#   python -B pair_eval_v4_zone3_timeavg.py --model_path pair_correction_nn_v4_bounded.pt --dt_rep 0.04
#
# Optional previous-run comparison:
#   If perf_summary_T100_timeavg.txt exists in the same folder, the script parses it
#   and writes old-vs-v4 comparison tables/figures.

import os, time, math, argparse, re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "font.size": 15, "axes.titlesize": 20, "axes.labelsize": 20,
    "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 15,
    "figure.titlesize": 20, "mathtext.fontset": "cm", "mathtext.rm": "serif",
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.unicode_minus": False,
})

try:
    import rebound
except ImportError:
    print("[ERROR] rebound not found. Install it in your environment.")
    raise


# =============================================================================
# v4 bounded 6-input model
# =============================================================================
class PairCorrectionNNv4(nn.Module):
    def __init__(self, hidden: int = 64, c_min: float = 0.25, c_max: float = 3.0):
        super().__init__()
        if not (0.0 < float(c_min) < float(c_max)):
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


def infer_hidden_from_state_dict(sd: Dict[str, torch.Tensor]) -> int:
    if "net.0.weight" not in sd:
        raise KeyError("Model state_dict does not contain net.0.weight")
    w = sd["net.0.weight"]
    if w.ndim != 2 or w.shape[1] != 6:
        raise ValueError(
            f"Expected v4 first layer shape (hidden, 6), got {tuple(w.shape)}. "
            "This evaluator requires pair_correction_nn_v4_bounded.pt, not the old 3-input model."
        )
    return int(w.shape[0])


def load_v4_model(model_path: str) -> PairCorrectionNNv4:
    sd = torch.load(model_path, map_location="cpu")
    hidden = infer_hidden_from_state_dict(sd)
    if "log_c_min" not in sd or "log_c_max" not in sd:
        raise KeyError("The model does not contain log_c_min/log_c_max; it is not a v4 bounded model.")
    c_min = float(torch.exp(sd["log_c_min"]).cpu().item())
    c_max = float(torch.exp(sd["log_c_max"]).cpu().item())
    model = PairCorrectionNNv4(hidden=hidden, c_min=c_min, c_max=c_max)
    model.load_state_dict(sd)
    model.eval()
    print(f"[eval-v4] Loaded {model_path} | hidden={hidden} | params={sum(p.numel() for p in model.parameters())}")
    print(f"[eval-v4] v4 bounded c range: [{c_min:.6g}, {c_max:.6g}]")
    return model


def extract_weights_numpy_v4(model: PairCorrectionNNv4) -> Dict[str, np.ndarray]:
    sd = model.state_dict()
    return {
        "mean": sd["input_mean"].cpu().numpy().astype(np.float32),
        "std": sd["input_std"].cpu().numpy().astype(np.float32) + 1e-8,
        "log_c_min": np.float64(sd["log_c_min"].cpu().item()),
        "log_c_max": np.float64(sd["log_c_max"].cpu().item()),
        "w0T": sd["net.0.weight"].cpu().numpy().T.astype(np.float32).copy(),
        "b0": sd["net.0.bias"].cpu().numpy().astype(np.float32),
        "w1T": sd["net.2.weight"].cpu().numpy().T.astype(np.float32).copy(),
        "b1": sd["net.2.bias"].cpu().numpy().astype(np.float32),
        "w2T": sd["net.4.weight"].cpu().numpy().T.astype(np.float32).copy(),
        "b2": sd["net.4.bias"].cpu().numpy().astype(np.float32),
        "w3T": sd["net.6.weight"].cpu().numpy().T.astype(np.float32).copy(),
        "b3": sd["net.6.bias"].cpu().numpy().astype(np.float32),
    }


@dataclass
class HybridConfig:
    G: float = 1.0
    eps: float = 3e-4
    # Hard safety gates. v4 already bounds output to ~[0.25,3.0], but keep
    # the deployment gate in the evaluator for safety and reporting.
    c_min: float = 0.2
    c_max: float = 5.0
    r_soft_min: float = 5e-4
    zone1_r_gate: float = 4e-4
    adapt_thresh: float = 0.05
    nn_thresh: float = 0.15
    max_substeps: int = 16


# =============================================================================
# Numerical helpers
# =============================================================================
def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _silu_np(x: np.ndarray) -> np.ndarray:
    return x * _sigmoid_np(x)


def v4_forward_numpy(nn_in: np.ndarray, w: Dict[str, np.ndarray]) -> np.ndarray:
    """Return bounded c for rows of 6-input v4 features."""
    h = (nn_in - w["mean"]) / w["std"]
    h = _silu_np(h @ w["w0T"] + w["b0"])
    h = _silu_np(h @ w["w1T"] + w["b1"])
    h = _silu_np(h @ w["w2T"] + w["b2"])
    raw = (h @ w["w3T"] + w["b3"]).ravel().astype(np.float64)
    s = _sigmoid_np(raw)
    log_c = w["log_c_min"] + (w["log_c_max"] - w["log_c_min"]) * s
    return np.exp(log_c).astype(np.float64)


def simulate_rebound_ias15(x0, v0, m, G, T, n_samples):
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G
    for i in range(len(m)):
        sim.add(m=float(m[i]), x=float(x0[i,0]), y=float(x0[i,1]), z=float(x0[i,2]),
                vx=float(v0[i,0]), vy=float(v0[i,1]), vz=float(v0[i,2]))
    sim.move_to_com()
    times = np.linspace(0.0, T, n_samples)
    pos = np.zeros((n_samples, len(m), 3), dtype=np.float64)
    vel = np.zeros((n_samples, len(m), 3), dtype=np.float64)
    t0 = time.perf_counter()
    for k, t in enumerate(times):
        sim.integrate(t)
        for i, p in enumerate(sim.particles):
            pos[k, i] = [p.x, p.y, p.z]
            vel[k, i] = [p.vx, p.vy, p.vz]
    return times, pos, vel, {"total_time_sec": time.perf_counter() - t0}


def rms_sep(a, b):
    d = a - b
    pb = np.sqrt(np.sum(d**2, axis=-1))
    return np.sqrt(np.mean(pb**2, axis=1))


def fit_log_slope(times, delta, t0_frac=0.10, t1_frac=0.50):
    T = times[-1]
    t0 = t0_frac * T
    t1 = t1_frac * T
    mask = (times >= t0) & (times <= t1)
    x = times[mask]
    y = np.log(np.clip(delta[mask], 1e-30, None))
    x0 = x.mean()
    y0 = y.mean()
    return float(np.sum((x-x0)*(y-y0)) / (np.sum((x-x0)**2) + 1e-30)), (t0, t1)


# =============================================================================
# Revised v4 Zone 2 / Zone 3 simulator
# =============================================================================
def simulate_leapfrog_v4_zone3(
    x0, v0, m, model: PairCorrectionNNv4, cfg: HybridConfig, dt, T, n_samples,
    use_zone3_nn: bool = True,
    label: str = "v4_zone3_nn",
):
    """
    Revised evaluator:
      Zone 2: r < 0.05 AU -> direct Newtonian force; sub-stepping handles time resolution.
      Zone 3: 0.05 <= r < 0.15 AU -> v4 bounded trajectory NN.
      Far field: direct Newtonian.

    If use_zone3_nn=False, Zone 3 uses c=1 times F_soft for a same-method no-NN baseline.
    Zone 2 remains direct Newtonian in both cases.
    """
    w = extract_weights_numpy_v4(model)
    N = x0.shape[0]
    ii, jj = [], []
    for i in range(N):
        for j in range(i+1, N):
            ii.append(i); jj.append(j)
    ii = np.array(ii, dtype=np.int64)
    jj = np.array(jj, dtype=np.int64)
    P = len(ii)

    G = cfg.G
    eps2 = cfg.eps * cfg.eps
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    m_f = m.astype(np.float64)
    mi_arr = m_f[ii]
    mj_arr = m_f[jj]
    Gmimj = G * mi_arr * mj_arr
    inv_mi = 1.0 / mi_arr
    inv_mj = 1.0 / mj_arr
    log_mi = np.log(mi_arr + 1e-30).astype(np.float32)
    log_mj = np.log(mj_arr + 1e-30).astype(np.float32)

    times = np.linspace(0.0, T, n_samples)
    n_steps = int(math.ceil(T / dt))
    pos_out = np.zeros((n_samples, N, 3), dtype=np.float64)
    vel_out = np.zeros((n_samples, N, 3), dtype=np.float64)

    stats = {
        "pair_evals": 0,
        "zone1_pairs": 0,
        "zone2_pairs": 0,
        "zone3_pairs": 0,
        "zone3_nn_pairs": 0,
        "zone3_gate_fallback_pairs": 0,
        "zone3_no_nn_pairs": 0,
        "far_pairs": 0,
        "strong_approach_zone3_pairs": 0,
        "c_sum": 0.0,
        "c_count": 0,
        "c_min": np.inf,
        "c_max": -np.inf,
    }

    def compute_acc(pos, vel, step_dt):
        rij = pos[jj] - pos[ii]                       # (P,3)
        vij = vel[jj] - vel[ii]                       # (P,3)
        r2 = np.einsum("ij,ij->i", rij, rij)
        r = np.sqrt(r2 + 1e-30)

        # Default is exact Newtonian force for every pair.
        F_scalar = Gmimj / (r2 * r + 1e-30)

        zone1_mask = r < cfg.zone1_r_gate
        zone2_mask = (r >= cfg.zone1_r_gate) & (r < cfg.adapt_thresh)
        zone3_mask = (r >= cfg.adapt_thresh) & (r < cfg.nn_thresh)
        far_mask = r >= cfg.nn_thresh

        stats["pair_evals"] += P
        stats["zone1_pairs"] += int(np.sum(zone1_mask))
        stats["zone2_pairs"] += int(np.sum(zone2_mask))
        stats["zone3_pairs"] += int(np.sum(zone3_mask))
        stats["far_pairs"] += int(np.sum(far_mask))

        # Zone 2 intentionally does nothing here: F_scalar is already Newtonian.
        # Zone 1 also remains Newtonian as a hard safety fallback.

        if np.any(zone3_mask):
            z = zone3_mask
            r_soft = np.sqrt(r2[z] + eps2)
            F_soft = Gmimj[z] / ((r2[z] + eps2) ** 1.5 + 1e-30)

            if use_zone3_nn:
                # Velocity features match the generator convention:
                # v_rad_norm = dot(v_rel, r_hat) / sqrt(G*(mi+mj)/r)
                # negative = approaching, positive = receding.
                r_hat = rij[z] / (r[z, None] + 1e-30)
                v_rad = np.einsum("ij,ij->i", vij[z], r_hat)
                v_tan_vec = vij[z] - v_rad[:, None] * r_hat
                v_tan = np.sqrt(np.einsum("ij,ij->i", v_tan_vec, v_tan_vec) + 1e-30)
                v_scale = np.sqrt(G * (mi_arr[z] + mj_arr[z]) / (r[z] + 1e-30))
                v_rad_norm = v_rad / (v_scale + 1e-30)
                v_tan_norm = v_tan / (v_scale + 1e-30)

                nn_in = np.empty((int(np.sum(z)), 6), dtype=np.float32)
                nn_in[:, 0] = np.log(r_soft + 1e-30).astype(np.float32)
                nn_in[:, 1] = log_mi[z]
                nn_in[:, 2] = log_mj[z]
                nn_in[:, 3] = np.float32(np.log(float(step_dt) + 1e-30))
                nn_in[:, 4] = v_rad_norm.astype(np.float32)
                nn_in[:, 5] = v_tan_norm.astype(np.float32)

                c = v4_forward_numpy(nn_in, w)
                fb = ((r_soft < cfg.r_soft_min) | (c < cfg.c_min) |
                      (c > cfg.c_max) | ~np.isfinite(c))

                F_scalar[z] = np.where(fb, F_scalar[z], c * F_soft)

                stats["zone3_nn_pairs"] += int(np.sum(z))
                stats["zone3_gate_fallback_pairs"] += int(np.sum(fb))
                stats["strong_approach_zone3_pairs"] += int(np.sum(v_rad_norm < -0.6))
                good_c = c[np.isfinite(c)]
                if len(good_c):
                    stats["c_sum"] += float(np.sum(good_c))
                    stats["c_count"] += int(len(good_c))
                    stats["c_min"] = min(stats["c_min"], float(np.min(good_c)))
                    stats["c_max"] = max(stats["c_max"], float(np.max(good_c)))
            else:
                # Same revised Zone 2/3 method but no learned Zone 3 correction.
                # This is analogous to c=1 in Zone 3 only; Zone 2 is still Newtonian.
                F_scalar[z] = F_soft
                stats["zone3_no_nn_pairs"] += int(np.sum(z))

        F_vec = F_scalar[:, None] * rij
        acc = np.zeros((N, 3), dtype=np.float64)
        for p in range(P):
            acc[ii[p]] += F_vec[p] * inv_mi[p]
            acc[jj[p]] -= F_vec[p] * inv_mj[p]
        return acc

    def min_pair_dist(pos):
        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        return np.sqrt(np.min(r2) + 1e-30)

    def leapfrog_substep(x_in, v_in, a_in, sub_dt):
        vh = v_in + 0.5 * sub_dt * a_in
        x_new = x_in + sub_dt * vh
        # Use half-step velocity to compute velocity-aware Zone 3 features.
        a_new = compute_acc(x_new, vh, sub_dt)
        v_new = vh + 0.5 * sub_dt * a_new
        return x_new, v_new, a_new

    a = compute_acc(x, v, float(dt))
    si = 0
    nt = times[0]
    t_cur = 0.0
    while si < n_samples and t_cur >= nt - 1e-12:
        pos_out[si] = x
        vel_out[si] = v
        si += 1
        if si < n_samples:
            nt = times[si]

    steps = 0
    dt_f = float(dt)
    total_substeps = 0
    t_start = time.perf_counter()
    for _ in range(n_steps):
        r_min = min_pair_dist(x)
        if r_min < cfg.adapt_thresh:
            n_sub = min(cfg.max_substeps, max(2, int(np.ceil(cfg.adapt_thresh / r_min))))
            sub_dt = dt_f / n_sub
            for _ in range(n_sub):
                x, v, a = leapfrog_substep(x, v, a, sub_dt)
            total_substeps += n_sub
        else:
            vh = v + 0.5 * dt_f * a
            x = x + dt_f * vh
            a = compute_acc(x, vh, dt_f)
            v = vh + 0.5 * dt_f * a
            total_substeps += 1

        t_cur += dt_f
        steps += 1
        while si < n_samples and t_cur >= nt - 1e-12:
            pos_out[si] = x
            vel_out[si] = v
            si += 1
            if si < n_samples:
                nt = times[si]
        if t_cur >= T - 1e-12:
            break

    while si < n_samples:
        pos_out[si] = x
        vel_out[si] = v
        si += 1

    total_time = time.perf_counter() - t_start
    pair_evals = max(int(stats["pair_evals"]), 1)
    c_count = max(int(stats["c_count"]), 1)
    c_min = float(stats["c_min"]) if np.isfinite(stats["c_min"]) else float("nan")
    c_max = float(stats["c_max"]) if np.isfinite(stats["c_max"]) else float("nan")

    perf = {
        "label": label,
        "steps": steps,
        "dt": dt,
        "T_years": T,
        "n_samples": n_samples,
        "total_time_sec": total_time,
        "time_per_step_sec": total_time / max(steps, 1),
        "avg_pairs_per_step": P,
        "total_substeps": total_substeps,
        "zone1_frac": stats["zone1_pairs"] / pair_evals,
        "zone2_frac": stats["zone2_pairs"] / pair_evals,
        "zone3_frac": stats["zone3_pairs"] / pair_evals,
        "zone4_far_frac": stats["far_pairs"] / pair_evals,
        "zone3_nn_frac": stats["zone3_nn_pairs"] / pair_evals,
        "zone3_gate_fallback_frac": stats["zone3_gate_fallback_pairs"] / max(int(stats["zone3_nn_pairs"]), 1),
        "strong_approach_zone3_frac": stats["strong_approach_zone3_pairs"] / max(int(stats["zone3_nn_pairs"]), 1),
        "zone3_no_nn_frac": stats["zone3_no_nn_pairs"] / pair_evals,
        "c_pred_mean": stats["c_sum"] / c_count,
        "c_pred_min": c_min,
        "c_pred_max": c_max,
    }
    # Backward-compatible alias. In the old file this meant close-pair NN fraction.
    perf["avg_fallback_frac"] = perf["zone3_nn_frac"]
    return times, pos_out, vel_out, perf


# =============================================================================
# Plotting helpers, similar to pair_eval_after_adaptive_timeavg.py
# =============================================================================
def plot_overlay_xy_subplots_all_bodies(pr, pm, op, model_label="Hybrid"):
    N = pr.shape[1]
    fig, axes = plt.subplots(1, N, figsize=(5.2*N, 4.0))
    if N == 1:
        axes = [axes]
    for i in range(N):
        ax = axes[i]
        ax.plot(pr[:, i, 0], pr[:, i, 1], "-",  lw=1.4, label="IAS15")
        ax.plot(pm[:, i, 0], pm[:, i, 1], "--", lw=1.4, label=model_label)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_title(f"Body {i}", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(op, dpi=300)
    plt.close()


def plot_overlay_timeseries_all_bodies(times, pr, pm, op, model_label="Hybrid"):
    N = pr.shape[1]
    fig, axes = plt.subplots(N, 2, figsize=(10.5, 2.8*N), sharex=True)
    if N == 1:
        axes = np.array([axes])
    for i in range(N):
        axes[i, 0].plot(times, pr[:, i, 0], "-",  lw=1.3, label="IAS15")
        axes[i, 0].plot(times, pm[:, i, 0], "--", lw=1.3, label=model_label)
        axes[i, 0].set_ylabel(f"Body {i}: x(t)", fontsize=11)
        axes[i, 0].tick_params(axis="both", labelsize=10)
        axes[i, 0].grid(True, alpha=0.25)
        axes[i, 1].plot(times, pr[:, i, 1], "-",  lw=1.3, label="IAS15")
        axes[i, 1].plot(times, pm[:, i, 1], "--", lw=1.3, label=model_label)
        axes[i, 1].set_ylabel(f"Body {i}: y(t)", fontsize=11)
        axes[i, 1].tick_params(axis="both", labelsize=10)
        axes[i, 1].grid(True, alpha=0.25)
        if i == 0:
            axes[i, 0].legend(loc="upper left", fontsize=9, framealpha=0.85)
    axes[-1, 0].set_xlabel("Time (yr)", fontsize=11)
    axes[-1, 1].set_xlabel("Time (yr)", fontsize=11)
    plt.tight_layout()
    plt.savefig(op, dpi=300)
    plt.close()


def plot_divergence(times, delta, slope, window, op):
    plt.figure()
    plt.semilogy(times, delta, label="d(t) = RMS(model - IAS15)")
    plt.axvspan(window[0], window[1], alpha=0.15, label="fit window")
    plt.xlabel("Time (yr)")
    plt.ylabel("Root Mean Square Position")
    plt.title("SIMON-vs-REBOUND Divergence")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(op, dpi=200)
    plt.close()


def plot_two_divergence(times, delta_v4, delta_no, op):
    with plt.rc_context({"font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9}):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.semilogy(times, delta_v4, label="v4 Zone 3 NN")
        ax.semilogy(times, delta_no, "--", label="Zone 3 no-NN baseline")
        ax.set_xlabel("Time (yr)")
        ax.set_ylabel("RMS deviation vs IAS15 (AU, log)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(op, dpi=300)
        plt.close()


def plot_speed_accuracy_frontier(pts, op, old_pts: Optional[List[Dict[str, float]]] = None):
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8}):
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        xs = [p["throughput_samp_per_sec"] for p in pts]
        ys = [p["time_avg_rms"] for p in pts]
        ax.scatter(xs, ys, s=42, zorder=3, label="v4 Zone 3 NN")
        ax.plot(xs, ys, lw=1.0, alpha=0.6)
        for p in pts:
            ax.annotate(f"dt={p['dt']}", xy=(p["throughput_samp_per_sec"], p["time_avg_rms"]),
                        xytext=(4, 8), textcoords="offset points", fontsize=9)
        if old_pts:
            ox = [p["throughput_samp_per_sec"] for p in old_pts]
            oy = [p["time_avg_rms"] for p in old_pts]
            ax.scatter(ox, oy, marker="x", s=50, zorder=3, label="previous Run D")
            ax.plot(ox, oy, linestyle="--", lw=1.0, alpha=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Throughput (recorded samples / sec, log)")
        ax.set_ylabel("Time-averaged RMS deviation vs IAS15 over 0–100 yr (log)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.savefig(op, dpi=300)
        plt.close()


def plot_comp_cost_frontier(pts, op):
    dts = [p["dt"] for p in pts]
    tps = [p["time_per_step_sec"] for p in pts]
    tots = [p["total_time_sec"] for p in pts]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(dts, tps, marker="o")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel("dt (yr)"); axes[0].set_ylabel("time per integration step (sec)")
    axes[0].set_title("Time Per Step vs dt"); axes[0].grid(True, which="both", alpha=0.3)
    axes[1].plot(dts, tots, marker="o")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("dt (years, log)"); axes[1].set_ylabel("total simulation time (sec, log)")
    axes[1].set_title("Total sim time vs dt"); axes[1].grid(True, which="both", alpha=0.3)
    fig.suptitle("Computational cost (v4 Zone 3 model), T=100 yr")
    plt.tight_layout()
    plt.savefig(op, dpi=200)
    plt.close()


def plot_old_vs_new_bars(comparison_rows, op):
    if not comparison_rows:
        return
    with plt.rc_context({"font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8}):
        dts = [r["dt"] for r in comparison_rows]
        x = np.arange(len(dts))
        width = 0.36
        old = [r.get("old_time_avg_rms", np.nan) for r in comparison_rows]
        new = [r.get("new_time_avg_rms", np.nan) for r in comparison_rows]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.bar(x - width/2, old, width, label="previous Run D")
        ax.bar(x + width/2, new, width, label="v4 Zone 3")
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in dts])
        ax.set_yscale("log")
        ax.set_xlabel("dt (yr)")
        ax.set_ylabel("Time-averaged RMS vs IAS15 (AU, log)")
        ax.grid(True, axis="y", which="both", alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.savefig(op, dpi=300)
        plt.close()


# =============================================================================
# Previous-run summary parser
# =============================================================================
def parse_previous_summary(path: str) -> Tuple[Optional[float], List[Dict[str, float]]]:
    if not path or not os.path.exists(path):
        return None, []
    ias_time = None
    pts = []
    in_table = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "total_time_sec:" in line and ias_time is None:
                try:
                    ias_time = float(line.split(":", 1)[1].strip())
                except Exception:
                    pass
            if line.strip().startswith("dt\tfinal_err"):
                in_table = True
                continue
            if in_table:
                s = line.strip()
                if not s or s.startswith("Notes"):
                    break
                parts = re.split(r"\s+", s)
                if len(parts) >= 8:
                    try:
                        pts.append({
                            "dt": float(parts[0]),
                            "final_err": float(parts[1]),
                            "time_avg_rms": float(parts[2]),
                            "throughput_samp_per_sec": float(parts[3]),
                            "time_per_step_sec": float(parts[4]),
                            "total_time_sec": float(parts[5]),
                            "steps": float(parts[6]),
                            "avg_fallback_frac": float(parts[7]),
                        })
                    except Exception:
                        pass
    print(f"[eval-v4] Previous summary: {path} | loaded {len(pts)} frontier rows")
    return ias_time, pts


def make_comparison_rows(old_pts, new_pts):
    old_by_dt = {round(float(p["dt"]), 6): p for p in old_pts}
    rows = []
    for p in new_pts:
        d = round(float(p["dt"]), 6)
        if d not in old_by_dt:
            continue
        old = old_by_dt[d]
        old_rms = old["time_avg_rms"]
        new_rms = p["time_avg_rms"]
        old_final = old["final_err"]
        new_final = p["final_err"]
        rows.append({
            "dt": p["dt"],
            "old_time_avg_rms": old_rms,
            "new_time_avg_rms": new_rms,
            "time_avg_change_pct": 100.0 * (new_rms - old_rms) / max(abs(old_rms), 1e-30),
            "old_final_err": old_final,
            "new_final_err": new_final,
            "final_err_change_pct": 100.0 * (new_final - old_final) / max(abs(old_final), 1e-30),
            "old_speedup": old.get("speedup", np.nan),
            "new_speedup": p.get("speedup", np.nan),
        })
    return rows


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="pair_correction_nn_v4_bounded.pt")
    parser.add_argument("--out_dir", default="v4_zone3_eval_out")
    parser.add_argument("--old_summary", default="perf_summary_T100_timeavg.txt")
    parser.add_argument("--dt_rep", type=float, default=0.04)
    parser.add_argument("--T", type=float, default=100.0)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--dts", default="0.005,0.01,0.02,0.04,0.08")
    parser.add_argument("--skip_no_nn_baseline", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = HybridConfig()

    mp = args.model_path
    if not mp.endswith(".pt") and os.path.exists(mp + ".pt"):
        mp += ".pt"
    model = load_v4_model(mp)

    T = float(args.T)
    ns = int(args.n_samples)
    dts = [float(x.strip()) for x in args.dts.split(",") if x.strip()]

    # IC1 default, same as pair_eval_after_adaptive_timeavg.py.
    m = np.array([1.0, 0.01, 0.005], dtype=np.float64)
    x0 = np.array([[0,0,0], [1,0,0], [0,1.2,0]], dtype=np.float64)
    v0 = np.array([[0,0,0], [0,1,0], [-0.9,0,0]], dtype=np.float64)
    M = m.sum()
    x0 = x0 - (m[:, None] * x0).sum(0) / M
    v0 = v0 - (m[:, None] * v0).sum(0) / M

    print("[eval-v4] IAS15 baseline...")
    tr, pr, vr, perf_r = simulate_rebound_ias15(x0, v0, m, cfg.G, T, ns)
    print(f"[eval-v4] IAS15 time={perf_r['total_time_sec']:.3f}s")

    dr = float(args.dt_rep)
    print(f"[eval-v4] Representative v4 Zone 3 run dt={dr}...")
    tm, pm, vm, perf_m = simulate_leapfrog_v4_zone3(x0, v0, m, model, cfg, dr, T, ns, use_zone3_nn=True, label="v4_zone3_nn")
    sp = perf_r["total_time_sec"] / max(perf_m["total_time_sec"], 1e-12)
    delta = rms_sep(pm, pr)
    slope, win = fit_log_slope(tr, delta)
    final_err = float(delta[-1])
    time_avg_rms = float(np.sqrt(np.mean(delta**2)))
    print(f"[eval-v4] v4 dt={dr}: final_err={final_err:.3e} time_avg_rms={time_avg_rms:.3e} "
          f"time={perf_m['total_time_sec']:.3f}s speedup={sp:.2f}x zone3_nn_frac={perf_m['zone3_nn_frac']:.4f}")

    plot_overlay_xy_subplots_all_bodies(pr, pm, os.path.join(args.out_dir, "traj_overlay_xy_subplots_all_bodies_T100_v4.png"), model_label="v4 Zone 3")
    plot_overlay_timeseries_all_bodies(tr, pr, pm, os.path.join(args.out_dir, "traj_overlay_timeseries_all_bodies_T100_v4.png"), model_label="v4 Zone 3")
    plot_divergence(tr, delta, slope, win, os.path.join(args.out_dir, "model_vs_rebound_divergence_T100_v4.png"))

    no_nn_rep = None
    if not args.skip_no_nn_baseline:
        print(f"[eval-v4] Representative revised no-NN Zone 3 baseline dt={dr}...")
        _, p_no, _, perf_no = simulate_leapfrog_v4_zone3(x0, v0, m, model, cfg, dr, T, ns, use_zone3_nn=False, label="zone3_no_nn")
        delta_no = rms_sep(p_no, pr)
        no_nn_rep = {
            "perf": perf_no,
            "final_err": float(delta_no[-1]),
            "time_avg_rms": float(np.sqrt(np.mean(delta_no**2))),
        }
        print(f"[eval-v4] no-NN Zone 3 dt={dr}: final_err={no_nn_rep['final_err']:.3e} "
              f"time_avg_rms={no_nn_rep['time_avg_rms']:.3e}")
        plot_two_divergence(tr, delta, delta_no, os.path.join(args.out_dir, "v4_vs_zone3_no_nn_divergence_T100.png"))

    old_ias_time, old_pts = parse_previous_summary(args.old_summary)

    print("[eval-v4] Frontier sweep...")
    frontier = []
    for dt_val in dts:
        _, pd, _, pf = simulate_leapfrog_v4_zone3(x0, v0, m, model, cfg, dt_val, T, ns, use_zone3_nn=True, label="v4_zone3_nn")
        rms_t = rms_sep(pd, pr)
        fe = float(rms_t[-1])
        tar = float(np.sqrt(np.mean(rms_t**2)))
        tp = ns / max(pf["total_time_sec"], 1e-12)
        s2 = perf_r["total_time_sec"] / max(pf["total_time_sec"], 1e-12)
        row = {
            "dt": dt_val,
            "final_err": fe,
            "time_avg_rms": tar,
            "throughput_samp_per_sec": tp,
            "time_per_step_sec": pf["time_per_step_sec"],
            "total_time_sec": pf["total_time_sec"],
            "avg_fallback_frac": pf["zone3_nn_frac"],
            "zone2_frac": pf["zone2_frac"],
            "zone3_frac": pf["zone3_frac"],
            "zone3_nn_frac": pf["zone3_nn_frac"],
            "zone3_gate_fallback_frac": pf["zone3_gate_fallback_frac"],
            "strong_approach_zone3_frac": pf["strong_approach_zone3_frac"],
            "c_pred_mean": pf["c_pred_mean"],
            "c_pred_min": pf["c_pred_min"],
            "c_pred_max": pf["c_pred_max"],
            "steps": pf["steps"],
            "total_substeps": pf["total_substeps"],
            "speedup": s2,
        }
        frontier.append(row)
        print(f"  dt={dt_val:>6} final_err={fe:.3e} time_avg_rms={tar:.3e} "
              f"thrpt={tp:.0f} t/step={pf['time_per_step_sec']:.2e} total={pf['total_time_sec']:.3f}s "
              f"zone2={pf['zone2_frac']:.4f} zone3NN={pf['zone3_nn_frac']:.4f} "
              f"gate={pf['zone3_gate_fallback_frac']:.2%} speedup={s2:.2f}x")

    plot_speed_accuracy_frontier(frontier, os.path.join(args.out_dir, "speed_accuracy_frontier_T100_v4.png"), old_pts=old_pts)
    plot_comp_cost_frontier(frontier, os.path.join(args.out_dir, "comp_cost_frontier_T100_v4.png"))

    comparison_rows = make_comparison_rows(old_pts, frontier)
    plot_old_vs_new_bars(comparison_rows, os.path.join(args.out_dir, "old_runD_vs_v4_timeavg_rms.png"))

    # Write main summary.
    summary_path = os.path.join(args.out_dir, "perf_summary_T100_v4_zone3.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"T_years: {T}\n")
        f.write(f"n_samples: {ns}\n")
        f.write("method: Revised Zone 2/Zone 3\n")
        f.write("zone1: r < 4e-4 AU -> direct Newtonian hard fallback\n")
        f.write("zone2: 4e-4 AU <= r < 0.05 AU -> direct Newtonian + adaptive sub-steps, no NN\n")
        f.write("zone3: 0.05 AU <= r < 0.15 AU -> bounded v4 trajectory NN\n")
        f.write("zone4: r >= 0.15 AU -> direct Newtonian\n\n")
        f.write(f"Model: {mp}\n\n")
        f.write("Baseline (REBOUND IAS15):\n")
        f.write(f"  total_time_sec: {perf_r['total_time_sec']:.6f}\n\n")
        f.write(f"Representative v4 run (dt_rep={dr}):\n")
        f.write(f"  final_err: {final_err:.6e}\n")
        f.write(f"  time_avg_rms: {time_avg_rms:.6e}\n")
        f.write(f"  steps: {perf_m['steps']}\n")
        f.write(f"  total_time_sec: {perf_m['total_time_sec']:.6f}\n")
        f.write(f"  time_per_step_sec: {perf_m['time_per_step_sec']:.6e}\n")
        f.write(f"  divergence_slope_1_per_yr: {slope:.6e}\n")
        f.write(f"  divergence_fit_window_years: {win}\n")
        f.write(f"  speedup_vs_ias15: {sp:.2f}x\n")
        f.write(f"  zone2_frac: {perf_m['zone2_frac']:.6f}\n")
        f.write(f"  zone3_nn_frac: {perf_m['zone3_nn_frac']:.6f}\n")
        f.write(f"  zone3_gate_fallback_frac: {perf_m['zone3_gate_fallback_frac']:.6f}\n")
        f.write(f"  strong_approach_zone3_frac: {perf_m['strong_approach_zone3_frac']:.6f}\n")
        f.write(f"  c_pred_min/mean/max: {perf_m['c_pred_min']:.6f} / {perf_m['c_pred_mean']:.6f} / {perf_m['c_pred_max']:.6f}\n\n")
        if no_nn_rep is not None:
            f.write(f"Representative revised no-NN Zone 3 baseline (dt_rep={dr}):\n")
            f.write(f"  final_err: {no_nn_rep['final_err']:.6e}\n")
            f.write(f"  time_avg_rms: {no_nn_rep['time_avg_rms']:.6e}\n")
            imp = 100.0 * (no_nn_rep["time_avg_rms"] - time_avg_rms) / max(abs(no_nn_rep["time_avg_rms"]), 1e-30)
            f.write(f"  v4_time_avg_rms_improvement_vs_revised_no_nn: {imp:.2f}%\n\n")
        f.write("Frontier sweep (v4 Zone 3 NN):\n")
        f.write("dt\tfinal_err\ttime_avg_rms\tthroughput(samp/s)\ttime_per_step(s)\ttotal_time(s)\tsteps\tzone2_frac\tzone3_nn_frac\tzone3_gate_frac\tstrong_app_zone3_frac\tc_min\tc_mean\tc_max\tspeedup\n")
        for p in frontier:
            f.write(f"{p['dt']}\t{p['final_err']:.6e}\t{p['time_avg_rms']:.6e}\t"
                    f"{p['throughput_samp_per_sec']:.3f}\t{p['time_per_step_sec']:.6e}\t"
                    f"{p['total_time_sec']:.6f}\t{p['steps']}\t{p['zone2_frac']:.6f}\t"
                    f"{p['zone3_nn_frac']:.6f}\t{p['zone3_gate_fallback_frac']:.6f}\t"
                    f"{p['strong_approach_zone3_frac']:.6f}\t{p['c_pred_min']:.6f}\t"
                    f"{p['c_pred_mean']:.6f}\t{p['c_pred_max']:.6f}\t{p['speedup']:.2f}x\n")
        f.write("\nNotes:\n")
        f.write("- final_err is RMS(model - IAS15) at T=100 years.\n")
        f.write("- time_avg_rms is sqrt(mean_t RMS(model - IAS15)^2) over 0-T.\n")
        f.write("- zone2_frac is the pair-evaluation fraction in 4e-4 <= r < 0.05 AU; force is direct Newtonian.\n")
        f.write("- zone3_nn_frac is the pair-evaluation fraction in 0.05 <= r < 0.15 AU where v4 is eligible.\n")
        f.write("- zone3_gate_frac is the fraction of Zone 3 NN calls rejected by hard safety gate.\n")

    # Write comparison summary if old data were available.
    comp_path = os.path.join(args.out_dir, "comparison_previous_RunD_vs_v4.txt")
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write("Comparison: previous Run D summary vs revised v4 Zone 3 evaluator\n")
        f.write(f"Previous summary file: {args.old_summary if os.path.exists(args.old_summary) else 'NOT FOUND'}\n\n")
        if not comparison_rows:
            f.write("No matching previous rows found. Place perf_summary_T100_timeavg.txt in this folder or pass --old_summary.\n")
        else:
            f.write("dt\told_time_avg_rms\tv4_time_avg_rms\tchange_%\told_final_err\tv4_final_err\tchange_%\told_speedup\tv4_speedup\n")
            for r in comparison_rows:
                f.write(f"{r['dt']}\t{r['old_time_avg_rms']:.6e}\t{r['new_time_avg_rms']:.6e}\t"
                        f"{r['time_avg_change_pct']:+.2f}%\t{r['old_final_err']:.6e}\t"
                        f"{r['new_final_err']:.6e}\t{r['final_err_change_pct']:+.2f}%\t"
                        f"{r.get('old_speedup', np.nan)}\t{r.get('new_speedup', np.nan):.2f}x\n")
            f.write("\nNegative change means v4 improved the metric. Positive change means worse.\n")

    print(f"[eval-v4] wrote {args.out_dir}")
    print(f"[eval-v4] main summary: {summary_path}")
    print(f"[eval-v4] comparison summary: {comp_path}")
    print("[eval-v4] done.")


if __name__ == "__main__":
    main()
