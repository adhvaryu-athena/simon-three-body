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
#   v4_zone3_frozen_eval_out/
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
    def __init__(self, hidden: int = 128, c_min: float = 0.25, c_max: float = 3.0):
        super().__init__()
        if not (0.0 < float(c_min) < float(c_max)):
            raise ValueError(f"Invalid bounds: c_min={c_min}, c_max={c_max}")
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.skip = nn.Linear(6, 1, bias=True)
        self.register_buffer("input_mean", torch.zeros(6))
        self.register_buffer("input_std", torch.ones(6))
        self.register_buffer("log_c_min", torch.tensor(float(np.log(c_min)), dtype=torch.float32))
        self.register_buffer("log_c_max", torch.tensor(float(np.log(c_max)), dtype=torch.float32))

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.net(x_norm).squeeze(-1) + self.skip(x_norm).squeeze(-1)

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
        "skip_wT": sd["skip.weight"].cpu().numpy().T.astype(np.float32).copy(),
        "skip_b": sd["skip.bias"].cpu().numpy().astype(np.float32),
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
    """Return bounded c for rows of 6-input v5 features (main path + residual skip)."""
    h = (nn_in - w["mean"]) / w["std"]
    main = _silu_np(h @ w["w0T"] + w["b0"])
    main = _silu_np(main @ w["w1T"] + w["b1"])
    main = _silu_np(main @ w["w2T"] + w["b2"])
    main_out = (main @ w["w3T"] + w["b3"]).ravel()
    skip_out = (h @ w["skip_wT"] + w["skip_b"]).ravel()
    raw = (main_out + skip_out).astype(np.float64)
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
    label: str = "v4_zone3_nn_frozen",
    nn_log_rows=None,
):
    """
    OPTIMIZED evaluator with FROZEN macro-step Zone-3 correction.

    Methodology (unchanged from non-optimized version):
      1. Zone gating by minimum pairwise distance.
      2. Frozen-c per macro/sub-step: at the start of each step we compute
         a single c value for every pair currently in Zone 3 and freeze it
         for both leapfrog half-kicks of that step. This matches the Zone-3
         training target (one c_opt per leapfrog step).
      3. Adaptive sub-stepping triggers when r_min < adapt_thresh.

    Zone logic:
      Zone 1: r < 4e-4 AU       -> direct Newtonian hard fallback
      Zone 2: 4e-4 <= r < 0.05  -> direct Newtonian + adaptive sub-stepping, no NN
      Zone 3: 0.05 <= r < 0.15  -> bounded v4 trajectory NN frozen per step
      Zone 4: r >= 0.15         -> direct Newtonian

    If use_zone3_nn=False, Zone 3 uses c=1 times F_soft for a same-method
    no-NN baseline. Zone 2 remains direct Newtonian in both cases.

    Performance optimizations (do not change physics):
      (a) Leapfrog acceleration `a1` from a Zone-4-only step is cached and
          reused as `a0` for the next step *iff* the next step is also
          Zone-4-only. The cache is invalidated by any frozen-step or
          substep path because those paths apply state-dependent NN forces
          and the cached pure-Newtonian acceleration would be incorrect.
      (b) Each step path returns r_min at its terminal x, removing the need
          for a separate min_pair_dist() pass at the top of each step.
      (c) Inside verlet_frozen_step, pair geometry (rij, r2, r) computed at
          x_in is shared between make_frozen_zone3_state and the first
          compute_acc call. The second compute_acc call (at x_new) still
          recomputes geometry because x_new is a new state.
      (d) Stats counters in compute_acc are numpy int64 accumulators
          instead of Python ints inside a dict; converted to a final dict
          at the end.

    Correctness contract for the cache:
      `a_cache` is the acceleration vector evaluated at the CURRENT x using
      *pure Newtonian* forces. It is valid only after a Zone-4-only step
      because Zone-4-only steps use compute_acc_far (no NN, no softening).
      Any path that may apply NN or softened forces (verlet_frozen_step,
      substepping) invalidates a_cache to None. The next Zone-4 step then
      recomputes a0 from scratch and re-arms the cache.

      Stats are kept identical to the non-optimized version: when a cache
      hit lets us skip a compute_acc_far call, we still increment
      pair_evals / far_pairs to record the "as if" pair work, so that the
      reported zone fractions match the original simulator output.
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

    # --- numpy int64 counters (item 2) -------------------------------------
    pair_evals_n = np.int64(0)
    zone1_n = np.int64(0)
    zone2_n = np.int64(0)
    zone3_n = np.int64(0)
    far_n = np.int64(0)
    zone3_nn_n = np.int64(0)
    zone3_gate_fb_n = np.int64(0)
    zone3_no_nn_n = np.int64(0)
    strong_z3_n = np.int64(0)
    frozen_steps_with_z3_n = 0
    c_sum = 0.0
    c_count = 0
    c_lo = np.inf
    c_hi = -np.inf

    # Step-state trackers for optional CSV logging (only used if
    # nn_log_rows is not None). These are updated by the main loop
    # before each step path and read inside make_frozen_zone3_state_geom.
    _log_step_idx = 0          # macro step index 0..n_steps-1
    _log_substep_idx = 0       # 0 for normal macro step; 1..n_sub during substep loop
    _log_t_yr = 0.0            # t_cur at start of this (sub)step
    _log_step_dt = float(dt)   # dt_f for macro step or sub_dt during substep loop


    def _compute_pair_geometry(pos):
        """Pair displacements, squared distances, distances. One geometry pass."""
        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        r = np.sqrt(r2 + 1e-30)
        return rij, r2, r

    def _zone3_features_from_geom(rij, r2, r, vel, step_dt, zmask):
        """Build v4 NN features for Zone-3 pairs using pre-computed geometry."""
        n_z = int(np.sum(zmask))
        r_soft = np.sqrt(r2[zmask] + eps2)
        rij_z = rij[zmask]
        r_z = r[zmask]
        r_hat = rij_z / (r_z[:, None] + 1e-30)
        vij_z = (vel[jj] - vel[ii])[zmask]
        v_rad = np.einsum("ij,ij->i", vij_z, r_hat)
        v_tan_vec = vij_z - v_rad[:, None] * r_hat
        v_tan = np.sqrt(np.einsum("ij,ij->i", v_tan_vec, v_tan_vec) + 1e-30)
        v_scale = np.sqrt(G * (mi_arr[zmask] + mj_arr[zmask]) / (r_z + 1e-30))
        v_rad_norm = v_rad / (v_scale + 1e-30)
        v_tan_norm = v_tan / (v_scale + 1e-30)

        nn_in = np.empty((n_z, 6), dtype=np.float32)
        nn_in[:, 0] = np.log(r_soft + 1e-30).astype(np.float32)
        nn_in[:, 1] = log_mi[zmask]
        nn_in[:, 2] = log_mj[zmask]
        nn_in[:, 3] = np.float32(np.log(float(step_dt) + 1e-30))
        nn_in[:, 4] = v_rad_norm.astype(np.float32)
        nn_in[:, 5] = v_tan_norm.astype(np.float32)
        return nn_in, v_rad_norm

    def make_frozen_zone3_state_geom(rij, r2, r, vel, step_dt):
        """Compute one frozen c per pair for the coming step, using shared geometry."""
        nonlocal frozen_steps_with_z3_n, c_sum, c_count, c_lo, c_hi

        frozen_c = np.ones(P, dtype=np.float64)
        eligible = np.zeros(P, dtype=bool)
        fallback = np.zeros(P, dtype=bool)
        strong = np.zeros(P, dtype=bool)

        z0 = (r >= cfg.adapt_thresh) & (r < cfg.nn_thresh)
        if not np.any(z0):
            return frozen_c, eligible, fallback, strong

        if use_zone3_nn:
            nn_in, v_rad_norm = _zone3_features_from_geom(rij, r2, r, vel, step_dt, z0)
            c = v4_forward_numpy(nn_in, w)
            r_soft = np.sqrt(r2[z0] + eps2)
            fb = ((r_soft < cfg.r_soft_min) | (c < cfg.c_min) |
                  (c > cfg.c_max) | ~np.isfinite(c))

            frozen_c[z0] = c
            eligible[z0] = True
            fallback[z0] = fb
            strong[z0] = v_rad_norm < -0.6

            frozen_steps_with_z3_n += 1
            good_c = c[np.isfinite(c)]
            if len(good_c):
                c_sum += float(np.sum(good_c))
                c_count += int(len(good_c))
                cmin_local = float(np.min(good_c))
                cmax_local = float(np.max(good_c))
                if cmin_local < c_lo:
                    c_lo = cmin_local
                if cmax_local > c_hi:
                    c_hi = cmax_local

            # ---- Optional per-firing logger (no-op when nn_log_rows is None) ----
            # Records one row per Zone-3 pair that received an NN c on this step.
            # The c value is the ALREADY-COMPUTED prediction; the safety gate
            # decision is recorded separately so we can see which firings were
            # actually applied vs replaced by the analytic fallback.
            if nn_log_rows is not None:
                # Indices of pairs that are in Zone 3 this step
                z3_pair_idx = np.where(z0)[0]
                # Geometry + features (one entry per Zone 3 pair)
                r_z3 = r[z0]
                # Tangential velocity feature (we already have v_rad_norm)
                # _zone3_features_from_geom returned only v_rad_norm; recompute
                # v_tan_norm here from the geometry we already have.
                rij_z3 = rij[z0]
                r_hat_z3 = rij_z3 / (r_z3[:, None] + 1e-30)
                vij_z3 = (vel[jj] - vel[ii])[z0]
                v_rad_full = np.einsum("ij,ij->i", vij_z3, r_hat_z3)
                v_tan_vec = vij_z3 - v_rad_full[:, None] * r_hat_z3
                v_tan_mag = np.sqrt(np.einsum("ij,ij->i", v_tan_vec, v_tan_vec) + 1e-30)
                v_scale = np.sqrt(cfg.G * (mi_arr[z0] + mj_arr[z0]) / (r_z3 + 1e-30))
                v_tan_norm = v_tan_mag / (v_scale + 1e-30)
                # Analytic softening c for reference: c_anal = (r_soft/r)^3
                r_soft_z3 = np.sqrt(r2[z0] + eps2)
                c_anal_ref = (r_soft_z3 / (r_z3 + 1e-30)) ** 3
                n_z3_this_step = int(len(z3_pair_idx))
                for k, p_idx in enumerate(z3_pair_idx):
                    nn_log_rows.append({
                        "step_idx": _log_step_idx,
                        "substep_idx": _log_substep_idx,
                        "t_yr": float(_log_t_yr),
                        "dt_step": float(_log_step_dt),
                        "pair_i": int(ii[p_idx]),
                        "pair_j": int(jj[p_idx]),
                        "r_au": float(r_z3[k]),
                        "v_rad_norm": float(v_rad_norm[k]),
                        "v_tan_norm": float(v_tan_norm[k]),
                        "m_i": float(mi_arr[p_idx]),
                        "m_j": float(mj_arr[p_idx]),
                        "c_pred": float(c[k]) if np.isfinite(c[k]) else float("nan"),
                        "c_anal_softening": float(c_anal_ref[k]),
                        "gate_fallback": bool(fb[k]),
                        "strong_approach": bool(v_rad_norm[k] < -0.6),
                        "n_zone3_pairs_this_step": n_z3_this_step,
                    })
        else:
            eligible[z0] = True

        return frozen_c, eligible, fallback, strong

    def compute_acc_geom(rij, r2, r, frozen_c, eligible, fallback, strong):
        """Acceleration from shared geometry. Stats use numpy counters."""
        nonlocal pair_evals_n, zone1_n, zone2_n, zone3_n, far_n
        nonlocal zone3_nn_n, zone3_gate_fb_n, zone3_no_nn_n, strong_z3_n

        F_scalar = Gmimj / (r2 * r + 1e-30)

        zone1_mask = r < cfg.zone1_r_gate
        zone2_mask = (r >= cfg.zone1_r_gate) & (r < cfg.adapt_thresh)
        zone3_mask = (r >= cfg.adapt_thresh) & (r < cfg.nn_thresh)
        far_mask = r >= cfg.nn_thresh

        pair_evals_n += P
        zone1_n += zone1_mask.sum()
        zone2_n += zone2_mask.sum()
        zone3_n += zone3_mask.sum()
        far_n += far_mask.sum()

        apply_mask = eligible & zone3_mask
        if np.any(apply_mask):
            F_soft = Gmimj[apply_mask] / ((r2[apply_mask] + eps2) ** 1.5 + 1e-30)
            if use_zone3_nn:
                fb = fallback[apply_mask]
                c = frozen_c[apply_mask]
                F_scalar[apply_mask] = np.where(fb, F_scalar[apply_mask], c * F_soft)
                am_count = int(apply_mask.sum())
                zone3_nn_n += am_count
                zone3_gate_fb_n += int(fb.sum())
                strong_z3_n += int(strong[apply_mask].sum())
            else:
                F_scalar[apply_mask] = F_soft
                zone3_no_nn_n += int(apply_mask.sum())

        F_vec = F_scalar[:, None] * rij
        acc = np.zeros((N, 3), dtype=np.float64)
        for p in range(P):
            acc[ii[p]] += F_vec[p] * inv_mi[p]
            acc[jj[p]] -= F_vec[p] * inv_mj[p]
        return acc

    def compute_acc_far(pos):
        """Fast path: r_min >= nn_thresh so all pairs are Zone 4.
        Returns (acc, r_min, r2_array) so caller can extract r_min without
        a second geometry pass."""
        nonlocal pair_evals_n, far_n
        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        r = np.sqrt(r2 + 1e-30)
        F_scalar = Gmimj / (r2 * r + 1e-30)

        pair_evals_n += P
        far_n += P

        F_vec = F_scalar[:, None] * rij
        acc = np.zeros((N, 3), dtype=np.float64)
        for p in range(P):
            acc[ii[p]] += F_vec[p] * inv_mi[p]
            acc[jj[p]] -= F_vec[p] * inv_mj[p]
        return acc, float(np.sqrt(r2.min() + 1e-30))

    def verlet_frozen_step(x_in, v_in, step_dt):
        """Frozen-c leapfrog step. Returns (x_new, v_new, r_min_new).

        Geometry at x_in is computed ONCE and shared between the frozen-state
        computation and the first force evaluation (item 3).
        Geometry at x_new is computed inside the second compute_acc_geom call
        because x_new is fresh; r_min_new is read from that geometry (item 4).
        """
        rij_in, r2_in, r_in = _compute_pair_geometry(x_in)
        frozen_c, eligible, fallback, strong = make_frozen_zone3_state_geom(
            rij_in, r2_in, r_in, v_in, step_dt
        )
        a0 = compute_acc_geom(rij_in, r2_in, r_in,
                              frozen_c, eligible, fallback, strong)
        vh = v_in + 0.5 * step_dt * a0
        x_new = x_in + step_dt * vh
        rij_n, r2_n, r_n = _compute_pair_geometry(x_new)
        a1 = compute_acc_geom(rij_n, r2_n, r_n,
                              frozen_c, eligible, fallback, strong)
        v_new = vh + 0.5 * step_dt * a1
        r_min_new = float(np.sqrt(r2_n.min() + 1e-30))
        return x_new, v_new, r_min_new

    def verlet_no_zone3(x_in, v_in, step_dt, a_in=None):
        """Zone-4-only leapfrog step with optional cached a_in (item 1).
        Returns (x_new, v_new, a_new, r_min_new).
        If a_in is supplied, the first compute_acc_far call is skipped, but
        the equivalent stat counters are still incremented so that pair_evals
        / far_pairs match the non-cached version (preserves report semantics).
        """
        nonlocal pair_evals_n, far_n
        if a_in is not None:
            a0 = a_in
            # Account for the call we skipped, so stats stay comparable.
            pair_evals_n += P
            far_n += P
        else:
            a0, _ = compute_acc_far(x_in)
        vh = v_in + 0.5 * step_dt * a0
        x_new = x_in + step_dt * vh
        a1, r_min_new = compute_acc_far(x_new)
        v_new = vh + 0.5 * step_dt * a1
        return x_new, v_new, a1, r_min_new

    # ------------------------------------------------------------------
    # Initial state capture (sample at t=0 if requested)
    # ------------------------------------------------------------------
    si = 0
    nt = times[0]
    t_cur = 0.0
    while si < n_samples and t_cur >= nt - 1e-12:
        pos_out[si] = x
        vel_out[si] = v
        si += 1
        if si < n_samples:
            nt = times[si]

    # ------------------------------------------------------------------
    # Initial r_min computation (one-time startup cost). After this,
    # every step path returns the r_min at its terminal x, so we never
    # need a dedicated min-pair-distance pass again.
    # ------------------------------------------------------------------
    rij0, r2_0, r0 = _compute_pair_geometry(x)
    r_min_cur = float(np.sqrt(r2_0.min() + 1e-30))
    a_cache = None  # invalid until a Zone-4-only step populates it

    steps = 0
    dt_f = float(dt)
    total_substeps = 0
    t_start = time.perf_counter()
    for _ in range(n_steps):
        # Update step-state trackers for the logger BEFORE taking the step.
        # These reflect the step we are about to take.
        _log_step_idx = steps
        _log_t_yr = t_cur
        _log_step_dt = dt_f
        _log_substep_idx = 0

        if r_min_cur < cfg.adapt_thresh:
            # Zone 2 active: adaptive sub-stepping. Zone 3 NN may activate
            # during sub-steps if other pairs are in Zone 3, so use
            # verlet_frozen_step. Cache is invalidated.
            n_sub = min(cfg.max_substeps, max(2, int(np.ceil(cfg.adapt_thresh / r_min_cur))))
            sub_dt = dt_f / n_sub
            _log_step_dt = sub_dt  # NN sees sub_dt as its log_dt input
            sub_t = t_cur
            for sub_k in range(n_sub):
                _log_substep_idx = sub_k + 1   # 1-indexed
                _log_t_yr = sub_t
                x, v, r_min_cur = verlet_frozen_step(x, v, sub_dt)
                sub_t += sub_dt
            a_cache = None
            total_substeps += n_sub
        elif r_min_cur < cfg.nn_thresh:
            # Zone 3 range: at least one pair in [0.05, 0.15). Compute frozen-c
            # via make_frozen_zone3_state_geom and apply for both half-kicks.
            # Cache is invalidated because the final a1 may contain NN force.
            x, v, r_min_cur = verlet_frozen_step(x, v, dt_f)
            a_cache = None
            total_substeps += 1
        else:
            # Zone 4 only: pure Newtonian. Use a_cache if valid (set at end
            # of previous Zone-4-only step). Update a_cache with a1 for the
            # next step.
            x, v, a_cache, r_min_cur = verlet_no_zone3(x, v, dt_f, a_in=a_cache)
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

    # ------------------------------------------------------------------
    # Finalize stats: convert numpy counters to ints, build perf dict
    # in the exact shape the original returned.
    # ------------------------------------------------------------------
    pair_evals = max(int(pair_evals_n), 1)
    c_count_safe = max(int(c_count), 1)
    c_min_out = float(c_lo) if np.isfinite(c_lo) else float("nan")
    c_max_out = float(c_hi) if np.isfinite(c_hi) else float("nan")

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
        "zone1_frac": int(zone1_n) / pair_evals,
        "zone2_frac": int(zone2_n) / pair_evals,
        "zone3_frac": int(zone3_n) / pair_evals,
        "zone4_far_frac": int(far_n) / pair_evals,
        "zone3_nn_frac": int(zone3_nn_n) / pair_evals,
        "zone3_gate_fallback_frac": int(zone3_gate_fb_n) / max(int(zone3_nn_n), 1),
        "strong_approach_zone3_frac": int(strong_z3_n) / max(int(zone3_nn_n), 1),
        "zone3_no_nn_frac": int(zone3_no_nn_n) / pair_evals,
        "frozen_steps_with_zone3": int(frozen_steps_with_z3_n),
        "c_pred_mean": c_sum / c_count_safe,
        "c_pred_min": c_min_out,
        "c_pred_max": c_max_out,
    }
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
        ax.semilogy(times, delta_v4, label="v4 Zone 3 frozen-c NN")
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
        ax.scatter(xs, ys, s=42, zorder=3, label="v4 Zone 3 frozen-c NN")
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


def resolve_old_summary_path(path: str) -> str:
    """Resolve old summary robustly from cwd, script folder, or output folder."""
    if not path:
        return path
    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(os.path.join(os.getcwd(), path))
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return path



# =============================================================================
# Ensemble-perturbation main  --  Lever 2 implementation
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Lever 2 ensemble evaluator: runs the full frontier sweep "
                    "N times with slightly perturbed ICs to measure whether the "
                    "NN systematically helps or hurts across chaotic branches."
    )
    parser.add_argument("--model_path",    default="pair_correction_nn_v5.pt")
    parser.add_argument("--out_dir",       default="v4_zone3_frozen_eval_perturb_out")
    parser.add_argument("--dt_rep",        type=float, default=0.06)
    parser.add_argument("--T",             type=float, default=100.0)
    parser.add_argument("--n_samples",     type=int,   default=5000)
    parser.add_argument("--dts",           default="0.02,0.04,0.05,0.06,0.08,0.1")
    parser.add_argument("--n_seeds",       type=int,   default=10,
                        help="Number of IC perturbation seeds. Each seed shifts "
                             "initial positions by ±perturb_scale AU.")
    parser.add_argument("--perturb_scale", type=float, default=1e-8,
                        help="Half-width of uniform position perturbation (AU). "
                             "Default 1e-8 AU is well below numerical precision "
                             "of any real observation while exciting different "
                             "chaotic branches.")
    parser.add_argument("--ic",            default="IC1",
                        choices=["IC1", "IC3", "IC4", "IC_custom"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── load model ─────────────────────────────────────────────────────────
    mp = args.model_path
    if not mp.endswith(".pt") and os.path.exists(mp + ".pt"):
        mp += ".pt"
    model = load_v4_model(mp)

    T   = float(args.T)
    ns  = int(args.n_samples)
    dts = [float(x.strip()) for x in args.dts.split(",") if x.strip()]
    cfg = HybridConfig()

    # ── base initial conditions ────────────────────────────────────────────
    IC_DEFS = {
        "IC1": {
            "m":  np.array([1.0, 0.01, 0.005], dtype=np.float64),
            "x0": np.array([[0,0,0],[1,0,0],[0,1.2,0]], dtype=np.float64),
            "v0": np.array([[0,0,0],[0,1,0],[-0.9,0,0]], dtype=np.float64),
        },
        "IC3": {
            "m":  np.array([1.0, 0.01, 0.005], dtype=np.float64),
            "x0": np.array([[0,0,0],[0.5,0,0],[0,2.5,0]], dtype=np.float64),
            "v0": np.array([[0,0,0],[0,1.3,0],[-0.4,0,0]], dtype=np.float64),
        },
        "IC4": {
            "m":  np.array([1.0, 0.01, 0.005], dtype=np.float64),
            "x0": np.array([[0,0,0],[1,0,0],[0,5.0,0]], dtype=np.float64),
            "v0": np.array([[0,0,0],[0,1.0,0],[-0.12,0,0]], dtype=np.float64),
        },
        "IC_custom": {
            "m":  np.array([1.0, 0.05, 0.005], dtype=np.float64),
            "x0": np.array([[0,0,0],[0.09,0,0],[0,2.0,0]], dtype=np.float64),
            "v0": np.array([[0,0,0],[0,4.11,0],[-0.65,0,0]], dtype=np.float64),
        },
    }
    ic_def   = IC_DEFS[args.ic]
    m_base   = ic_def["m"]
    x0_base  = ic_def["x0"].copy().astype(np.float64)
    v0_base  = ic_def["v0"].copy().astype(np.float64)
    M_base   = m_base.sum()
    x0_base -= (m_base[:, None] * x0_base).sum(0) / M_base
    v0_base -= (m_base[:, None] * v0_base).sum(0) / M_base
    print(f"[perturb-eval] IC={args.ic}  T={T}  n_seeds={args.n_seeds}  "
          f"perturb_scale={args.perturb_scale:.1e}  dts={dts}")

    # ── storage: results_by_seed[seed_idx][dt] = dict ────────────────────
    # Each dict contains: delta_pct, tar_nn, tar_no_nn, firings, speedup
    results_by_seed = {}

    for seed_idx in range(args.n_seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed_idx}/{args.n_seeds-1}  "
              f"(perturb ±{args.perturb_scale:.1e} AU)")
        print(f"{'='*60}")

        # Perturb positions and re-centre COM
        rng    = np.random.RandomState(seed_idx)
        dx     = rng.uniform(-args.perturb_scale, args.perturb_scale,
                             size=x0_base.shape)
        x0_s   = x0_base + dx
        v0_s   = v0_base.copy()
        M_s    = m_base.sum()
        # Re-centre after perturbation (perturbation shifts COM slightly)
        x0_s  -= (m_base[:, None] * x0_s).sum(0) / M_s
        v0_s  -= (m_base[:, None] * v0_s).sum(0) / M_s
        m_s    = m_base.copy()

        # IAS15 reference on perturbed IC
        print(f"  [seed {seed_idx}] Running IAS15 reference ...", flush=True)
        _, pr_s, _, perf_r_s = simulate_rebound_ias15(x0_s, v0_s, m_s,
                                                       cfg.G, T, ns)
        ias15_time = perf_r_s["total_time_sec"]
        print(f"  [seed {seed_idx}] IAS15 done in {ias15_time:.3f}s")

        seed_results = {}

        for dt_val in dts:
            # -- NN run --
            _, pd_nn, _, pf_nn = simulate_leapfrog_v4_zone3(
                x0_s, v0_s, m_s, model, cfg, dt_val, T, ns,
                use_zone3_nn=True, label="nn"
            )
            rms_nn  = rms_sep(pd_nn, pr_s)
            tar_nn  = float(np.sqrt(np.mean(rms_nn**2)))
            speedup = ias15_time / max(pf_nn["total_time_sec"], 1e-12)

            # -- no-NN baseline on same perturbed IC --
            _, pd_no, _, pf_no = simulate_leapfrog_v4_zone3(
                x0_s, v0_s, m_s, model, cfg, dt_val, T, ns,
                use_zone3_nn=False, label="no_nn"
            )
            rms_no  = rms_sep(pd_no, pr_s)
            tar_no  = float(np.sqrt(np.mean(rms_no**2)))

            denom      = max(abs(tar_no), 1e-30)
            delta_pct  = 100.0 * (tar_nn - tar_no) / denom
            firings    = int(round(pf_nn["zone3_nn_frac"] *
                                   pf_nn["steps"] * 3))  # approx

            seed_results[dt_val] = {
                "delta_pct":  delta_pct,
                "tar_nn":     tar_nn,
                "tar_no_nn":  tar_no,
                "firings":    firings,
                "speedup":    speedup,
                "zone3_nn_frac": pf_nn["zone3_nn_frac"],
            }
            verdict = ("NN helps" if delta_pct < -0.5 else
                       "NN hurts" if delta_pct > +0.5 else "tie")
            print(f"  dt={dt_val:>5}  tar_nn={tar_nn:.4f}  "
                  f"tar_no={tar_no:.4f}  Δ={delta_pct:+.2f}%  "
                  f"[{verdict}]  firings~{firings}  speedup={speedup:.2f}x",
                  flush=True)

        results_by_seed[seed_idx] = seed_results

    # ── Ensemble statistics ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ENSEMBLE SUMMARY")
    print(f"{'='*60}")

    summary = {}  # dt -> stats dict
    for dt_val in dts:
        deltas   = [results_by_seed[s][dt_val]["delta_pct"]
                    for s in range(args.n_seeds)]
        tars_nn  = [results_by_seed[s][dt_val]["tar_nn"]
                    for s in range(args.n_seeds)]
        tars_no  = [results_by_seed[s][dt_val]["tar_no_nn"]
                    for s in range(args.n_seeds)]
        speedups = [results_by_seed[s][dt_val]["speedup"]
                    for s in range(args.n_seeds)]
        firings  = [results_by_seed[s][dt_val]["firings"]
                    for s in range(args.n_seeds)]

        arr      = np.array(deltas)
        n_helps  = int(np.sum(arr < -0.5))
        n_hurts  = int(np.sum(arr > +0.5))
        n_tie    = args.n_seeds - n_helps - n_hurts

        # Verdict: systematic help/hurt if >= 7/10 seeds agree and mean < ±1%
        mean_d   = float(np.mean(arr))
        std_d    = float(np.std(arr))
        if n_helps >= 7 and mean_d < -0.5:
            verdict = "SYSTEMATICALLY HELPS"
        elif n_hurts >= 7 and mean_d > +0.5:
            verdict = "SYSTEMATICALLY HURTS"
        elif abs(mean_d) < 0.5:
            verdict = "NEUTRAL / TIE"
        else:
            verdict = "INCONCLUSIVE"

        summary[dt_val] = {
            "mean_delta":   mean_d,
            "std_delta":    std_d,
            "median_delta": float(np.median(arr)),
            "min_delta":    float(np.min(arr)),
            "max_delta":    float(np.max(arr)),
            "n_helps":      n_helps,
            "n_hurts":      n_hurts,
            "n_tie":        n_tie,
            "mean_tar_nn":  float(np.mean(tars_nn)),
            "mean_tar_no":  float(np.mean(tars_no)),
            "mean_speedup": float(np.mean(speedups)),
            "mean_firings": float(np.mean(firings)),
            "verdict":      verdict,
            "all_deltas":   deltas,
        }
        print(f"  dt={dt_val:>5}  mean_Δ={mean_d:+.2f}%  "
              f"std={std_d:.2f}%  "
              f"[{n_helps} helps / {n_tie} tie / {n_hurts} hurts]  "
              f"-> {verdict}")

    # ── Write text report ─────────────────────────────────────────────────
    report_path = os.path.join(args.out_dir, "ensemble_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("LEVER 2 ENSEMBLE PERTURBATION REPORT\n")
        f.write("Zone 3 Scalar Force-Correction NN — Ungated\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"IC:              {args.ic}\n")
        f.write(f"Model:           {mp}\n")
        f.write(f"T (yr):          {T}\n")
        f.write(f"n_seeds:         {args.n_seeds}\n")
        f.write(f"perturb_scale:   {args.perturb_scale:.2e} AU\n")
        f.write(f"dts tested:      {dts}\n")
        f.write(f"n_samples:       {ns}\n\n")

        # Per-seed results table
        f.write("-" * 70 + "\n")
        f.write("PER-SEED DELTA (%) TABLE  [negative = NN helps]\n")
        f.write("-" * 70 + "\n")
        header = f"{'seed':>5}  " + "  ".join(f"dt={d:>4}" for d in dts) + "\n"
        f.write(header)
        f.write("-" * len(header.rstrip()) + "\n")
        for s in range(args.n_seeds):
            row_vals = "  ".join(
                f"{results_by_seed[s][d]['delta_pct']:+7.2f}%"
                for d in dts
            )
            f.write(f"{s:>5}  {row_vals}\n")
        f.write("\n")

        # Summary statistics table
        f.write("-" * 70 + "\n")
        f.write("ENSEMBLE SUMMARY TABLE\n")
        f.write("-" * 70 + "\n")
        cols = ["dt", "mean_Δ%", "std_Δ%", "median_Δ%",
                "min_Δ%", "max_Δ%", "helps", "tie", "hurts",
                "mean_tar_nn", "mean_tar_no", "mean_spdup", "verdict"]
        col_w = [6, 9, 7, 10, 8, 8, 6, 4, 6, 12, 12, 10, 25]
        f.write("  ".join(f"{c:>{w}}" for c, w in zip(cols, col_w)) + "\n")
        f.write("-" * 70 + "\n")
        for dt_val in dts:
            s = summary[dt_val]
            row = [
                f"{dt_val:>6}",
                f"{s['mean_delta']:>+9.2f}",
                f"{s['std_delta']:>7.2f}",
                f"{s['median_delta']:>+10.2f}",
                f"{s['min_delta']:>+8.2f}",
                f"{s['max_delta']:>+8.2f}",
                f"{s['n_helps']:>6}",
                f"{s['n_tie']:>4}",
                f"{s['n_hurts']:>6}",
                f"{s['mean_tar_nn']:>12.5f}",
                f"{s['mean_tar_no']:>12.5f}",
                f"{s['mean_speedup']:>10.2f}",
                f"{s['verdict']:>25}",
            ]
            f.write("  ".join(row) + "\n")
        f.write("\n")

        # All delta values per dt
        f.write("-" * 70 + "\n")
        f.write("ALL DELTA VALUES PER DT (for distribution analysis)\n")
        f.write("-" * 70 + "\n")
        for dt_val in dts:
            vals = summary[dt_val]["all_deltas"]
            vals_str = ", ".join(f"{v:+.3f}" for v in vals)
            f.write(f"  dt={dt_val}: [{vals_str}]\n")
        f.write("\n")

        # Final verdict
        f.write("=" * 70 + "\n")
        f.write("FINAL VERDICTS PER DT\n")
        f.write("=" * 70 + "\n")
        for dt_val in dts:
            s = summary[dt_val]
            f.write(f"  dt={dt_val:>5}:  {s['verdict']:30}  "
                    f"mean={s['mean_delta']:+.2f}% ± {s['std_delta']:.2f}%  "
                    f"speedup={s['mean_speedup']:.2f}x\n")
        f.write("\n")

    print(f"\n[perturb-eval] Report written: {report_path}")

    # ── Summary figure ────────────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        dt_labels = [str(d) for d in dts]
        x_pos     = np.arange(len(dts))

        # Box plot data
        box_data = [summary[d]["all_deltas"] for d in dts]
        bp = ax.boxplot(box_data, positions=x_pos, widths=0.5,
                        patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))

        # Colour boxes by verdict
        for i, (patch, dt_val) in enumerate(zip(bp["boxes"], dts)):
            v = summary[dt_val]["verdict"]
            if "HELPS" in v:
                patch.set_facecolor("#bbf7d0")
            elif "HURTS" in v:
                patch.set_facecolor("#fecaca")
            elif "NEUTRAL" in v:
                patch.set_facecolor("#fef3c7")
            else:
                patch.set_facecolor("#e0e7ff")

        # Scatter individual seed points
        for i, dt_val in enumerate(dts):
            ys = summary[dt_val]["all_deltas"]
            xs = np.full(len(ys), i) + np.random.RandomState(99).uniform(
                -0.12, 0.12, len(ys))
            ax.scatter(xs, ys, s=22, zorder=4, color="navy", alpha=0.6)

        # Mean ± std markers
        means = [summary[d]["mean_delta"] for d in dts]
        stds  = [summary[d]["std_delta"]  for d in dts]
        ax.errorbar(x_pos, means, yerr=stds, fmt="D", color="red",
                    markersize=6, capsize=5, zorder=5, label="mean ± std")

        ax.axhline(0, color="gray", linestyle="--", lw=1.2, label="no change")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"dt={d}" for d in dts], fontsize=11)
        ax.set_ylabel("Δ NN vs no-NN (%)  [negative = NN helps]", fontsize=12)
        ax.set_title(
            f"Lever 2 Ensemble: {args.ic}, T={T} yr, "
            f"n={args.n_seeds} seeds, perturb={args.perturb_scale:.0e} AU",
            fontsize=12
        )
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=10)
        plt.tight_layout()
        fig_path = os.path.join(args.out_dir, "ensemble_delta_boxplot.png")
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"[perturb-eval] Figure written: {fig_path}")
    except Exception as e:
        print(f"[perturb-eval] Figure generation failed (non-fatal): {e}")

    print(f"[perturb-eval] All done. Send ensemble_report.txt for analysis.")


if __name__ == "__main__":
    main()
