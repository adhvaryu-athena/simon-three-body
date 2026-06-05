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
# Encounter-level residual NN model and SIMON-encounterNN rollout
# =============================================================================
class EncounterResidualMLP(nn.Module):
    """
    Final dt=0.08 encounter residual model.

    Input:  X_rel18 at encounter entry state.
    Output: 18D residual [dx_flat9, dv_flat9] in physical units.
    Deployment formula:
        corrected_exit = noNN_exit + predicted_residual
    """
    def __init__(self, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
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
        y_norm = self.forward_norm(x_raw)
        return y_norm * (self.target_std + 1e-8) + self.target_mean


def infer_encounter_hidden(sd: Dict[str, torch.Tensor]) -> int:
    if "net.0.weight" not in sd:
        raise KeyError("Encounter model is missing net.0.weight")
    w = sd["net.0.weight"]
    if w.ndim != 2 or int(w.shape[1]) != 18:
        raise ValueError(f"Expected encounter first layer shape (hidden,18), got {tuple(w.shape)}")
    if "target_mean" not in sd or "target_std" not in sd:
        raise KeyError("Encounter checkpoint is missing target_mean/target_std; use the velocity-safe residual model.")
    return int(w.shape[0])


def load_encounter_model(model_path: str, device: str = "cpu") -> EncounterResidualMLP:
    sd = torch.load(model_path, map_location="cpu")
    hidden = infer_encounter_hidden(sd)
    model = EncounterResidualMLP(hidden=hidden, dropout=0.0)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    print(f"[encounterNN] Loaded {model_path} | hidden={hidden} | params={sum(p.numel() for p in model.parameters())}")
    return model


def com_center(m: np.ndarray, x: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = float(np.sum(m))
    x = x - np.sum(m[:, None] * x, axis=0) / M
    v = v - np.sum(m[:, None] * v, axis=0) / M
    return x.astype(np.float64), v.astype(np.float64)


def max_radius_state(x: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(x, axis=1)))


def all_finite_state(*arrs) -> bool:
    return all(np.all(np.isfinite(a)) for a in arrs)


def pair_distances_3(x: np.ndarray) -> np.ndarray:
    return np.array([
        np.linalg.norm(x[1] - x[0]),
        np.linalg.norm(x[2] - x[0]),
        np.linalg.norm(x[2] - x[1]),
    ], dtype=np.float64)


def min_pair_distance_3(x: np.ndarray) -> float:
    return float(np.min(pair_distances_3(x)))


def total_energy_state(x: np.ndarray, v: np.ndarray, m: np.ndarray, G: float = 1.0) -> float:
    ke = 0.5 * float(np.sum(m[:, None] * v * v))
    pe = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            r = float(np.linalg.norm(x[j] - x[i]))
            pe -= G * float(m[i]) * float(m[j]) / (r + 1e-30)
    return float(ke + pe)


def compute_pair_velocity_features_12(x: np.ndarray, v: np.ndarray, m: np.ndarray, cfg: HybridConfig) -> Tuple[float, float, float]:
    # Final validated encounter model is for active pair 1-2 in IC1.
    i, j = 1, 2
    rij = x[j] - x[i]
    vij = v[j] - v[i]
    r = float(np.linalg.norm(rij))
    r_hat = rij / (r + 1e-30)
    v_rad = float(np.dot(vij, r_hat))
    v_tan_vec = vij - v_rad * r_hat
    v_tan = float(np.linalg.norm(v_tan_vec))
    v_scale = math.sqrt(cfg.G * (float(m[i]) + float(m[j])) / (r + 1e-30))
    return float(v_rad / (v_scale + 1e-30)), float(v_tan / (v_scale + 1e-30)), r


def make_X_rel18_pair12(x: np.ndarray, v: np.ndarray, m: np.ndarray, dt: float, cfg: HybridConfig) -> Tuple[np.ndarray, float, float, float]:
    # Same feature construction used for the final rollout-local residual model.
    i, j, k = 1, 2, 0
    mi, mj = float(m[i]), float(m[j])
    pair_com_x = (mi * x[i] + mj * x[j]) / (mi + mj)
    pair_com_v = (mi * v[i] + mj * v[j]) / (mi + mj)
    rij = x[j] - x[i]
    vij = v[j] - v[i]
    r3 = x[k] - pair_com_x
    v3 = v[k] - pair_com_v
    vr, vt, r = compute_pair_velocity_features_12(x, v, m, cfg)
    X = np.concatenate([
        np.log(m + 1e-30),
        rij, vij, r3, v3,
        np.array([math.log(float(dt) + 1e-30), vr, vt], dtype=np.float64),
    ]).astype(np.float32)
    if X.shape != (18,):
        raise RuntimeError(f"X_rel18 has bad shape {X.shape}")
    return X, vr, vt, r


def project_com_to_reference(x_corr: np.ndarray, v_corr: np.ndarray,
                             x_ref: np.ndarray, v_ref: np.ndarray,
                             m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = float(np.sum(m))
    com_x_corr = np.sum(m[:, None] * x_corr, axis=0) / M
    com_v_corr = np.sum(m[:, None] * v_corr, axis=0) / M
    com_x_ref = np.sum(m[:, None] * x_ref, axis=0) / M
    com_v_ref = np.sum(m[:, None] * v_ref, axis=0) / M
    x_out = x_corr - (com_x_corr - com_x_ref)
    v_out = v_corr - (com_v_corr - com_v_ref)
    return x_out, v_out


def predict_encounter_residual(model: EncounterResidualMLP, X: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray, float, float]:
    xb = torch.tensor(X.reshape(1, 18), dtype=torch.float32, device=device)
    with torch.no_grad():
        y = model(xb).detach().cpu().numpy().reshape(18).astype(np.float64)
    dx = y[:9].reshape(3, 3)
    dv = y[9:].reshape(3, 3)
    pred_pos_norm = float(np.sqrt(np.mean(np.sum(dx * dx, axis=1))))
    pred_vel_norm = float(np.sqrt(np.mean(np.sum(dv * dv, axis=1))))
    return dx, dv, pred_pos_norm, pred_vel_norm


def revised_no_nn_acc_simple(pos: np.ndarray, m: np.ndarray, cfg: HybridConfig) -> Tuple[np.ndarray, float, Dict[str, int]]:
    """NoNN force law matching the evaluator's no-NN Zone 2/3/4 definitions."""
    acc = np.zeros((3, 3), dtype=np.float64)
    min_r = float("inf")
    counts = {"zone1": 0, "zone2": 0, "zone3": 0, "zone4": 0}
    for a in range(3):
        for b in range(a + 1, 3):
            rij = pos[b] - pos[a]
            r2 = float(np.dot(rij, rij))
            r = math.sqrt(r2 + 1e-30)
            min_r = min(min_r, r)
            Gmimj = cfg.G * float(m[a]) * float(m[b])
            if r < cfg.zone1_r_gate:
                scalar = Gmimj / (r2 * r + 1e-30)
                counts["zone1"] += 1
            elif r < cfg.adapt_thresh:
                scalar = Gmimj / (r2 * r + 1e-30)
                counts["zone2"] += 1
            elif r < cfg.nn_thresh:
                scalar = Gmimj / ((r2 + cfg.eps * cfg.eps) ** 1.5 + 1e-30)
                counts["zone3"] += 1
            else:
                scalar = Gmimj / (r2 * r + 1e-30)
                counts["zone4"] += 1
            F = scalar * rij
            acc[a] += F / float(m[a])
            acc[b] -= F / float(m[b])
    return acc, min_r, counts


def no_nn_step_simple(x: np.ndarray, v: np.ndarray, m: np.ndarray, cfg: HybridConfig,
                      step_dt: float, a_in: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Dict[str, int], int]:
    x = x.astype(np.float64, copy=True)
    v = v.astype(np.float64, copy=True)
    if a_in is None:
        a, _, c0 = revised_no_nn_acc_simple(x, m, cfg)
    else:
        a = a_in.astype(np.float64, copy=True)
        c0 = {"zone1": 0, "zone2": 0, "zone3": 0, "zone4": 0}

    r_now = min_pair_distance_3(x)
    if r_now < cfg.adapt_thresh:
        n_sub = min(cfg.max_substeps, max(2, int(math.ceil(cfg.adapt_thresh / max(r_now, 1e-30)))))
    else:
        n_sub = 1
    sub_dt = float(step_dt) / n_sub
    counts = dict(c0)
    min_r_seen = r_now
    for _ in range(n_sub):
        vh = v + 0.5 * sub_dt * a
        x = x + sub_dt * vh
        a, rmin, cc = revised_no_nn_acc_simple(x, m, cfg)
        v = vh + 0.5 * sub_dt * a
        min_r_seen = min(min_r_seen, rmin)
        for key in counts:
            counts[key] += int(cc[key])
        if not all_finite_state(x, v):
            raise FloatingPointError("noNN step produced non-finite state")
    return x, v, a, float(min_r_seen), counts, int(n_sub)


def simulate_leapfrog_encounterNN(
    x0: np.ndarray,
    v0: np.ndarray,
    m: np.ndarray,
    encounter_model: EncounterResidualMLP,
    cfg: HybridConfig,
    dt: float,
    T: float,
    n_samples: int,
    window_years: float = 0.5,
    vr_thresh: float = -0.40,
    energy_gate: float = 0.20,
    max_radius_gate: float = 1e4,
    device: str = "cpu",
    com_project: bool = True,
    event_rows: Optional[List[Dict[str, object]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Optimized SIMON-encounterNN ablation path.

    This is deliberately built from the same optimized kernel strategy as
    simulate_leapfrog_v4_zone3(..., use_zone3_nn=False):
      * vectorized pair geometry for the 3 pairs,
      * cached Zone-4 pure-Newtonian acceleration across consecutive far-field steps,
      * no separate min-distance pass at the top of every step,
      * shared geometry for frozen Zone-3 noNN force evaluation,
      * numpy/int counters rather than dictionary counters in the hot force loop.

    The encounter residual module is applied only when the validated gate fires:
        active pair 1-2 in Zone 3 and v_rad_norm < vr_thresh.

    Deployment formula:
        1. advance the same SIMON-noNN local window to obtain F_S,new;
        2. predict residual R_hat = EncounterNN(entry_state);
        3. set corrected_exit = F_S,new + R_hat;
        4. resume the optimized noNN rollout from the corrected state.

    Important: this is still the residual-correction architecture, not a direct
    skip-ahead model. It removes the artificial v1 slowdown caused by using a
    separate simple Python stepper for the whole rollout.
    """
    # ------------------------------------------------------------------
    # Geometry/constants setup, matching simulate_leapfrog_v4_zone3.
    # ------------------------------------------------------------------
    N = x0.shape[0]
    ii, jj = [], []
    for i in range(N):
        for j in range(i + 1, N):
            ii.append(i); jj.append(j)
    ii = np.array(ii, dtype=np.int64)
    jj = np.array(jj, dtype=np.int64)
    P = len(ii)
    if N != 3 or P != 3:
        raise ValueError("This optimized encounterNN evaluator is written for the 3-body IC1 case.")

    G = float(cfg.G)
    eps2 = float(cfg.eps * cfg.eps)
    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    m_f = m.astype(np.float64)
    mi_arr = m_f[ii]
    mj_arr = m_f[jj]
    Gmimj = G * mi_arr * mj_arr
    inv_mi = 1.0 / mi_arr
    inv_mj = 1.0 / mj_arr

    times = np.linspace(0.0, float(T), int(n_samples))
    n_steps = int(math.ceil(float(T) / float(dt)))
    pos_out = np.zeros((len(times), N, 3), dtype=np.float64)
    vel_out = np.zeros_like(pos_out)

    # Pair index for body pair 1-2 in the ii/jj arrays.
    pair12_idx = None
    for p, (a, b) in enumerate(zip(ii.tolist(), jj.tolist())):
        if a == 1 and b == 2:
            pair12_idx = p
            break
    if pair12_idx is None:
        raise RuntimeError("Could not locate pair 1-2 in pair list.")

    # ------------------------------------------------------------------
    # Hot-loop counters.
    # ------------------------------------------------------------------
    pair_evals_n = np.int64(0)
    zone1_n = np.int64(0)
    zone2_n = np.int64(0)
    zone3_n = np.int64(0)
    far_n = np.int64(0)
    zone3_no_nn_n = np.int64(0)
    total_substeps = 0
    steps = 0

    gate_candidates = 0
    encounter_attempts = 0
    encounter_used = 0
    encounter_fallback = 0
    first_used_t = float("nan")
    last_used_t = float("nan")
    pred_pos_norm_sum = 0.0
    pred_vel_norm_sum = 0.0
    min_r_global = float("inf")
    max_radius_global = max_radius_state(x)

    # ------------------------------------------------------------------
    # Optimized noNN force machinery, matching original v4 noNN semantics.
    # ------------------------------------------------------------------
    def _compute_pair_geometry(pos: np.ndarray):
        rij = pos[jj] - pos[ii]
        r2 = np.einsum("ij,ij->i", rij, rij)
        r = np.sqrt(r2 + 1e-30)
        return rij, r2, r

    def make_frozen_noNN_state_geom(r: np.ndarray):
        # In noNN mode, Zone 3 is eligible and uses c=1 times F_soft.
        frozen_c = np.ones(P, dtype=np.float64)
        eligible = (r >= cfg.adapt_thresh) & (r < cfg.nn_thresh)
        fallback = np.zeros(P, dtype=bool)
        strong = np.zeros(P, dtype=bool)
        return frozen_c, eligible, fallback, strong

    def compute_acc_geom(rij, r2, r, frozen_c, eligible, fallback, strong):
        nonlocal pair_evals_n, zone1_n, zone2_n, zone3_n, far_n, zone3_no_nn_n

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
            F_scalar[apply_mask] = F_soft  # c=1 noNN Zone-3 baseline
            zone3_no_nn_n += int(apply_mask.sum())

        F_vec = F_scalar[:, None] * rij
        acc = np.zeros((N, 3), dtype=np.float64)
        for p in range(P):
            acc[ii[p]] += F_vec[p] * inv_mi[p]
            acc[jj[p]] -= F_vec[p] * inv_mj[p]
        return acc

    def compute_acc_far(pos: np.ndarray):
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

    def verlet_frozen_noNN_step(x_in: np.ndarray, v_in: np.ndarray, step_dt: float):
        rij_in, r2_in, r_in = _compute_pair_geometry(x_in)
        frozen_c, eligible, fallback, strong = make_frozen_noNN_state_geom(r_in)
        a0 = compute_acc_geom(rij_in, r2_in, r_in, frozen_c, eligible, fallback, strong)
        vh = v_in + 0.5 * step_dt * a0
        x_new = x_in + step_dt * vh
        rij_n, r2_n, r_n = _compute_pair_geometry(x_new)
        a1 = compute_acc_geom(rij_n, r2_n, r_n, frozen_c, eligible, fallback, strong)
        v_new = vh + 0.5 * step_dt * a1
        return x_new, v_new, float(np.sqrt(r2_n.min() + 1e-30))

    def verlet_far_step(x_in: np.ndarray, v_in: np.ndarray, step_dt: float, a_in: Optional[np.ndarray] = None):
        nonlocal pair_evals_n, far_n
        if a_in is not None:
            a0 = a_in
            # Preserve report semantics, as in the original optimized evaluator.
            pair_evals_n += P
            far_n += P
        else:
            a0, _ = compute_acc_far(x_in)
        vh = v_in + 0.5 * step_dt * a0
        x_new = x_in + step_dt * vh
        a1, r_min_new = compute_acc_far(x_new)
        v_new = vh + 0.5 * step_dt * a1
        return x_new, v_new, a1, r_min_new

    def noNN_optimized_step(x_in: np.ndarray, v_in: np.ndarray, step_dt: float,
                            r_min_in: float, a_cache_in: Optional[np.ndarray]):
        """One optimized SIMON-noNN macro/short step.
        Returns x, v, r_min_new, a_cache_new, n_substeps_used.
        """
        if r_min_in < cfg.adapt_thresh:
            n_sub = min(cfg.max_substeps, max(2, int(np.ceil(cfg.adapt_thresh / max(r_min_in, 1e-30)))))
            sub_dt = float(step_dt) / n_sub
            x_loc = x_in
            v_loc = v_in
            r_min_loc = r_min_in
            for _ in range(n_sub):
                x_loc, v_loc, r_min_loc = verlet_frozen_noNN_step(x_loc, v_loc, sub_dt)
            return x_loc, v_loc, r_min_loc, None, int(n_sub)
        if r_min_in < cfg.nn_thresh:
            x_new, v_new, r_min_new = verlet_frozen_noNN_step(x_in, v_in, float(step_dt))
            return x_new, v_new, r_min_new, None, 1
        x_new, v_new, a_new, r_min_new = verlet_far_step(x_in, v_in, float(step_dt), a_in=a_cache_in)
        return x_new, v_new, r_min_new, a_new, 1

    def pair12_features_from_state(x_now: np.ndarray, v_now: np.ndarray):
        rij = x_now[2] - x_now[1]
        vij = v_now[2] - v_now[1]
        r = float(np.linalg.norm(rij))
        r_hat = rij / (r + 1e-30)
        v_rad = float(np.dot(vij, r_hat))
        v_tan_vec = vij - v_rad * r_hat
        v_tan = float(np.linalg.norm(v_tan_vec))
        v_scale = math.sqrt(cfg.G * (float(m_f[1]) + float(m_f[2])) / (r + 1e-30))
        return float(v_rad / (v_scale + 1e-30)), float(v_tan / (v_scale + 1e-30)), r

    # ------------------------------------------------------------------
    # Output capture helpers.
    # ------------------------------------------------------------------
    si = 0
    t_cur = 0.0

    def fill_outputs_current():
        nonlocal si
        while si < len(times) and t_cur >= times[si] - 1e-12:
            pos_out[si] = x
            vel_out[si] = v
            si += 1

    fill_outputs_current()

    rij0, r2_0, r0 = _compute_pair_geometry(x)
    r_min_cur = float(np.sqrt(r2_0.min() + 1e-30))
    min_r_global = min(min_r_global, r_min_cur)
    a_cache = None

    t_start = time.perf_counter()
    while t_cur < float(T) - 1e-14 and steps < n_steps + 10:
        step_dt_macro = min(float(dt), float(T) - t_cur)

        # Gate check at the start of the macro step, before taking the noNN step.
        X_entry, vr_entry, vt_entry, r_pair = make_X_rel18_pair12(x, v, m_f, float(dt), cfg)
        in_zone3_pair12 = (cfg.adapt_thresh <= r_pair < cfg.nn_thresh)
        if in_zone3_pair12:
            gate_candidates += 1
        gate_ok = bool(
            in_zone3_pair12
            and (vr_entry < float(vr_thresh))
            and (t_cur + float(window_years) <= float(T) + 1e-12)
        )

        if gate_ok:
            encounter_attempts += 1
            t_event = float(t_cur)
            x_start = x.copy()
            v_start = v.copy()
            E_start = total_energy_state(x_start, v_start, m_f, G=cfg.G)
            min_r_window = r_min_cur
            exit_time = t_event + float(window_years)

            # Advance the required noNN window with the same optimized noNN kernel.
            # We do not use scalarNN here because the residual target is IAS15_exit - noNN_exit.
            remaining = float(window_years)
            while remaining > 1e-14:
                step_dt_w = min(float(dt), remaining)
                x, v, r_min_cur, a_cache, n_sub = noNN_optimized_step(x, v, step_dt_w, r_min_cur, a_cache)
                t_cur += step_dt_w
                remaining -= step_dt_w
                steps += 1
                total_substeps += int(n_sub)
                min_r_window = min(min_r_window, r_min_cur)
                min_r_global = min(min_r_global, r_min_cur)
                max_radius_global = max(max_radius_global, max_radius_state(x))
                fill_outputs_current()
                if not all_finite_state(x, v) or max_radius_global > float(max_radius_gate):
                    break

            x_no_exit = x.copy()
            v_no_exit = v.copy()

            dx, dv, pred_pos_norm, pred_vel_norm = predict_encounter_residual(encounter_model, X_entry, device)
            x_corr = x_no_exit + dx
            v_corr = v_no_exit + dv
            if com_project:
                x_corr, v_corr = project_com_to_reference(x_corr, v_corr, x_no_exit, v_no_exit, m_f)

            relE_corr = abs((total_energy_state(x_corr, v_corr, m_f, G=cfg.G) - E_start) / (abs(E_start) + 1e-30))
            reason = "used"
            use = True
            if not all_finite_state(x_corr, v_corr):
                use = False; reason = "nonfinite"
            elif max_radius_state(x_corr) > float(max_radius_gate):
                use = False; reason = "max_radius_gate"
            elif relE_corr > float(energy_gate):
                use = False; reason = "energy_gate"

            if use:
                x = x_corr
                v = v_corr
                rijc, r2c, rc = _compute_pair_geometry(x)
                r_min_cur = float(np.sqrt(r2c.min() + 1e-30))
                a_cache = None  # corrected state invalidates far-field cache
                encounter_used += 1
                if not np.isfinite(first_used_t):
                    first_used_t = t_event
                last_used_t = t_event
                pred_pos_norm_sum += pred_pos_norm
                pred_vel_norm_sum += pred_vel_norm
                min_r_global = min(min_r_global, r_min_cur)
                max_radius_global = max(max_radius_global, max_radius_state(x))

                # If an output sample was written exactly at the exit using the noNN exit,
                # overwrite it with the corrected exit state.
                if si > 0 and abs(times[si - 1] - t_cur) <= max(1e-10, 1e-9 * abs(t_cur)):
                    pos_out[si - 1] = x
                    vel_out[si - 1] = v
                fill_outputs_current()
            else:
                x = x_no_exit
                v = v_no_exit
                a_cache = None
                encounter_fallback += 1

            if event_rows is not None:
                event_rows.append({
                    "t_start": t_event,
                    "t_exit": float(t_cur),
                    "used": int(use),
                    "reason": reason,
                    "r_pair": float(r_pair),
                    "v_rad_norm": float(vr_entry),
                    "v_tan_norm": float(vt_entry),
                    "min_r_window": float(min_r_window),
                    "pred_pos_norm": float(pred_pos_norm),
                    "pred_vel_norm": float(pred_vel_norm),
                    "relE_corr": float(relE_corr),
                })
            continue

        # Standard optimized noNN macro step.
        x, v, r_min_cur, a_cache, n_sub = noNN_optimized_step(x, v, step_dt_macro, r_min_cur, a_cache)
        t_cur += step_dt_macro
        steps += 1
        total_substeps += int(n_sub)
        min_r_global = min(min_r_global, r_min_cur)
        max_radius_global = max(max_radius_global, max_radius_state(x))
        fill_outputs_current()
        if not all_finite_state(x, v) or max_radius_global > float(max_radius_gate):
            break

    while si < len(times):
        pos_out[si] = x
        vel_out[si] = v
        si += 1

    total_time = time.perf_counter() - t_start
    pair_evals = max(int(pair_evals_n), 1)
    used_safe = max(int(encounter_used), 1)
    perf = {
        "label": "SIMON_encounterNN_optimized",
        "steps": int(steps),
        "dt": float(dt),
        "T_years": float(T),
        "n_samples": int(n_samples),
        "total_time_sec": float(total_time),
        "time_per_step_sec": float(total_time / max(int(steps), 1)),
        "total_substeps": int(total_substeps),
        "zone1_frac": int(zone1_n) / pair_evals,
        "zone2_frac": int(zone2_n) / pair_evals,
        "zone3_frac": int(zone3_n) / pair_evals,
        "zone4_far_frac": int(far_n) / pair_evals,
        "zone3_no_nn_frac": int(zone3_no_nn_n) / pair_evals,
        "gate_candidates": int(gate_candidates),
        "encounter_attempts": int(encounter_attempts),
        "encounter_used": int(encounter_used),
        "encounter_fallback": int(encounter_fallback),
        "first_used_t": float(first_used_t),
        "last_used_t": float(last_used_t),
        "pred_pos_norm_mean": float(pred_pos_norm_sum / used_safe),
        "pred_vel_norm_mean": float(pred_vel_norm_sum / used_safe),
        "min_r": float(min_r_global),
        "max_radius": float(max_radius_global),
    }
    return times, pos_out, vel_out, perf

def velocity_rms_sep(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return rms_sep(a, b)


def metric_block(pos: np.ndarray, vel: np.ndarray, pos_ref: np.ndarray, vel_ref: np.ndarray) -> Dict[str, float]:
    pr = rms_sep(pos, pos_ref)
    vr = velocity_rms_sep(vel, vel_ref)
    return {
        "pos_final": float(pr[-1]),
        "pos_timeavg": float(np.sqrt(np.mean(pr * pr))),
        "pos_med": float(np.median(pr)),
        "pos_p95": float(np.percentile(pr, 95)),
        "vel_final": float(vr[-1]),
        "vel_timeavg": float(np.sqrt(np.mean(vr * vr))),
        "vel_med": float(np.median(vr)),
        "vel_p95": float(np.percentile(vr, 95)),
    }


def write_event_csv(path: str, rows: List[Dict[str, object]]) -> None:
    import csv
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("t_start,t_exit,used,reason,r_pair,v_rad_norm,v_tan_norm,min_r_window,pred_pos_norm,pred_vel_norm,relE_corr\n")
        return
    headers = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for row in rows:
            w.writerow(row)


# =============================================================================
# Main: optimized ablation evaluator
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="SIMON encounterNN full ablation evaluator for IC1 dt=0.08.")
    parser.add_argument("--scalar_model", default="pair_correction_nn_v4_bounded.pt",
                        help="v4 scalar Zone-3 model used for SIMON-scalarNN baseline.")
    parser.add_argument("--encounter_model", default="encounter_surrogate_v3_rollout_local_velocitysafe.pt",
                        help="final rollout-local 18D residual encounterNN model.")
    parser.add_argument("--out_dir", default="encounterNN_ablation_v2_optimized_out")
    parser.add_argument("--dt", type=float, default=0.08,
                        help="Validated dt for encounterNN. Current final model should be used at dt=0.08 only.")
    parser.add_argument("--T", type=float, default=100.0)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--window_years", type=float, default=0.5)
    parser.add_argument("--vr_thresh", type=float, default=-0.40)
    parser.add_argument("--energy_gate", type=float, default=0.20)
    parser.add_argument("--max_radius_gate", type=float, default=1e4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_scalar", action="store_true", help="Skip SIMON-scalarNN baseline if only testing encounterNN.")
    parser.add_argument("--write_arrays", action="store_true", help="Save rollout arrays to NPZ; off by default for speed/storage.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = HybridConfig()

    if abs(float(args.dt) - 0.08) > 1e-12:
        print("[WARNING] The final encounterNN model was validated for dt=0.08 only. Continue only if this is intentional.")

    scalar_path = args.scalar_model
    if not scalar_path.endswith(".pt") and os.path.exists(scalar_path + ".pt"):
        scalar_path += ".pt"
    encounter_path = args.encounter_model
    if not encounter_path.endswith(".pt") and os.path.exists(encounter_path + ".pt"):
        encounter_path += ".pt"

    print("=" * 88)
    print("SIMON ENCOUNTERNN FULL ABLATION v2 OPTIMIZED")
    print(f"  scalar_model    : {scalar_path}")
    print(f"  encounter_model : {encounter_path}")
    print(f"  dt/T/samples    : {args.dt:.6f} yr / {args.T:.1f} yr / {args.n_samples}")
    print(f"  gate            : pair 1-2 in Zone 3 and v_rad_norm < {args.vr_thresh:.3f}")
    print(f"  window          : {args.window_years:.3f} yr")
    print(f"  out_dir         : {args.out_dir}")
    print("=" * 88)

    scalar_model = load_v4_model(scalar_path)
    encounter_model = load_encounter_model(encounter_path, device=args.device)

    # IC1 default, same as existing v4 evaluator.
    m = np.array([1.0, 0.01, 0.005], dtype=np.float64)
    x0 = np.array([[0,0,0], [1,0,0], [0,1.2,0]], dtype=np.float64)
    v0 = np.array([[0,0,0], [0,1,0], [-0.9,0,0]], dtype=np.float64)
    M = m.sum()
    x0 = x0 - (m[:, None] * x0).sum(0) / M
    v0 = v0 - (m[:, None] * v0).sum(0) / M

    print("[1/4] IAS15 reference...")
    tr, pr, vr_ref, perf_r = simulate_rebound_ias15(x0, v0, m, cfg.G, float(args.T), int(args.n_samples))
    print(f"      IAS15 time={perf_r['total_time_sec']:.6f}s")

    print("[2/4] SIMON-noNN baseline using original v4 evaluator path...")
    _, p_no, v_no, perf_no = simulate_leapfrog_v4_zone3(
        x0, v0, m, scalar_model, cfg, float(args.dt), float(args.T), int(args.n_samples),
        use_zone3_nn=False, label="SIMON_noNN")
    met_no = metric_block(p_no, v_no, pr, vr_ref)
    sp_no = perf_r["total_time_sec"] / max(perf_no["total_time_sec"], 1e-12)
    print(f"      noNN pos_timeavg={met_no['pos_timeavg']:.6e} pos_final={met_no['pos_final']:.6e} time={perf_no['total_time_sec']:.6f}s speedup={sp_no:.2f}x")

    met_scalar = None
    perf_scalar = None
    sp_scalar = float("nan")
    if not args.skip_scalar:
        print("[3/4] SIMON-scalarNN baseline using original v4 evaluator path...")
        _, p_sc, v_sc, perf_scalar = simulate_leapfrog_v4_zone3(
            x0, v0, m, scalar_model, cfg, float(args.dt), float(args.T), int(args.n_samples),
            use_zone3_nn=True, label="SIMON_scalarNN")
        met_scalar = metric_block(p_sc, v_sc, pr, vr_ref)
        sp_scalar = perf_r["total_time_sec"] / max(perf_scalar["total_time_sec"], 1e-12)
        print(f"      scalarNN pos_timeavg={met_scalar['pos_timeavg']:.6e} pos_final={met_scalar['pos_final']:.6e} time={perf_scalar['total_time_sec']:.6f}s speedup={sp_scalar:.2f}x")
    else:
        print("[3/4] SIMON-scalarNN baseline skipped.")

    print("[4/4] SIMON-encounterNN residual-correction rollout...")
    event_rows: List[Dict[str, object]] = []
    _, p_en, v_en, perf_en = simulate_leapfrog_encounterNN(
        x0, v0, m, encounter_model, cfg,
        dt=float(args.dt), T=float(args.T), n_samples=int(args.n_samples),
        window_years=float(args.window_years), vr_thresh=float(args.vr_thresh),
        energy_gate=float(args.energy_gate), max_radius_gate=float(args.max_radius_gate),
        device=args.device, com_project=True, event_rows=event_rows)
    met_en = metric_block(p_en, v_en, pr, vr_ref)
    sp_en = perf_r["total_time_sec"] / max(perf_en["total_time_sec"], 1e-12)
    print(f"      encounterNN pos_timeavg={met_en['pos_timeavg']:.6e} pos_final={met_en['pos_final']:.6e} time={perf_en['total_time_sec']:.6f}s speedup={sp_en:.2f}x used={int(perf_en['encounter_used'])}")

    # Output event CSV.
    event_csv = os.path.join(args.out_dir, "encounter_events.csv")
    write_event_csv(event_csv, event_rows)

    # Optional arrays.
    if args.write_arrays:
        np.savez_compressed(
            os.path.join(args.out_dir, "ablation_arrays.npz"),
            times=tr, pos_ias15=pr, vel_ias15=vr_ref,
            pos_noNN=p_no, vel_noNN=v_no,
            pos_encounterNN=p_en, vel_encounterNN=v_en,
            x0=x0, v0=v0, m=m,
        )
        if met_scalar is not None:
            # Add scalar arrays in a separate file to avoid holding too much memory in the writer call above.
            np.savez_compressed(os.path.join(args.out_dir, "scalarNN_arrays.npz"), pos_scalarNN=p_sc, vel_scalarNN=v_sc)

    def pct_gain(base: float, new: float) -> float:
        return 100.0 * (base - new) / max(abs(base), 1e-30)

    # Summary.
    summary_path = os.path.join(args.out_dir, "encounterNN_ablation_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SIMON encounterNN ablation v2 optimized\n")
        f.write("=" * 88 + "\n")
        f.write("Purpose: compare IAS15, SIMON-noNN, SIMON-scalarNN, and SIMON-encounterNN on IC1.\n")
        f.write("EncounterNN deployment formula: corrected_exit = noNN_exit + predicted_residual.\n")
        f.write("The noNN and scalarNN baselines are run through the original v4 evaluator path. The encounterNN path also reuses the same optimized noNN kernel strategy, with only a rare residual-correction branch at gated encounters.\n\n")
        f.write(f"scalar_model: {scalar_path}\n")
        f.write(f"encounter_model: {encounter_path}\n")
        f.write(f"dt: {args.dt:.8f}\nT_years: {args.T:.8f}\nn_samples: {args.n_samples}\n")
        f.write(f"window_years: {args.window_years:.8f}\nvr_thresh: {args.vr_thresh:.8f}\nenergy_gate: {args.energy_gate:.8f}\n")
        f.write(f"device: {args.device}\n\n")
        f.write("Runtime\n")
        f.write("-" * 88 + "\n")
        f.write(f"IAS15_total_time_sec: {perf_r['total_time_sec']:.9f}\n")
        f.write(f"SIMON_noNN_total_time_sec: {perf_no['total_time_sec']:.9f}\n")
        f.write(f"SIMON_noNN_speedup_vs_IAS15: {sp_no:.6f}\n")
        if perf_scalar is not None:
            f.write(f"SIMON_scalarNN_total_time_sec: {perf_scalar['total_time_sec']:.9f}\n")
            f.write(f"SIMON_scalarNN_speedup_vs_IAS15: {sp_scalar:.6f}\n")
        f.write(f"SIMON_encounterNN_total_time_sec: {perf_en['total_time_sec']:.9f}\n")
        f.write(f"SIMON_encounterNN_speedup_vs_IAS15: {sp_en:.6f}\n\n")

        f.write("Accuracy vs IAS15\n")
        f.write("-" * 88 + "\n")
        f.write("method\tpos_final\tpos_timeavg\tpos_med\tpos_p95\tvel_final\tvel_timeavg\tvel_med\tvel_p95\ttotal_time_sec\tspeedup_vs_ias15\n")
        f.write(f"SIMON-noNN\t{met_no['pos_final']:.9e}\t{met_no['pos_timeavg']:.9e}\t{met_no['pos_med']:.9e}\t{met_no['pos_p95']:.9e}\t{met_no['vel_final']:.9e}\t{met_no['vel_timeavg']:.9e}\t{met_no['vel_med']:.9e}\t{met_no['vel_p95']:.9e}\t{perf_no['total_time_sec']:.9f}\t{sp_no:.6f}\n")
        if met_scalar is not None:
            f.write(f"SIMON-scalarNN\t{met_scalar['pos_final']:.9e}\t{met_scalar['pos_timeavg']:.9e}\t{met_scalar['pos_med']:.9e}\t{met_scalar['pos_p95']:.9e}\t{met_scalar['vel_final']:.9e}\t{met_scalar['vel_timeavg']:.9e}\t{met_scalar['vel_med']:.9e}\t{met_scalar['vel_p95']:.9e}\t{perf_scalar['total_time_sec']:.9f}\t{sp_scalar:.6f}\n")
        f.write(f"SIMON-encounterNN\t{met_en['pos_final']:.9e}\t{met_en['pos_timeavg']:.9e}\t{met_en['pos_med']:.9e}\t{met_en['pos_p95']:.9e}\t{met_en['vel_final']:.9e}\t{met_en['vel_timeavg']:.9e}\t{met_en['vel_med']:.9e}\t{met_en['vel_p95']:.9e}\t{perf_en['total_time_sec']:.9f}\t{sp_en:.6f}\n\n")

        f.write("EncounterNN event stats\n")
        f.write("-" * 88 + "\n")
        for key in ["gate_candidates", "encounter_attempts", "encounter_used", "encounter_fallback", "first_used_t", "last_used_t", "pred_pos_norm_mean", "pred_vel_norm_mean", "min_r", "max_radius"]:
            f.write(f"{key}: {perf_en.get(key)}\n")
        f.write(f"event_csv: {event_csv}\n\n")

        f.write("Delta vs SIMON-noNN\n")
        f.write("-" * 88 + "\n")
        for name, met in [("SIMON-encounterNN", met_en)] + ([] if met_scalar is None else [("SIMON-scalarNN", met_scalar)]):
            f.write(f"{name}_pos_final_gain_pct: {pct_gain(met_no['pos_final'], met['pos_final']):+.6f}\n")
            f.write(f"{name}_pos_timeavg_gain_pct: {pct_gain(met_no['pos_timeavg'], met['pos_timeavg']):+.6f}\n")
            f.write(f"{name}_vel_final_gain_pct: {pct_gain(met_no['vel_final'], met['vel_final']):+.6f}\n")
            f.write(f"{name}_vel_timeavg_gain_pct: {pct_gain(met_no['vel_timeavg'], met['vel_timeavg']):+.6f}\n")
        f.write("\nNotes\n")
        f.write("- noNN and scalarNN use the original v4 Zone2/Zone3 evaluator path, so their accuracy should match the v3/v4 log file for the same model, dt, T, n_samples, and environment.\n")
        f.write("- Runtime can vary slightly across runs and hardware; numerical accuracy values should match up to normal floating-point/order effects.\n")
        f.write("- EncounterNN is currently validated for dt=0.08, IC1, active pair 1-2, window_years=0.5.\n")

    # Compact console table.
    print("\nFinal ablation table")
    print("method              pos_final     pos_timeavg   vel_final     vel_timeavg   time(s)   speedup")
    print("-" * 94)
    print(f"SIMON-noNN         {met_no['pos_final']:11.4e} {met_no['pos_timeavg']:12.4e} {met_no['vel_final']:11.4e} {met_no['vel_timeavg']:12.4e} {perf_no['total_time_sec']:8.4f} {sp_no:8.3f}x")
    if met_scalar is not None:
        print(f"SIMON-scalarNN     {met_scalar['pos_final']:11.4e} {met_scalar['pos_timeavg']:12.4e} {met_scalar['vel_final']:11.4e} {met_scalar['vel_timeavg']:12.4e} {perf_scalar['total_time_sec']:8.4f} {sp_scalar:8.3f}x")
    print(f"SIMON-encounterNN  {met_en['pos_final']:11.4e} {met_en['pos_timeavg']:12.4e} {met_en['vel_final']:11.4e} {met_en['vel_timeavg']:12.4e} {perf_en['total_time_sec']:8.4f} {sp_en:8.3f}x")
    print(f"\n[done] wrote {summary_path}")
    print(f"[done] wrote {event_csv}")


if __name__ == "__main__":
    main()
