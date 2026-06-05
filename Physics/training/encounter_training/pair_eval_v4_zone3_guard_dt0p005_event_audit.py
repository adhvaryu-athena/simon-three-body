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
    # Pair-level episode guard: once a pair enters Zone 2, keep Zone 3 NN
    # disabled for that pair until it has moved safely above this radius.
    handoff_unlock_r: float = 0.08


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
    handoff_guard: bool = False,
    handoff_unlock_r: Optional[float] = None,
    handoff_transition_rows=None,
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
    zone3_handoff_skip_n = np.int64(0)
    strong_z3_n = np.int64(0)
    frozen_steps_with_z3_n = 0
    c_sum = 0.0
    c_count = 0
    c_lo = np.inf
    c_hi = -np.inf

    # Pair-level handoff guard state. This preserves the optimized step logic;
    # it only changes whether an eligible Zone-3 pair uses the NN or the
    # same c=1*F_soft fallback as the no-NN baseline.
    handoff_guard_enabled = bool(handoff_guard and use_zone3_nn)
    unlock_r = float(handoff_unlock_r if handoff_unlock_r is not None else cfg.handoff_unlock_r)
    if unlock_r <= cfg.adapt_thresh:
        raise ValueError("handoff_unlock_r must be larger than adapt_thresh")
    handoff_locked = np.zeros(P, dtype=bool)

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
        handoff_skip = np.zeros(P, dtype=bool)

        # Pair-level episode guard.  This is deliberately local to the pair:
        # a pair that enters Zone 2 is not allowed to resume Zone 3 NN until
        # it has moved outside the buffer radius.  This avoids the observed
        # Zone3(boost) -> Zone2(substep) -> Zone3(weaken) split.
        if handoff_guard_enabled:
            prev_locked = handoff_locked.copy()
            unlock_mask = prev_locked & (r >= unlock_r)
            lock_mask = (~prev_locked) & (r < cfg.adapt_thresh)

            if handoff_transition_rows is not None and (np.any(lock_mask) or np.any(unlock_mask)):
                for p_idx in np.where(lock_mask | unlock_mask)[0]:
                    rij_p = rij[p_idx]
                    r_p = float(r[p_idx])
                    r_hat_p = rij_p / (r_p + 1e-30)
                    vij_p = vel[jj[p_idx]] - vel[ii[p_idx]]
                    v_rad_p = float(np.dot(vij_p, r_hat_p))
                    v_tan_vec_p = vij_p - v_rad_p * r_hat_p
                    v_tan_p = float(np.sqrt(np.dot(v_tan_vec_p, v_tan_vec_p) + 1e-30))
                    v_scale_p = float(np.sqrt(cfg.G * (mi_arr[p_idx] + mj_arr[p_idx]) / (r_p + 1e-30)))
                    event_type = "LOCK_ENTER_ZONE2" if lock_mask[p_idx] else "UNLOCK_EXIT_BUFFER"
                    handoff_transition_rows.append({
                        "step_idx": int(_log_step_idx),
                        "substep_idx": int(_log_substep_idx),
                        "t_yr": float(_log_t_yr),
                        "dt_step": float(_log_step_dt),
                        "event_type": event_type,
                        "pair_i": int(ii[p_idx]),
                        "pair_j": int(jj[p_idx]),
                        "r_au": r_p,
                        "v_rad_norm": v_rad_p / (v_scale_p + 1e-30),
                        "v_tan_norm": v_tan_p / (v_scale_p + 1e-30),
                        "was_locked_before": bool(prev_locked[p_idx]),
                    })

            handoff_locked[unlock_mask] = False
            handoff_locked[lock_mask] = True

        z0 = (r >= cfg.adapt_thresh) & (r < cfg.nn_thresh)
        if not np.any(z0):
            return frozen_c, eligible, fallback, strong, handoff_skip

        if use_zone3_nn:
            handoff_skip = z0 & handoff_locked if handoff_guard_enabled else handoff_skip
            z_nn = z0 & ~handoff_skip

            # Guard-skipped Zone 3 pairs remain eligible so compute_acc applies
            # c=1*F_soft, matching the no-Zone-3-NN baseline for those pairs.
            if np.any(handoff_skip):
                eligible[handoff_skip] = True

            if np.any(z_nn):
                nn_in, v_rad_norm = _zone3_features_from_geom(rij, r2, r, vel, step_dt, z_nn)
                c = v4_forward_numpy(nn_in, w)
                r_soft = np.sqrt(r2[z_nn] + eps2)
                fb = ((r_soft < cfg.r_soft_min) | (c < cfg.c_min) |
                      (c > cfg.c_max) | ~np.isfinite(c))

                frozen_c[z_nn] = c
                eligible[z_nn] = True
                fallback[z_nn] = fb
                strong[z_nn] = v_rad_norm < -0.6

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

            # ---- Optional per-firing/skip logger (no-op when nn_log_rows is None) ----
            if nn_log_rows is not None:
                z3_pair_idx = np.where(z0)[0]
                r_z3 = r[z0]
                rij_z3 = rij[z0]
                r_hat_z3 = rij_z3 / (r_z3[:, None] + 1e-30)
                vij_z3 = (vel[jj] - vel[ii])[z0]
                v_rad_full = np.einsum("ij,ij->i", vij_z3, r_hat_z3)
                v_tan_vec = vij_z3 - v_rad_full[:, None] * r_hat_z3
                v_tan_mag = np.sqrt(np.einsum("ij,ij->i", v_tan_vec, v_tan_vec) + 1e-30)
                v_scale = np.sqrt(cfg.G * (mi_arr[z0] + mj_arr[z0]) / (r_z3 + 1e-30))
                v_rad_norm_all = v_rad_full / (v_scale + 1e-30)
                v_tan_norm = v_tan_mag / (v_scale + 1e-30)
                r_soft_z3 = np.sqrt(r2[z0] + eps2)
                c_anal_ref = (r_soft_z3 / (r_z3 + 1e-30)) ** 3
                n_z3_this_step = int(len(z3_pair_idx))
                for k, p_idx in enumerate(z3_pair_idx):
                    skipped = bool(handoff_skip[p_idx])
                    nn_log_rows.append({
                        "step_idx": _log_step_idx,
                        "substep_idx": _log_substep_idx,
                        "t_yr": float(_log_t_yr),
                        "dt_step": float(_log_step_dt),
                        "pair_i": int(ii[p_idx]),
                        "pair_j": int(jj[p_idx]),
                        "r_au": float(r_z3[k]),
                        "v_rad_norm": float(v_rad_norm_all[k]),
                        "v_tan_norm": float(v_tan_norm[k]),
                        "m_i": float(mi_arr[p_idx]),
                        "m_j": float(mj_arr[p_idx]),
                        "c_pred": float(frozen_c[p_idx]) if (not skipped and np.isfinite(frozen_c[p_idx])) else float("nan"),
                        "c_anal_softening": float(c_anal_ref[k]),
                        "gate_fallback": bool(fallback[p_idx]),
                        "strong_approach": bool(v_rad_norm_all[k] < -0.6),
                        "handoff_guard_skip": skipped,
                        "handoff_locked": bool(handoff_locked[p_idx]) if handoff_guard_enabled else False,
                        "n_zone3_pairs_this_step": n_z3_this_step,
                    })
        else:
            eligible[z0] = True

        return frozen_c, eligible, fallback, strong, handoff_skip

    def compute_acc_geom(rij, r2, r, frozen_c, eligible, fallback, strong, handoff_skip):
        """Acceleration from shared geometry. Stats use numpy counters."""
        nonlocal pair_evals_n, zone1_n, zone2_n, zone3_n, far_n
        nonlocal zone3_nn_n, zone3_gate_fb_n, zone3_no_nn_n, zone3_handoff_skip_n, strong_z3_n

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
                idx_apply = np.where(apply_mask)[0]
                skip = handoff_skip[idx_apply]

                # Guard-skipped Zone 3 pairs use the same no-NN baseline force:
                # c=1*F_soft.  Active NN pairs use c*F_soft unless hard-gated,
                # in which case they fall back to direct Newtonian.
                F_scalar[idx_apply] = F_soft

                active_idx = idx_apply[~skip]
                if len(active_idx):
                    fb = fallback[active_idx]
                    c = frozen_c[active_idx]
                    F_soft_active = Gmimj[active_idx] / ((r2[active_idx] + eps2) ** 1.5 + 1e-30)
                    F_scalar[active_idx] = np.where(fb, F_scalar[active_idx], c * F_soft_active)
                    zone3_nn_n += int(len(active_idx))
                    zone3_gate_fb_n += int(fb.sum())
                    strong_z3_n += int(strong[active_idx].sum())
                zone3_handoff_skip_n += int(skip.sum())
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
        frozen_c, eligible, fallback, strong, handoff_skip = make_frozen_zone3_state_geom(
            rij_in, r2_in, r_in, v_in, step_dt
        )
        a0 = compute_acc_geom(rij_in, r2_in, r_in,
                              frozen_c, eligible, fallback, strong, handoff_skip)
        vh = v_in + 0.5 * step_dt * a0
        x_new = x_in + step_dt * vh
        rij_n, r2_n, r_n = _compute_pair_geometry(x_new)
        a1 = compute_acc_geom(rij_n, r2_n, r_n,
                              frozen_c, eligible, fallback, strong, handoff_skip)
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
        "zone3_handoff_skip_frac": int(zone3_handoff_skip_n) / pair_evals,
        "handoff_guard_enabled": bool(handoff_guard_enabled),
        "handoff_unlock_r": float(unlock_r),
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
# dt=0.005 event audit: noNN vs unguarded NN vs guarded NN
# =============================================================================
def _nearest_idx(times, t):
    return int(np.argmin(np.abs(times - float(t))))


def _rms_at_index(pos, ref, idx):
    return float(rms_sep(pos[[idx]], ref[[idx]])[0])


def _augment_event_rows(rows, times, ref_pos, no_pos, mode_pos):
    """Add local RMS diagnostics to NN firing/skip rows."""
    if rows is None:
        return []
    out = []
    T = float(times[-1])
    for row in rows:
        r2 = dict(row)
        t0 = float(row.get("t_yr", 0.0))
        for tag, off in [("minus_1yr", -1.0), ("at_event", 0.0), ("plus_1yr", 1.0), ("plus_5yr", 5.0)]:
            tt = min(max(t0 + off, 0.0), T)
            k = _nearest_idx(times, tt)
            e_no = _rms_at_index(no_pos, ref_pos, k)
            e_mode = _rms_at_index(mode_pos, ref_pos, k)
            r2[f"sample_time_{tag}"] = float(times[k])
            r2[f"noNN_err_{tag}"] = e_no
            r2[f"mode_err_{tag}"] = e_mode
            r2[f"mode_minus_noNN_{tag}"] = e_mode - e_no
        out.append(r2)
    return out


def _write_rows_csv(path, rows, preferred_headers=None):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    headers = []
    if preferred_headers:
        headers.extend(preferred_headers)
    for row in rows:
        for k in row.keys():
            if k not in headers:
                headers.append(k)
    if not headers:
        headers = preferred_headers or []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pair_min_distance_series(pos):
    pairs = [(0, 1), (0, 2), (1, 2)]
    n = pos.shape[0]
    rmin = np.zeros(n, dtype=np.float64)
    pair_name = []
    for k in range(n):
        vals = []
        for (i, j) in pairs:
            vals.append(float(np.linalg.norm(pos[k, j] - pos[k, i])))
        idx = int(np.argmin(vals))
        rmin[k] = vals[idx]
        pair_name.append(f"{pairs[idx][0]}-{pairs[idx][1]}")
    return rmin, pair_name


def _summarize_rows_by_kind(rows):
    n = len(rows)
    z3_nn = sum(1 for r in rows if not str(r.get("handoff_guard_skip", "False")).lower() in ("true", "1"))
    z3_skip = sum(1 for r in rows if str(r.get("handoff_guard_skip", "False")).lower() in ("true", "1"))
    strong = sum(1 for r in rows if str(r.get("strong_approach", "False")).lower() in ("true", "1"))
    return n, z3_nn, z3_skip, strong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="pair_correction_nn_v4_bounded.pt")
    parser.add_argument("--out_dir", default="zone3_dt0p005_guard_event_audit_out")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--T", type=float, default=100.0)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--handoff_unlock_r", type=float, default=0.08)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = HybridConfig()
    cfg.handoff_unlock_r = float(args.handoff_unlock_r)

    mp = args.model_path
    if not mp.endswith(".pt") and os.path.exists(mp + ".pt"):
        mp += ".pt"
    model = load_v4_model(mp)

    T = float(args.T)
    ns = int(args.n_samples)
    dt = float(args.dt)

    # IC1 default, same as previous evaluators.
    m = np.array([1.0, 0.01, 0.005], dtype=np.float64)
    x0 = np.array([[0,0,0], [1,0,0], [0,1.2,0]], dtype=np.float64)
    v0 = np.array([[0,0,0], [0,1,0], [-0.9,0,0]], dtype=np.float64)
    M = m.sum()
    x0 = x0 - (m[:, None] * x0).sum(0) / M
    v0 = v0 - (m[:, None] * v0).sum(0) / M

    print("[dt0p005-audit] IAS15 baseline...")
    tr, pr, vr, perf_r = simulate_rebound_ias15(x0, v0, m, cfg.G, T, ns)
    print(f"[dt0p005-audit] IAS15 time={perf_r['total_time_sec']:.3f}s")

    print(f"[dt0p005-audit] no-Zone-3-NN baseline dt={dt}...")
    _, p_no, _, perf_no = simulate_leapfrog_v4_zone3(
        x0, v0, m, model, cfg, dt, T, ns,
        use_zone3_nn=False, label="zone3_no_nn",
    )
    delta_no = rms_sep(p_no, pr)

    print(f"[dt0p005-audit] unguarded v4 Zone 3 NN dt={dt}...")
    rows_ung = []
    _, p_ung, _, perf_ung = simulate_leapfrog_v4_zone3(
        x0, v0, m, model, cfg, dt, T, ns,
        use_zone3_nn=True, label="v4_zone3_nn_unguarded",
        nn_log_rows=rows_ung,
        handoff_guard=False,
    )
    delta_ung = rms_sep(p_ung, pr)

    print(f"[dt0p005-audit] guarded v4 Zone 3 NN dt={dt}, unlock_r={args.handoff_unlock_r}...")
    rows_guard = []
    trans_guard = []
    _, p_guard, _, perf_guard = simulate_leapfrog_v4_zone3(
        x0, v0, m, model, cfg, dt, T, ns,
        use_zone3_nn=True, label="v4_zone3_nn_handoff_guard",
        nn_log_rows=rows_guard,
        handoff_guard=True,
        handoff_unlock_r=args.handoff_unlock_r,
        handoff_transition_rows=trans_guard,
    )
    delta_guard = rms_sep(p_guard, pr)

    # Augment rows with local RMS diagnostics versus noNN and IAS15.
    rows_ung_aug = _augment_event_rows(rows_ung, tr, pr, p_no, p_ung)
    rows_guard_aug = _augment_event_rows(rows_guard, tr, pr, p_no, p_guard)
    trans_guard_aug = _augment_event_rows(trans_guard, tr, pr, p_no, p_guard)

    preferred = [
        "step_idx", "substep_idx", "t_yr", "dt_step", "event_type",
        "pair_i", "pair_j", "r_au", "v_rad_norm", "v_tan_norm",
        "m_i", "m_j", "c_pred", "c_anal_softening", "gate_fallback",
        "strong_approach", "handoff_guard_skip", "handoff_locked",
        "n_zone3_pairs_this_step",
        "sample_time_minus_1yr", "noNN_err_minus_1yr", "mode_err_minus_1yr", "mode_minus_noNN_minus_1yr",
        "sample_time_at_event", "noNN_err_at_event", "mode_err_at_event", "mode_minus_noNN_at_event",
        "sample_time_plus_1yr", "noNN_err_plus_1yr", "mode_err_plus_1yr", "mode_minus_noNN_plus_1yr",
        "sample_time_plus_5yr", "noNN_err_plus_5yr", "mode_err_plus_5yr", "mode_minus_noNN_plus_5yr",
    ]
    trans_pref = [
        "step_idx", "substep_idx", "t_yr", "dt_step", "event_type",
        "pair_i", "pair_j", "r_au", "v_rad_norm", "v_tan_norm", "was_locked_before",
        "sample_time_minus_1yr", "noNN_err_minus_1yr", "mode_err_minus_1yr", "mode_minus_noNN_minus_1yr",
        "sample_time_at_event", "noNN_err_at_event", "mode_err_at_event", "mode_minus_noNN_at_event",
        "sample_time_plus_1yr", "noNN_err_plus_1yr", "mode_err_plus_1yr", "mode_minus_noNN_plus_1yr",
        "sample_time_plus_5yr", "noNN_err_plus_5yr", "mode_err_plus_5yr", "mode_minus_noNN_plus_5yr",
    ]

    _write_rows_csv(os.path.join(args.out_dir, "zone3_events_unguarded_dt0p005.csv"), rows_ung_aug, preferred)
    _write_rows_csv(os.path.join(args.out_dir, "zone3_events_guarded_dt0p005.csv"), rows_guard_aug, preferred)
    _write_rows_csv(os.path.join(args.out_dir, "handoff_transitions_guarded_dt0p005.csv"), trans_guard_aug, trans_pref)

    # Min distance + RMS error series for plotting/inspection.
    rmin_no, pair_no = _pair_min_distance_series(p_no)
    rmin_ung, pair_ung = _pair_min_distance_series(p_ung)
    rmin_guard, pair_guard = _pair_min_distance_series(p_guard)
    rmin_ref, pair_ref = _pair_min_distance_series(pr)
    series_rows = []
    for k, t in enumerate(tr):
        series_rows.append({
            "t_yr": float(t),
            "ias15_rmin": float(rmin_ref[k]),
            "noNN_rmin": float(rmin_no[k]),
            "unguarded_rmin": float(rmin_ung[k]),
            "guarded_rmin": float(rmin_guard[k]),
            "ias15_pair": pair_ref[k],
            "noNN_pair": pair_no[k],
            "unguarded_pair": pair_ung[k],
            "guarded_pair": pair_guard[k],
            "noNN_err": float(delta_no[k]),
            "unguarded_err": float(delta_ung[k]),
            "guarded_err": float(delta_guard[k]),
            "unguarded_minus_noNN": float(delta_ung[k] - delta_no[k]),
            "guarded_minus_noNN": float(delta_guard[k] - delta_no[k]),
        })
    _write_rows_csv(os.path.join(args.out_dir, "min_distance_error_dt0p005.csv"), series_rows)

    # Summary text.
    n_ung, n_ung_nn, n_ung_skip, n_ung_strong = _summarize_rows_by_kind(rows_ung_aug)
    n_g, n_g_nn, n_g_skip, n_g_strong = _summarize_rows_by_kind(rows_guard_aug)
    n_lock = sum(1 for r in trans_guard_aug if r.get("event_type") == "LOCK_ENTER_ZONE2")
    n_unlock = sum(1 for r in trans_guard_aug if r.get("event_type") == "UNLOCK_EXIT_BUFFER")

    summary_path = os.path.join(args.out_dir, "dt0p005_guard_event_audit_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("dt=0.005 Zone 3 handoff-guard event audit\n")
        f.write("="*78 + "\n")
        f.write(f"T_years: {T}\n")
        f.write(f"n_samples: {ns}\n")
        f.write(f"dt: {dt}\n")
        f.write(f"model: {mp}\n")
        f.write(f"handoff_unlock_r: {args.handoff_unlock_r:.6f} AU\n")
        f.write("method: Zone 2 direct Newtonian + substeps; Zone 3 frozen v4 NN; compare noNN, unguarded, guarded.\n\n")
        f.write("Performance summary\n")
        f.write("mode\tfinal_err\ttime_avg_rms\tzone2_frac\tzone3_nn_frac\tzone3_skip_frac\tgate_frac\tsteps_with_zone3\n")
        f.write(f"noNN\t{float(delta_no[-1]):.6e}\t{float(np.sqrt(np.mean(delta_no**2))):.6e}\t{perf_no['zone2_frac']:.6f}\t{perf_no['zone3_nn_frac']:.6f}\t{perf_no['zone3_handoff_skip_frac']:.6f}\t{perf_no['zone3_gate_fallback_frac']:.6f}\t{perf_no['frozen_steps_with_zone3']}\n")
        f.write(f"unguarded_NN\t{float(delta_ung[-1]):.6e}\t{float(np.sqrt(np.mean(delta_ung**2))):.6e}\t{perf_ung['zone2_frac']:.6f}\t{perf_ung['zone3_nn_frac']:.6f}\t{perf_ung['zone3_handoff_skip_frac']:.6f}\t{perf_ung['zone3_gate_fallback_frac']:.6f}\t{perf_ung['frozen_steps_with_zone3']}\n")
        f.write(f"guarded_NN\t{float(delta_guard[-1]):.6e}\t{float(np.sqrt(np.mean(delta_guard**2))):.6e}\t{perf_guard['zone2_frac']:.6f}\t{perf_guard['zone3_nn_frac']:.6f}\t{perf_guard['zone3_handoff_skip_frac']:.6f}\t{perf_guard['zone3_gate_fallback_frac']:.6f}\t{perf_guard['frozen_steps_with_zone3']}\n\n")
        f.write("Event counts\n")
        f.write(f"unguarded rows: total={n_ung}, active_nn={n_ung_nn}, skips={n_ung_skip}, strong={n_ung_strong}\n")
        f.write(f"guarded rows: total={n_g}, active_nn={n_g_nn}, skips={n_g_skip}, strong={n_g_strong}\n")
        f.write(f"guarded lock transitions: locks={n_lock}, unlocks={n_unlock}\n\n")
        f.write("Top guarded events/skips by guarded_minus_noNN_plus_5yr\n")
        top_g = sorted(rows_guard_aug, key=lambda r: float(r.get("mode_minus_noNN_plus_5yr", 0.0)), reverse=True)[:12]
        f.write("t\tpair\tr\tvr_norm\tc_pred\tskip\tlocked\tminus_1yr\tat_event\tplus_1yr\tplus_5yr\n")
        for r in top_g:
            pair = f"{r.get('pair_i')}-{r.get('pair_j')}"
            f.write(f"{float(r.get('t_yr', np.nan)):.6f}\t{pair}\t{float(r.get('r_au', np.nan)):.6e}\t{float(r.get('v_rad_norm', np.nan)):.6f}\t{float(r.get('c_pred', np.nan)):.6f}\t{r.get('handoff_guard_skip')}\t{r.get('handoff_locked')}\t{float(r.get('mode_minus_noNN_minus_1yr', np.nan)):.6e}\t{float(r.get('mode_minus_noNN_at_event', np.nan)):.6e}\t{float(r.get('mode_minus_noNN_plus_1yr', np.nan)):.6e}\t{float(r.get('mode_minus_noNN_plus_5yr', np.nan)):.6e}\n")
        f.write("\nTop handoff transitions by guarded_minus_noNN_plus_5yr\n")
        top_t = sorted(trans_guard_aug, key=lambda r: float(r.get("mode_minus_noNN_plus_5yr", 0.0)), reverse=True)[:12]
        f.write("t\tevent\tpair\tr\tvr_norm\tminus_1yr\tat_event\tplus_1yr\tplus_5yr\n")
        for r in top_t:
            pair = f"{r.get('pair_i')}-{r.get('pair_j')}"
            f.write(f"{float(r.get('t_yr', np.nan)):.6f}\t{r.get('event_type')}\t{pair}\t{float(r.get('r_au', np.nan)):.6e}\t{float(r.get('v_rad_norm', np.nan)):.6f}\t{float(r.get('mode_minus_noNN_minus_1yr', np.nan)):.6e}\t{float(r.get('mode_minus_noNN_at_event', np.nan)):.6e}\t{float(r.get('mode_minus_noNN_plus_1yr', np.nan)):.6e}\t{float(r.get('mode_minus_noNN_plus_5yr', np.nan)):.6e}\n")
        f.write("\nFiles written\n")
        f.write("zone3_events_unguarded_dt0p005.csv\n")
        f.write("zone3_events_guarded_dt0p005.csv\n")
        f.write("handoff_transitions_guarded_dt0p005.csv\n")
        f.write("min_distance_error_dt0p005.csv\n")

    # Small diagnostic plot.
    try:
        plt.figure(figsize=(8, 4.6))
        plt.semilogy(tr, delta_no, label="no-Zone-3-NN")
        plt.semilogy(tr, delta_ung, label="unguarded NN")
        plt.semilogy(tr, delta_guard, label="guarded NN")
        for r in rows_guard_aug:
            if str(r.get("handoff_guard_skip", "False")).lower() in ("true", "1"):
                plt.axvline(float(r["t_yr"]), color="gray", alpha=0.12, lw=0.8)
        for r in trans_guard_aug:
            if r.get("event_type") == "LOCK_ENTER_ZONE2":
                plt.axvline(float(r["t_yr"]), color="red", alpha=0.18, lw=0.9)
            elif r.get("event_type") == "UNLOCK_EXIT_BUFFER":
                plt.axvline(float(r["t_yr"]), color="green", alpha=0.18, lw=0.9)
        plt.xlabel("Time (yr)")
        plt.ylabel("RMS error vs IAS15")
        plt.title("dt=0.005: noNN vs unguarded vs guarded")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "dt0p005_error_audit.png"), dpi=220)
        plt.close()
    except Exception as e:
        print(f"[dt0p005-audit] plot failed: {e}")

    print(f"[dt0p005-audit] wrote {args.out_dir}")
    print(f"[dt0p005-audit] summary: {summary_path}")
    print("[dt0p005-audit] done.")


if __name__ == "__main__":
    main()
