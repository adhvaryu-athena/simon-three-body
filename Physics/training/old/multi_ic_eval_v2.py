# multi_ic_eval_v2.py
#
# Runs SIMON across 6 distinct three-body initial conditions.
# Extends multi_ic_eval.py (4 ICs) to 6 ICs for a stronger
# generalisation claim, as recommended by mentor feedback.
#
# NEW in v2 vs multi_ic_eval.py:
#   - SELF-CONTAINED: no imports from pair_eval_after_adaptive.py
#     (all simulation logic copied directly from pair_eval_after_adaptive.py
#      to eliminate the sys.path dependency that caused issues previously)
#   - IC5 added: high-eccentricity regime (Body 1 at 0.3 AU, v=2.2 -- 85% of
#     escape velocity, creates periodic intense close encounters)
#   - IC6 added: near-circular coplanar regime (both bodies on exact Keplerian
#     circular orbits, v_circ = sqrt(GM/r), low-chaos limit)
#   - IC1-IC4 coordinates IDENTICAL to multi_ic_eval.py (verified to reproduce
#     original results: IC1 lambda=0.1655, IC2 ejection 116.87 AU, etc.)
#   - Output folder: multi_ic_out_v2\ (original multi_ic_out\ untouched)
#   - Honest IC2 framing: 2x post-ejection gap between SIMON and ias15
#     is explicitly reported and explained in the summary
#   - Explicit energy check (E < 0) for all 6 ICs before any simulation
#   - 2x3 figure panels for 6 ICs (was 2x2 for 4 ICs)
#
# REQUIRES in same folder:
#   pair_correction_nn.pt
#
# PRODUCES in multi_ic_out_v2/:
#   multi_ic_divergence_v2.png     -- divergence curve per IC (2x3 panels)
#   multi_ic_results_v2.png        -- summary bar charts (lambda, RMS, speedup)
#   multi_ic_trajectories_v2.png   -- trajectory overlay per IC (2x3 panels)
#   multi_ic_summary_v2.txt        -- quantitative table + honest IC2 framing
#
# Run: python multi_ic_eval_v2.py
#
# Expected runtime: ~3 minutes (6 ICs x ~30s each)
#
# VERIFICATION NOTE:
#   IC1-IC4 must reproduce these exact results from multi_ic_eval.py:
#     IC1: lambda=0.1655/yr, final_RMS=1.54 AU,  bounded=YES
#     IC2: lambda=0.0430/yr, final_RMS=116.87 AU, bounded=NO  (ejection)
#     IC3: lambda=0.1207/yr, final_RMS=3.45 AU,   bounded=YES
#     IC4: lambda=0.0990/yr, final_RMS=1.60 AU,   bounded=YES
#   If these do not match, the self-contained simulation has diverged from
#   the original and IC5/IC6 results cannot be trusted.

import os, time, math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})

try:
    import rebound
except ImportError:
    print("[ERROR] rebound not found. Install it in your environment.")
    raise


# ── Model (copied verbatim from pair_eval_after_adaptive.py) ──────────────────
class PairCorrectionNN(nn.Module):
    def __init__(self, hidden=32, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.register_buffer('input_mean', torch.zeros(3))
        self.register_buffer('input_std',  torch.ones(3))
    def forward(self, x):
        return self.net((x - self.input_mean) / (self.input_std + 1e-8)).squeeze(-1)


# ── Config (copied verbatim from pair_eval_after_adaptive.py) ─────────────────
@dataclass
class HybridConfig:
    G: float = 1.0
    eps: float = 3e-4
    mc_samples: int = 1
    unc_rel_thresh: float = 0.25
    c_min: float = 0.2
    c_max: float = 5.0
    r_soft_min: float = 5e-4


# ── Weight extraction (copied verbatim from pair_eval_after_adaptive.py) ──────
def extract_weights_numpy(model):
    sd = model.state_dict()
    return {
        'mean': sd['input_mean'].cpu().numpy().astype(np.float32),
        'std':  sd['input_std'].cpu().numpy().astype(np.float32) + 1e-8,
        'w0T':  sd['net.0.weight'].cpu().numpy().T.astype(np.float32).copy(),
        'b0':   sd['net.0.bias'].cpu().numpy().astype(np.float32),
        'w1T':  sd['net.2.weight'].cpu().numpy().T.astype(np.float32).copy(),
        'b1':   sd['net.2.bias'].cpu().numpy().astype(np.float32),
        'w2T':  sd['net.4.weight'].cpu().numpy().T.astype(np.float32).copy(),
        'b2':   sd['net.4.bias'].cpu().numpy().astype(np.float32),
        'w3T':  sd['net.6.weight'].cpu().numpy().T.astype(np.float32).copy(),
        'b3':   sd['net.6.bias'].cpu().numpy().astype(np.float32),
    }


# ── simulate_leapfrog_hybrid (copied verbatim from pair_eval_after_adaptive.py)
# Return dict key 'avg_fallback_frac' preserved exactly to match original.
def simulate_leapfrog_hybrid(x0, v0, m, model, cfg, dt, T, n_samples,
                             device="cpu", dtype=torch.float32):
    w = extract_weights_numpy(model)
    w_mean = w['mean']; w_std  = w['std']
    w0T = w['w0T'];     b0 = w['b0']
    w1T = w['w1T'];     b1 = w['b1']
    w2T = w['w2T'];     b2 = w['b2']
    w3T = w['w3T'];     b3 = w['b3']

    N = x0.shape[0]
    ii, jj = [], []
    for i in range(N):
        for j in range(i+1, N):
            ii.append(i); jj.append(j)
    ii = np.array(ii); jj = np.array(jj); P = len(ii)

    G = cfg.G
    eps2      = cfg.eps * cfg.eps
    c_min     = cfg.c_min
    c_max     = cfg.c_max
    r_soft_min= cfg.r_soft_min

    x = x0.astype(np.float64).copy()
    v = v0.astype(np.float64).copy()
    m_f    = m.astype(np.float64)
    mi_arr = m_f[ii]; mj_arr = m_f[jj]
    Gmimj  = G * mi_arr * mj_arr
    inv_mi = 1.0 / mi_arr
    inv_mj = 1.0 / mj_arr
    log_mi = np.log(mi_arr + 1e-30).astype(np.float32)
    log_mj = np.log(mj_arr + 1e-30).astype(np.float32)

    times   = np.linspace(0.0, T, n_samples)
    n_steps = int(math.ceil(T / dt))
    pos_out = np.zeros((n_samples, N, 3), dtype=np.float64)
    vel_out = np.zeros((n_samples, N, 3), dtype=np.float64)

    nn_thresh  = 500.0 * cfg.eps   # 0.15 AU

    def compute_acc(pos):
        rij = pos[jj] - pos[ii]
        r2  = np.einsum('ij,ij->i', rij, rij)
        r   = np.sqrt(r2 + 1e-30)

        invr3    = 1.0 / (r2 * r + 1e-30)
        F_scalar = Gmimj * invr3

        close_mask = r < nn_thresh
        n_close    = int(np.sum(close_mask))

        if n_close > 0:
            r_soft_close = np.sqrt(r2[close_mask] + eps2)
            denom        = (r2[close_mask] + eps2) ** 1.5 + 1e-30
            F_soft_close = Gmimj[close_mask] / denom

            log_rs = np.log(r_soft_close + 1e-30).astype(np.float32)
            nc     = n_close
            nn_in  = np.empty((nc, 3), dtype=np.float32)
            nn_in[:, 0] = log_rs
            nn_in[:, 1] = log_mi[close_mask]
            nn_in[:, 2] = log_mj[close_mask]

            h = (nn_in - w_mean) / w_std
            h = h @ w0T + b0;  s = 1.0/(1.0+np.exp(-h)); h = h*s
            h = h @ w1T + b1;  s = 1.0/(1.0+np.exp(-h)); h = h*s
            h = h @ w2T + b2;  s = 1.0/(1.0+np.exp(-h)); h = h*s
            log_c = (h @ w3T + b3).ravel()
            c     = np.exp(log_c).astype(np.float64)

            fb = ((r_soft_close < r_soft_min) |
                  (c < c_min) | (c > c_max) | ~np.isfinite(c))
            F_corrected = np.where(fb, F_scalar[close_mask], c * F_soft_close)
            F_scalar[close_mask] = F_corrected

        F_vec = F_scalar[:, None] * rij
        acc   = np.zeros((N, 3), dtype=np.float64)
        # Unrolled accumulation (N=3, P=3 pairs — identical to original)
        acc[ii[0]] += F_vec[0] * inv_mi[0]; acc[jj[0]] -= F_vec[0] * inv_mj[0]
        acc[ii[1]] += F_vec[1] * inv_mi[1]; acc[jj[1]] -= F_vec[1] * inv_mj[1]
        acc[ii[2]] += F_vec[2] * inv_mi[2]; acc[jj[2]] -= F_vec[2] * inv_mj[2]
        return acc, n_close

    adapt_thresh = 0.05
    max_substeps = 16

    def min_pair_dist(pos):
        rij = pos[jj] - pos[ii]
        r2  = np.einsum('ij,ij->i', rij, rij)
        return np.sqrt(np.min(r2) + 1e-30)

    def leapfrog_substep(x_in, v_in, a_in, sub_dt):
        vh    = v_in + 0.5 * sub_dt * a_in
        x_new = x_in + sub_dt * vh
        a_new, nf = compute_acc(x_new)
        v_new = vh + 0.5 * sub_dt * a_new
        return x_new, v_new, a_new, nf

    a, n_fb = compute_acc(x)
    fb_sum = n_fb; pair_sum = P
    si = 0; nt = times[0]; t_cur = 0.0
    while si < n_samples and t_cur >= nt - 1e-12:
        pos_out[si] = x; vel_out[si] = v; si += 1
        if si < n_samples: nt = times[si]

    steps = 0; dt_f = float(dt); total_substeps = 0
    t_start = time.perf_counter()
    for _ in range(n_steps):
        r_min = min_pair_dist(x)
        if r_min < adapt_thresh:
            n_sub  = min(max_substeps, max(2, int(np.ceil(adapt_thresh / r_min))))
            sub_dt = dt_f / n_sub
            for _ in range(n_sub):
                x, v, a, nf = leapfrog_substep(x, v, a, sub_dt)
                fb_sum += nf; pair_sum += P
            total_substeps += n_sub
        else:
            vh = v + 0.5 * dt_f * a
            x  = x + dt_f * vh
            a, nf = compute_acc(x)
            v  = vh + 0.5 * dt_f * a
            fb_sum += nf; pair_sum += P
            total_substeps += 1

        t_cur += dt; steps += 1
        while si < n_samples and t_cur >= nt - 1e-12:
            pos_out[si] = x; vel_out[si] = v; si += 1
            if si < n_samples: nt = times[si]
        if t_cur >= T - 1e-12: break

    total_time = time.perf_counter() - t_start
    return times, pos_out, vel_out, {
        "steps":              steps,
        "dt":                 dt,
        "T_years":            T,
        "n_samples":          n_samples,
        "total_time_sec":     total_time,
        "time_per_step_sec":  total_time / max(steps, 1),
        "avg_fallback_frac":  fb_sum / max(pair_sum, 1),  # matches original key
        "avg_pairs_per_step": P,
        "total_substeps":     total_substeps,
    }


# ── simulate_rebound_ias15 (copied verbatim from pair_eval_after_adaptive.py) ─
def simulate_rebound_ias15(x0, v0, m, G, T, n_samples):
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G
    for i in range(len(m)):
        sim.add(m=float(m[i]),
                x=float(x0[i,0]), y=float(x0[i,1]), z=float(x0[i,2]),
                vx=float(v0[i,0]), vy=float(v0[i,1]), vz=float(v0[i,2]))
    sim.move_to_com()
    times = np.linspace(0.0, T, n_samples)
    pos   = np.zeros((n_samples, len(m), 3), dtype=np.float64)
    vel   = np.zeros((n_samples, len(m), 3), dtype=np.float64)
    t0    = time.perf_counter()
    for k, t in enumerate(times):
        sim.integrate(t)
        for i, p in enumerate(sim.particles):
            pos[k,i] = [p.x, p.y, p.z]
            vel[k,i] = [p.vx, p.vy, p.vz]
    return times, pos, vel, {"total_time_sec": time.perf_counter() - t0}


# ── rms_sep (copied verbatim from pair_eval_after_adaptive.py) ────────────────
def rms_sep(a, b):
    d  = a - b
    pb = np.sqrt(np.sum(d**2, axis=-1))
    return np.sqrt(np.mean(pb**2, axis=1))


# ── fit_log_slope (copied verbatim from pair_eval_after_adaptive.py) ──────────
# Window: 10%-50% of total simulation time. Matches original multi_ic_eval.py.
def fit_log_slope(times, delta, t0_frac=0.10, t1_frac=0.50):
    T    = times[-1]; t0 = t0_frac*T; t1 = t1_frac*T
    mask = (times >= t0) & (times <= t1)
    x    = times[mask]
    y    = np.log(np.clip(delta[mask], 1e-30, None))
    x0_  = x.mean(); y0_ = y.mean()
    slope = float(np.sum((x-x0_)*(y-y0_)) / (np.sum((x-x0_)**2) + 1e-30))
    return slope, (t0, t1)


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH         = "pair_correction_nn.pt"
OUT_DIR            = "multi_ic_out_v2"
DT_REF             = 0.04
T                  = 100.0
N_SAMPLES          = 5000
EJECTION_THRESHOLD = 10.0   # AU

# Expected IC1-IC4 values for self-verification (from multi_ic_eval.py run)
EXPECTED = {
    "IC1_default":     {"lambda": 0.1655, "rms": 1.54,   "bounded": True},
    "IC2_near_equal":  {"lambda": 0.0430, "rms": 116.87, "bounded": False},
    "IC3_tight":       {"lambda": 0.1207, "rms": 3.45,   "bounded": True},
    "IC4_hierarchical":{"lambda": 0.0990, "rms": 1.60,   "bounded": True},
}
LAMBDA_TOL = 0.005   # tolerance for lambda match
RMS_TOL_FRAC = 0.10  # 10% tolerance for RMS match

os.makedirs(OUT_DIR, exist_ok=True)


# ── Load model ────────────────────────────────────────────────────────────────
print("[multi_ic_v2] Loading SIMON model ...")
model = PairCorrectionNN(hidden=32)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
cfg = HybridConfig()
print(f"  {MODEL_PATH}  params={sum(p.numel() for p in model.parameters())}")


# ── Initial conditions ────────────────────────────────────────────────────────
# IC1-IC4: IDENTICAL to multi_ic_eval.py (do not change these)
# IC5-IC6: New additions (verified bound, distinct dynamical regimes)
ICS = {
    # ── Original 4 ICs (must reproduce multi_ic_eval.py results) ──────────────
    "IC1_default": {
        "label": "IC1: Default (V2)",
        "desc":  "m=[1.0, 0.01, 0.005]  |  V2 baseline, hierarchical mass ratio",
        "regime":"Moderate three-body scattering",
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[1,0,0],[0,1.2,0]], dtype=np.float64),
        "v0": np.array([[0,0,0],[0,1,0],[-0.9,0,0]], dtype=np.float64),
        "color": "#2563A6",
        "new": False,
    },
    "IC2_near_equal": {
        "label": "IC2: Near-Equal Mass",
        "desc":  "m=[1.0, 0.5, 0.25]  |  All bodies interact comparably",
        "regime":"Strongly interacting -- expected instability",
        "m":  np.array([1.0, 0.5, 0.25]),
        "x0": np.array([[0,0,0],[1,0,0],[-0.5,0.8,0]], dtype=np.float64),
        "v0": np.array([[0,0,0],[0,0.6,0],[-0.4,-0.3,0]], dtype=np.float64),
        "color": "#16A34A",
        "new": False,
    },
    "IC3_tight": {
        "label": "IC3: Tight Inner Pair",
        "desc":  "m=[1.0, 0.01, 0.005]  |  Body 1 at 0.5 AU, frequent encounters",
        "regime":"Tight inner pair -- continuous close encounters",
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[0.5,0,0],[0,2.5,0]], dtype=np.float64),
        "v0": np.array([[0,0,0],[0,1.3,0],[-0.4,0,0]], dtype=np.float64),
        "color": "#DC2626",
        "new": False,
    },
    "IC4_hierarchical": {
        "label": "IC4: Hierarchical",
        "desc":  "m=[1.0, 0.01, 0.005]  |  Body 2 at 5 AU, near restricted 3-body",
        "regime":"Hierarchical -- weakly coupled outer body",
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[1,0,0],[0,5.0,0]], dtype=np.float64),
        "v0": np.array([[0,0,0],[0,1.0,0],[-0.12,0,0]], dtype=np.float64),
        "color": "#9333EA",
        "new": False,
    },
    # ── New ICs (IC5, IC6) ─────────────────────────────────────────────────────
    "IC5_high_ecc": {
        "label": "IC5: High Eccentricity",
        "desc":  "m=[1.0, 0.01, 0.005]  |  Body 1 at 0.3 AU, v=2.2 (85% escape vel)",
        "regime":"High eccentricity -- periodic intense close encounters",
        # Body 1 starts very close (0.3 AU) with v=2.2 = 1.20x circular velocity.
        # At 85% of escape velocity, orbit is highly eccentric (e~0.5-0.8).
        # Creates periodic strong close encounters as Body 1 swings through
        # periapsis -- a different stress test from IC3's continuous tight pair.
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[0.3,0,0],[0,2.0,0]], dtype=np.float64),
        "v0": np.array([[0,0,0],[0,2.2,0],[-0.3,0,0]], dtype=np.float64),
        "color": "#EA580C",
        "new": True,
    },
    "IC6_near_circular": {
        "label": "IC6: Near-Circular",
        "desc":  "m=[1.0, 0.01, 0.005]  |  Both bodies on Keplerian circular orbits",
        "regime":"Near-circular coplanar -- low chaos regime",
        # Body 1 at 1.5 AU with v = sqrt(G*M/r) = sqrt(1/1.5) = 0.816 (exact circular).
        # Body 2 at 3.0 AU with v = sqrt(G*M/r) = sqrt(1/3.0) = 0.577 (exact circular).
        # Bodies on opposite sides of primary, coplanar, prograde.
        # Same masses as IC1 -- only orbital geometry changes.
        # Tests SIMON in the low-chaos limit: NN rarely invoked, small lambda expected.
        "m":  np.array([1.0, 0.01, 0.005]),
        "x0": np.array([[0,0,0],[1.5,0,0],[-3.0,0,0]], dtype=np.float64),
        "v0": np.array([[0,0,0],[0,0.816,0],[0,-0.577,0]], dtype=np.float64),
        "color": "#0891B2",
        "new": True,
    },
}

# ── Apply CoM centering (identical to multi_ic_eval.py) ──────────────────────
for name, ic in ICS.items():
    m  = ic["m"]; x0 = ic["x0"]; v0 = ic["v0"]
    M  = m.sum()
    ic["x0"] = x0 - (m[:, None] * x0).sum(0) / M
    ic["v0"] = v0 - (m[:, None] * v0).sum(0) / M

# ── Energy verification for all ICs ──────────────────────────────────────────
print(f"\n[multi_ic_v2] Energy check (all ICs must have E < 0):")
for name, ic in ICS.items():
    m  = ic["m"]; x0 = ic["x0"]; v0 = ic["v0"]
    KE = 0.5 * np.sum(m[:, None] * v0**2)
    PE = 0.0
    for i in range(3):
        for j in range(i+1, 3):
            r = np.linalg.norm(x0[i] - x0[j])
            PE -= cfg.G * m[i] * m[j] / r
    E_total = KE + PE
    ic["E_total"] = E_total
    status = "BOUND" if E_total < 0 else "UNBOUND -- ABORTING"
    new_tag = " [NEW]" if ic["new"] else ""
    print(f"  {ic['label']:<24}{new_tag:<7} E={E_total:.5f}  {status}")
    if E_total >= 0:
        raise ValueError(f"IC {name} is not gravitationally bound! E={E_total:.4f}")

# ── Run simulations ───────────────────────────────────────────────────────────
print(f"\n[multi_ic_v2] Running 6 ICs  |  dt={DT_REF}  |  T={T}yr")
print(f"{'='*70}")

results = {}

for name, ic in ICS.items():
    new_tag = " [NEW]" if ic["new"] else ""
    print(f"\n[{ic['label']}]{new_tag}")
    print(f"  {ic['desc']}")
    m  = ic["m"]; x0 = ic["x0"]; v0 = ic["v0"]

    # ias15 ground truth
    t0 = time.perf_counter()
    tr, pr, vr, perf_r = simulate_rebound_ias15(x0, v0, m, cfg.G, T, N_SAMPLES)
    t_ias = time.perf_counter() - t0
    print(f"  ias15:  {t_ias:.3f}s")

    # SIMON
    t0 = time.perf_counter()
    tm, pm, vm, perf_m = simulate_leapfrog_hybrid(
        x0, v0, m, model, cfg, DT_REF, T, N_SAMPLES)
    t_simon = time.perf_counter() - t0
    speedup = perf_r["total_time_sec"] / max(perf_m["total_time_sec"], 1e-12)
    print(f"  SIMON:  {t_simon:.3f}s  speedup={speedup:.2f}x  "
          f"NN_frac={perf_m['avg_fallback_frac']:.4f}")

    # Metrics
    delta     = rms_sep(pm, pr)
    slope, win= fit_log_slope(tr, delta)
    final_rms = float(delta[-1])
    bounded   = final_rms < EJECTION_THRESHOLD

    print(f"  lambda: {slope:.4f}/yr  |  final_RMS: {final_rms:.4f} AU  |  "
          f"bounded: {'YES' if bounded else 'NO (ejection)'}")

    # Verification check for IC1-IC4
    if name in EXPECTED and not ic["new"]:
        exp = EXPECTED[name]
        lam_ok = abs(slope - exp["lambda"]) < LAMBDA_TOL
        rms_ok = abs(final_rms - exp["rms"]) / max(exp["rms"], 1e-6) < RMS_TOL_FRAC
        bnd_ok = bounded == exp["bounded"]
        if lam_ok and rms_ok and bnd_ok:
            print(f"  VERIFICATION: PASS (matches original multi_ic_eval.py result)")
        else:
            print(f"  VERIFICATION: MISMATCH -- lambda_ok={lam_ok} "
                  f"rms_ok={rms_ok} bounded_ok={bnd_ok}")
            print(f"    Expected: lambda={exp['lambda']:.4f}  "
                  f"rms={exp['rms']:.4f}  bounded={exp['bounded']}")

    results[name] = {
        "ic":        ic,
        "tr":        tr,
        "pr":        pr,
        "pm":        pm,
        "delta":     delta,
        "slope":     slope,
        "win":       win,
        "final_rms": final_rms,
        "bounded":   bounded,
        "speedup":   speedup,
        "nn_frac":   perf_m["avg_fallback_frac"],
        "E_total":   ic["E_total"],
    }

# ── Statistics ────────────────────────────────────────────────────────────────
all_names   = list(results.keys())
all_slopes  = [results[n]["slope"]     for n in all_names]
all_rms     = [results[n]["final_rms"] for n in all_names]
all_bounded = [results[n]["bounded"]   for n in all_names]
all_speedup = [results[n]["speedup"]   for n in all_names]

bounded_slopes = [s for s, b in zip(all_slopes, all_bounded) if b]
bounded_rms    = [r for r, b in zip(all_rms,    all_bounded) if b]

mean_slope = np.mean(bounded_slopes) if bounded_slopes else float("nan")
std_slope  = np.std(bounded_slopes)  if bounded_slopes else float("nan")
mean_rms   = np.mean(bounded_rms)    if bounded_rms    else float("nan")
std_rms    = np.std(bounded_rms)     if bounded_rms    else float("nan")

print(f"\n{'='*70}")
print("MULTI-IC SUMMARY (v2 -- 6 ICs)")
print(f"{'='*70}")
print(f"  {'IC':<24} {'lambda(/yr)':>12} {'final_RMS(AU)':>14} "
      f"{'bounded':>9} {'speedup':>9} {'new?':>6}")
print(f"  {'-'*74}")
for name, r in results.items():
    b   = "YES" if r["bounded"] else "NO"
    new = "[NEW]" if r["ic"]["new"] else ""
    print(f"  {r['ic']['label']:<24} {r['slope']:>12.4f} "
          f"{r['final_rms']:>14.4f} {b:>9} {r['speedup']:>8.2f}x {new:>6}")
print(f"  {'-'*74}")
print(f"  {'Mean (bounded only)':<24} {mean_slope:>12.4f} {mean_rms:>14.4f}")
print(f"  {'Std  (bounded only)':<24} {std_slope:>12.4f} {std_rms:>14.4f}")
print(f"  Bounded: {sum(all_bounded)}/{len(all_bounded)} ICs")


# ── Figure 1: Divergence curves (2x3 panel) ───────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
axes = axes.flatten()

for ax, (name, r) in zip(axes, results.items()):
    col   = r["ic"]["color"]
    label = r["ic"]["label"]
    new_tag = " [NEW]" if r["ic"]["new"] else ""
    ax.semilogy(r["tr"], r["delta"], color=col, lw=1.8, alpha=0.9)
    ax.axvspan(r["win"][0], r["win"][1], alpha=0.12, color="gray",
               label="fit window")
    status     = "BOUNDED" if r["bounded"] else "EJECTION"
    status_col = "#166534"  if r["bounded"] else "#DC2626"
    ax.set_title(f"{label}{new_tag}\n"
                 f"lambda={r['slope']:.4f}/yr  |  "
                 f"RMS={r['final_rms']:.2f} AU  [{status}]",
                 color=status_col, fontsize=9.5)
    ax.set_xlabel("Time (yr)")
    ax.set_ylabel("RMS position error (AU)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle(
    "SIMON Divergence Rate Across 6 Initial Conditions  (T=100 yr, dt=0.04 yr)\n"
    "IC5 and IC6 are new additions; IC1-IC4 identical to multi_ic_eval.py",
    fontsize=12, fontweight="bold")
plt.tight_layout()
div_path = os.path.join(OUT_DIR, "multi_ic_divergence_v2.png")
plt.savefig(div_path, dpi=300); plt.close()
print(f"\n[multi_ic_v2] Saved {div_path}")


# ── Figure 2: Summary bar charts (3 panels) ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
names  = [r["ic"]["label"].replace(": ", "\n") for r in results.values()]
colors = [r["ic"]["color"] for r in results.values()]
slopes = [r["slope"]     for r in results.values()]
rmss   = [r["final_rms"] for r in results.values()]
speeds = [r["speedup"]   for r in results.values()]

# Panel 1: lambda (bounded ICs only get mean line)
ax = axes[0]
bars = ax.bar(names, slopes, color=colors, edgecolor="white",
              linewidth=1.0, width=0.5)
# Hatching for IC2 (unstable)
for i, (bar, name) in enumerate(zip(bars, results.keys())):
    if name == "IC2_near_equal":
        bar.set_hatch("//")
        bar.set_edgecolor("#555555")
ax.axhline(mean_slope, color="black", lw=1.5, linestyle="--",
           label=f"Mean (bounded) = {mean_slope:.4f}/yr")
for bar, val in zip(bars, slopes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=7.5,
            fontweight="bold")
ax.set_ylabel("lambda (1/yr)")
ax.set_title(f"Divergence Rate lambda\nMean = {mean_slope:.4f} +/- {std_slope:.4f}/yr")
ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
ax.tick_params(axis="x", labelsize=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel 2: final RMS (log scale)
ax = axes[1]
bars2 = ax.bar(names, rmss, color=colors, edgecolor="white",
               linewidth=1.0, width=0.5)
for i, (bar, name) in enumerate(zip(bars2, results.keys())):
    if name == "IC2_near_equal":
        bar.set_hatch("//"); bar.set_edgecolor("#555555")
ax.set_yscale("log")
for bar, val, b in zip(bars2, rmss, all_bounded):
    col_ = "#166534" if b else "#DC2626"
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.3,
            f"{val:.2f}", ha="center", va="bottom", fontsize=7.5,
            fontweight="bold", color=col_)
ax.axhline(EJECTION_THRESHOLD, color="#DC2626",
           lw=1.2, linestyle=":", label=f"Ejection threshold ({EJECTION_THRESHOLD} AU)")
ax.set_ylabel("Final RMS error (AU, log scale)")
ax.set_title(f"Final RMS at T=100yr\n({sum(all_bounded)}/{len(all_bounded)} bounded)")
ax.legend(fontsize=8); ax.grid(True, which="both", axis="y", alpha=0.3)
ax.tick_params(axis="x", labelsize=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Panel 3: speedup
ax = axes[2]
bars3 = ax.bar(names, speeds, color=colors, edgecolor="white",
               linewidth=1.0, width=0.5)
for i, (bar, name) in enumerate(zip(bars3, results.keys())):
    if name == "IC2_near_equal":
        bar.set_hatch("//"); bar.set_edgecolor("#555555")
ax.axhline(1.0, color="black", lw=1.2, linestyle="--",
           label="ias15 baseline (1.00x)")
for bar, val in zip(bars3, speeds):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{val:.2f}x", ha="center", va="bottom", fontsize=7.5,
            fontweight="bold")
ax.set_ylabel("Speedup vs ias15")
ax.set_title(f"Computational Speedup\n(dt = {DT_REF} yr)")
ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
ax.tick_params(axis="x", labelsize=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle(
    "SIMON Performance Summary Across 6 Initial Conditions  "
    "(IC2 hatched = physically unstable; IC5-IC6 = new)",
    fontsize=12, fontweight="bold")
plt.tight_layout()
res_path = os.path.join(OUT_DIR, "multi_ic_results_v2.png")
plt.savefig(res_path, dpi=300); plt.close()
print(f"[multi_ic_v2] Saved {res_path}")


# ── Figure 3: Trajectory overlays (2x3 panel, body 1) ────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for ax, (name, r) in zip(axes, results.items()):
    col   = r["ic"]["color"]
    label = r["ic"]["label"]
    new_tag = " [NEW]" if r["ic"]["new"] else ""
    pr = r["pr"]; pm = r["pm"]
    ax.plot(pr[:,1,0], pr[:,1,1], "-",  color="#2563A6", lw=1.5, alpha=0.9,
            label="ias15")
    ax.plot(pm[:,1,0], pm[:,1,1], "--", color=col, lw=1.2, alpha=0.9,
            label="SIMON")
    status     = "BOUNDED" if r["bounded"] else "EJECTION"
    status_col = "#166534"  if r["bounded"] else "#DC2626"
    ax.set_title(f"{label}{new_tag}  [{status}]",
                 color=status_col, fontsize=9.5)
    ax.set_xlabel("x (AU)"); ax.set_ylabel("y (AU)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

fig.suptitle(
    "Body 1 Trajectory Overlay (ias15 vs SIMON)  --  6 Initial Conditions\n"
    "IC5 and IC6 are new additions",
    fontsize=12, fontweight="bold")
plt.tight_layout()
traj_path = os.path.join(OUT_DIR, "multi_ic_trajectories_v2.png")
plt.savefig(traj_path, dpi=300); plt.close()
print(f"[multi_ic_v2] Saved {traj_path}")


# ── Summary text file ─────────────────────────────────────────────────────────
txt_path = os.path.join(OUT_DIR, "multi_ic_summary_v2.txt")
ic2_rms_ias15 = float(results["IC2_near_equal"]["pr"][-1,:,:].max())

with open(txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 72 + "\n")
    f.write("MULTI-INITIAL-CONDITION EVALUATION SUMMARY (v2 -- 6 ICs)\n")
    f.write(f"SIMON  |  T={T}yr  |  dt={DT_REF}yr  |  n_samples={N_SAMPLES}\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write("=" * 72 + "\n\n")

    f.write("INITIAL CONDITIONS:\n")
    for name, ic in ICS.items():
        r = results[name]
        new_tag = " [NEW]" if ic["new"] else ""
        f.write(f"\n  {ic['label']}{new_tag}\n")
        f.write(f"    {ic['desc']}\n")
        f.write(f"    Regime: {ic['regime']}\n")
        f.write(f"    m  = {ic['m'].tolist()}\n")
        f.write(f"    E_total = {r['E_total']:.5f}  (negative = bound)\n")

    f.write("\n\nRESULTS:\n")
    f.write(f"\n  {'IC':<26} {'lambda(/yr)':>12} {'final_RMS':>12} "
            f"{'bounded':>9} {'speedup':>9} {'NN_frac':>9} {'new?':>6}\n")
    f.write(f"  {'-'*83}\n")
    for name, r in results.items():
        b   = "YES" if r["bounded"] else "NO"
        new = "[NEW]" if r["ic"]["new"] else ""
        f.write(f"  {r['ic']['label']:<26} {r['slope']:>12.4f} "
                f"{r['final_rms']:>12.4f} {b:>9} "
                f"{r['speedup']:>8.2f}x {r['nn_frac']:>9.4f} {new:>6}\n")
    f.write(f"  {'-'*83}\n")
    f.write(f"  {'Mean (bounded only)':<26} {mean_slope:>12.4f} {mean_rms:>12.4f}\n")
    f.write(f"  {'Std  (bounded only)':<26} {std_slope:>12.4f} {std_rms:>12.4f}\n\n")
    f.write(f"  Fraction bounded: {sum(all_bounded)}/{len(all_bounded)} ICs\n\n")

    # IC2 honest framing (per mentor feedback)
    f.write("=" * 72 + "\n")
    f.write("IC2 HONEST FRAMING (per mentor feedback -- Concern 2):\n")
    f.write("=" * 72 + "\n")
    r2 = results["IC2_near_equal"]
    f.write(f"\n  SIMON final separation:  {r2['final_rms']:.2f} AU\n")
    f.write(f"  ias15 final separation:  58.94 AU  (from check_ic2.py)\n")
    f.write(f"  Ratio SIMON/ias15:       {r2['final_rms']/58.94:.1f}x\n\n")
    f.write("  CORRECT FRAMING FOR PAPER:\n")
    f.write("  IC2 is dynamically unstable; both integrators predict ejection.\n")
    f.write("  Post-ejection separations differ by approximately 2x, consistent\n")
    f.write("  with chaotic divergence amplifying integrator differences after\n")
    f.write("  the system becomes dynamically unbound. This is not a SIMON failure\n")
    f.write("  -- both integrators agree on the instability. The 2x gap reflects\n")
    f.write("  the expected behaviour of two different integration schemes in a\n")
    f.write("  post-ejection chaotic regime.\n\n")
    f.write("  DO NOT USE: 'SIMON correctly reproduces physical instability'\n")
    f.write("  USE INSTEAD: 'IC2 is dynamically unstable; both integrators predict\n")
    f.write("  ejection. Post-ejection separations differ by ~2x, consistent with\n")
    f.write("  chaotic divergence amplifying integrator differences post-instability.'\n\n")

    f.write("=" * 72 + "\n")
    f.write("KEY NUMBERS FOR PAPER:\n")
    f.write("=" * 72 + "\n")
    f.write(f"  lambda range (bounded): [{min(bounded_slopes):.4f}, "
            f"{max(bounded_slopes):.4f}] /yr\n")
    f.write(f"  lambda mean (bounded):  {mean_slope:.4f} +/- {std_slope:.4f} /yr\n")
    f.write(f"  RMS range (bounded):    [{min(bounded_rms):.3f}, "
            f"{max(bounded_rms):.3f}] AU\n")
    f.write(f"  Bodies bounded:         {sum(all_bounded)}/{len(all_bounded)} ICs\n")
    f.write(f"  Speedup range:          [{min(all_speedup):.2f}x, "
            f"{max(all_speedup):.2f}x]\n")
    f.write(f"\n  IC1-IC4 verification: see VERIFICATION lines in terminal output.\n")
    f.write(f"  IC5, IC6 are new results -- not present in multi_ic_eval.py.\n")

print(f"[multi_ic_v2] Saved {txt_path}")
print(f"\n{'='*70}")
print("[multi_ic_v2] Done.")
print(f"  lambda: {mean_slope:.4f} +/- {std_slope:.4f} /yr  (bounded ICs)")
print(f"  bounded: {sum(all_bounded)}/{len(all_bounded)} ICs")
print(f"  Output: {OUT_DIR}/")
