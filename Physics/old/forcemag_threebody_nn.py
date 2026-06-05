# ============================
# evaluate_model.py
# Speed–accuracy sweep + Lyapunov-style divergence
# Baseline: REBOUND IAS15 (true gravity)
# NN run:   REBOUND WHFast + TorchScript force-mag NN via additional_forces
#
# Files expected in same folder:
#   force_mag_threebody_nn_scripted.pt   (TorchScript model)
#
# Output:
#   PNG plots + printed summary
# ============================

import time
import math
import numpy as np
import torch
import rebound
import matplotlib.pyplot as plt


# ----------------------------
# User-configurable settings
# ----------------------------
MODEL_PATH = "force_mag_threebody_nn_scripted.pt"

# Simulation horizon + sampling
HORIZON_YEARS = 100.0
NSAMPLES = 5000
TIMEGRID = np.linspace(0.0, HORIZON_YEARS, NSAMPLES)

# Softening used in the NN distance calc (should match training convention)
EPS_SOFT = 3e-4

# Initial conditions (choose something non-trivial but stable-ish)
# Units are "code units" (G=1 implicit in REBOUND if you don't modify).
# You can replace these with your paper's setup.
BODIES = [
    # m,  x,   y,   z,  vx,  vy,  vz
    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # body 0 (central)
    (0.01, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0), # body 1
    (0.005, 0.0, 1.3, 0.0, -0.9, 0.0, 0.0) # body 2
]

# Sweep over WHFast timesteps (dt) for speed–accuracy curve
WHFAST_DTS = [0.005, 0.01, 0.02, 0.04, 0.08]

# Lyapunov perturbation magnitude (in position space)
LLE_PERTURB = 1e-9

# Shadowing threshold (for "shadowing time")
SHADOW_THRESH = 1.0  # RMS position error threshold


# ----------------------------
# Timing / phase logger
# ----------------------------
_T0 = time.perf_counter()
_LAST = _T0

def log_phase(msg: str):
    global _LAST
    now = time.perf_counter()
    print(f"[t+{now-_T0:8.3f}s] {msg} (prev phase took {now-_LAST:0.3f}s)")
    _LAST = now


# ----------------------------
# Helpers: build sim, record state
# ----------------------------
def build_sim(integrator: str, dt: float | None = None) -> rebound.Simulation:
    sim = rebound.Simulation()
    sim.integrator = integrator
    if dt is not None:
        sim.dt = float(dt)

    # Add bodies
    for (m, x, y, z, vx, vy, vz) in BODIES:
        sim.add(m=m, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)

    sim.move_to_com()
    return sim


def record_positions(sim: rebound.Simulation) -> np.ndarray:
    """Return (N,3) positions float64."""
    ps = sim.particles
    N = sim.N
    out = np.empty((N, 3), dtype=np.float64)
    for i in range(N):
        out[i, 0] = ps[i].x
        out[i, 1] = ps[i].y
        out[i, 2] = ps[i].z
    return out


def record_masses(sim: rebound.Simulation) -> np.ndarray:
    ps = sim.particles
    N = sim.N
    m = np.empty((N,), dtype=np.float64)
    for i in range(N):
        m[i] = ps[i].m
    return m


# ----------------------------
# NN force callback (force-mag TorchScript)
# ----------------------------
@torch.no_grad()
def forces_from_scripted_forcemag(scripted_model, pos_np: np.ndarray, m_np: np.ndarray,
                                 eps_soft: float = EPS_SOFT) -> np.ndarray:
    """
    Compute total forces on each body from NN-predicted pairwise force magnitudes.

    pos_np: (N,3) float64
    m_np:   (N,)  float64
    returns F: (N,3) float64
    """
    N = pos_np.shape[0]
    ii, jj = np.triu_indices(N, k=1)
    P = ii.size

    rij = pos_np[jj] - pos_np[ii]                      # (P,3)
    r2 = np.sum(rij * rij, axis=1) + (eps_soft**2)      # (P,)
    r = np.sqrt(r2)                                     # (P,)

    r_t  = torch.from_numpy(r.astype(np.float32))              # (P,)
    mi_t = torch.from_numpy(m_np[ii].astype(np.float32))       # (P,)
    mj_t = torch.from_numpy(m_np[jj].astype(np.float32))       # (P,)

    # Call scripted model DIRECTLY: forward(r, mi, mj)
    fmag_t = scripted_model(r_t, mi_t, mj_t)                   # (P,)
    fmag = fmag_t.cpu().numpy().astype(np.float64)             # (P,)

    inv_r = 1.0 / r
    fvec = (fmag * inv_r)[:, None] * rij                       # (P,3)

    F = np.zeros((N, 3), dtype=np.float64)
    np.add.at(F, ii, +fvec)
    np.add.at(F, jj, -fvec)

    if not np.isfinite(F).all():
        raise RuntimeError("Non-finite NN forces produced (NaN/Inf).")

    return F


def make_nn_additional_forces(scripted_model, eps_soft: float = EPS_SOFT):
    scripted_model.eval()

    def additional_forces(sim):
        ps = sim.contents.particles
        N = sim.contents.N

        pos = np.empty((N, 3), dtype=np.float64)
        m = np.empty((N,), dtype=np.float64)
        for i in range(N):
            pos[i, 0] = ps[i].x
            pos[i, 1] = ps[i].y
            pos[i, 2] = ps[i].z
            m[i] = ps[i].m

        F = forces_from_scripted_forcemag(scripted_model, pos, m, eps_soft=eps_soft)

        for i in range(N):
            invm = 1.0 / m[i]
            ps[i].ax += F[i, 0] * invm
            ps[i].ay += F[i, 1] * invm
            ps[i].az += F[i, 2] * invm

    return additional_forces


# ----------------------------
# Metrics
# ----------------------------
def rms_position_error(traj_a: np.ndarray, traj_b: np.ndarray) -> np.ndarray:
    """
    traj_*: (T,N,3)
    returns: (T,) RMS over bodies and xyz
    """
    diff = traj_a - traj_b
    per_body = np.sqrt(np.sum(diff * diff, axis=2))     # (T,N)
    rms = np.sqrt(np.mean(per_body * per_body, axis=1)) # (T,)
    return rms


def shadowing_time(rms_err: np.ndarray, tgrid: np.ndarray, thresh: float = SHADOW_THRESH) -> float:
    idx = np.argmax(rms_err > thresh)
    if rms_err[idx] <= thresh:
        return float(tgrid[-1])
    return float(tgrid[idx])


def estimate_lle_from_divergence(delta: np.ndarray, tgrid: np.ndarray,
                                fit_start: float = 0.0, fit_end: float | None = None) -> float:
    """
    Simple Lyapunov-style estimate:
      fit log(delta(t)) ~ a + lambda * t over a window.
    delta: (T,) separation measure (RMS position difference)
    """
    if fit_end is None:
        fit_end = float(tgrid[-1])

    mask = (tgrid >= fit_start) & (tgrid <= fit_end) & np.isfinite(delta) & (delta > 0)
    tt = tgrid[mask]
    yy = np.log(delta[mask])

    if tt.size < 10:
        return float("nan")

    # Linear regression
    A = np.vstack([tt, np.ones_like(tt)]).T
    lam, _ = np.linalg.lstsq(A, yy, rcond=None)[0]
    return float(lam)


# ----------------------------
# Run a sim and record trajectory
# ----------------------------
def integrate_and_record(sim: rebound.Simulation, tgrid: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Integrate sim to each t in tgrid and record positions.
    Returns:
      traj: (T,N,3)
      runtime_sec
    """
    N = sim.N
    T = tgrid.size
    traj = np.empty((T, N, 3), dtype=np.float64)

    start = time.perf_counter()
    for k, t in enumerate(tgrid):
        sim.integrate(float(t))
        traj[k] = record_positions(sim)
    runtime = time.perf_counter() - start
    return traj, float(runtime)


# ----------------------------
# Main sweep
# ----------------------------
def main():
    log_phase("=== evaluate_model.py starting ===")

    log_phase("--- phase: load TorchScript model ---")
    scripted = torch.jit.load(MODEL_PATH, map_location="cpu")
    scripted.eval()
    print(f"Loaded model: {MODEL_PATH}")

    log_phase("--- phase: build baseline sim (IAS15) ---")
    base_sim = build_sim("ias15", dt=None)

    log_phase("--- phase: integrate + record baseline trajectory (IAS15) ---")
    base_traj, base_time = integrate_and_record(base_sim, TIMEGRID)
    base_throughput = NSAMPLES / base_time
    print(f"Baseline IAS15 total time: {base_time:.3f}s | throughput: {base_throughput:.1f} samples/s")

    # Baseline LLE reference: IAS15 vs IAS15 perturbed
    log_phase("--- phase: baseline Lyapunov run (IAS15 vs IAS15 perturbed) ---")
    base_sim2 = build_sim("ias15", dt=None)
    # Apply small perturbation to body 1 position in x
    base_sim2.particles[1].x += LLE_PERTURB
    base_traj2, base_time2 = integrate_and_record(base_sim2, TIMEGRID)

    base_sep = rms_position_error(base_traj, base_traj2)
    base_lle = estimate_lle_from_divergence(base_sep, TIMEGRID, fit_start=0.0, fit_end=HORIZON_YEARS)
    print(f"Baseline IAS15 LLE-style slope over {HORIZON_YEARS:.1f} years: {base_lle:.6e} 1/yr")

    # Sweep results containers
    sweep_dt = []
    sweep_runtime = []
    sweep_throughput = []
    sweep_final_rms = []
    sweep_shadow = []
    sweep_lle = []

    # Also store one representative NN run for plotting time-series overlays
    rep_dt = WHFAST_DTS[len(WHFAST_DTS)//2]
    rep_nn_traj = None
    rep_rms = None
    rep_sep = None

    for dt in WHFAST_DTS:
        log_phase(f"--- phase: NN run build (WHFast dt={dt}) ---")
        nn_sim = build_sim("whfast", dt=dt)
        nn_sim.additional_forces = make_nn_additional_forces(scripted, eps_soft=EPS_SOFT)
        nn_sim.force_is_velocity_dependent = False

        log_phase(f"--- phase: integrate + record NN trajectory (WHFast+NN, dt={dt}) ---")
        nn_traj, nn_time = integrate_and_record(nn_sim, TIMEGRID)
        nn_throughput = NSAMPLES / nn_time

        rms = rms_position_error(nn_traj, base_traj)
        shad = shadowing_time(rms, TIMEGRID, thresh=SHADOW_THRESH)

        # Lyapunov-style for NN: (WHFast+NN) vs (WHFast+NN perturbed)
        log_phase(f"--- phase: NN Lyapunov run (WHFast+NN dt={dt}) ---")
        nn_sim2 = build_sim("whfast", dt=dt)
        nn_sim2.additional_forces = make_nn_additional_forces(scripted, eps_soft=EPS_SOFT)
        nn_sim2.force_is_velocity_dependent = False
        nn_sim2.particles[1].x += LLE_PERTURB

        nn_traj2, _ = integrate_and_record(nn_sim2, TIMEGRID)
        nn_sep = rms_position_error(nn_traj, nn_traj2)
        nn_lle = estimate_lle_from_divergence(nn_sep, TIMEGRID, fit_start=0.0, fit_end=HORIZON_YEARS)

        print(
            f"[dt={dt:0.4f}] NN runtime: {nn_time:.3f}s | thrpt: {nn_throughput:.1f}/s | "
            f"final RMS: {rms[-1]:.4f} | shadowing@{SHADOW_THRESH}: {shad:.2f} yr | "
            f"LLE-slope: {nn_lle:.3e} 1/yr"
        )

        sweep_dt.append(dt)
        sweep_runtime.append(nn_time)
        sweep_throughput.append(nn_throughput)
        sweep_final_rms.append(float(rms[-1]))
        sweep_shadow.append(shad)
        sweep_lle.append(nn_lle)

        if abs(dt - rep_dt) < 1e-12:
            rep_nn_traj = nn_traj
            rep_rms = rms
            rep_sep = nn_sep

    # ----------------------------
    # Plot 1: speed–accuracy curve
    # ----------------------------
    log_phase("--- phase: plotting speed–accuracy curve ---")
    plt.figure()
    plt.scatter(sweep_throughput, sweep_final_rms)
    for thr, err, dt in zip(sweep_throughput, sweep_final_rms, sweep_dt):
        plt.annotate(f"dt={dt}", (thr, err), textcoords="offset points", xytext=(5, 5))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Throughput (recorded samples per second, log scale)")
    plt.ylabel(f"Final RMS position deviation vs IAS15 at T={HORIZON_YEARS:.1f} yr (log scale)")
    plt.title(f"Speed–Accuracy Tradeoff (NN=WHFast+NN, baseline=IAS15), horizon={HORIZON_YEARS:.1f} years")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("speed_accuracy_tradeoff_curve.png", dpi=200)

    # ----------------------------
    # Plot 2: overlay representative trajectories (XY)
    # ----------------------------
    log_phase("--- phase: plotting trajectory overlay (representative dt) ---")
    if rep_nn_traj is None:
        rep_nn_traj = base_traj  # fallback (shouldn't happen)
        rep_rms = np.zeros_like(TIMEGRID)
    plt.figure(figsize=(7, 7))
    # plot each body in XY for both runs
    for i in range(base_traj.shape[1]):
        plt.plot(base_traj[:, i, 0], base_traj[:, i, 1], label=f"IAS15 body {i}")
        plt.plot(rep_nn_traj[:, i, 0], rep_nn_traj[:, i, 1], "--", label=f"NN+WHFast body {i} (dt={rep_dt})")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Trajectory Overlay (XY), horizon={HORIZON_YEARS:.1f} years\nBaseline=IAS15, NN=WHFast+NN (dt={rep_dt})")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("trajectory_overlay_xy.png", dpi=200)

    # ----------------------------
    # Plot 3: RMS deviation vs time (representative dt)
    # ----------------------------
    log_phase("--- phase: plotting RMS deviation vs time ---")
    plt.figure()
    plt.plot(TIMEGRID, rep_rms)
    plt.xlabel("time (years)")
    plt.ylabel("RMS position deviation (NN vs IAS15)")
    plt.title(f"Trajectory Deviation vs Time, horizon={HORIZON_YEARS:.1f} years (rep dt={rep_dt})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("rms_deviation_vs_time.png", dpi=200)

    # ----------------------------
    # Plot 4: Lyapunov-style separation curves (baseline + representative NN)
    # ----------------------------
    log_phase("--- phase: plotting Lyapunov-style separation curves ---")
    plt.figure()
    plt.semilogy(TIMEGRID, base_sep, label=f"Baseline IAS15 vs perturbed (LLE~{base_lle:.2e} 1/yr)")
    if rep_sep is not None:
        rep_lle = estimate_lle_from_divergence(rep_sep, TIMEGRID, 0.0, HORIZON_YEARS)
        plt.semilogy(TIMEGRID, rep_sep, label=f"NN (WHFast+NN dt={rep_dt}) vs perturbed (LLE~{rep_lle:.2e} 1/yr)")
    plt.xlabel("time (years)")
    plt.ylabel("Separation δ(t) (RMS position), log scale")
    plt.title(f"Lyapunov-style divergence proxy, horizon={HORIZON_YEARS:.1f} years")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("lyapunov_divergence_proxy.png", dpi=200)

    # ----------------------------
    # Summary printout
    # ----------------------------
    log_phase("--- phase: summary ---")
    speedup_best = (base_time / min(sweep_runtime)) if len(sweep_runtime) else float("nan")
    print("\n================ SUMMARY ================")
    print(f"Horizon: {HORIZON_YEARS:.1f} years | Samples: {NSAMPLES}")
    print(f"Baseline IAS15: {base_time:.3f}s total | {base_throughput:.1f} samples/s")
    print(f"Baseline LLE-style slope: {base_lle:.6e} 1/yr (using δ(t) from perturbed run)")

    print("\nNN sweep (WHFast+NN):")
    for dt, rt, thr, ferr, sh, ll in zip(sweep_dt, sweep_runtime, sweep_throughput, sweep_final_rms, sweep_shadow, sweep_lle):
        print(f"  dt={dt:0.4f} | time={rt:.3f}s | thrpt={thr:.1f}/s | final RMS={ferr:.4f} | shadow={sh:.2f} yr | LLE~{ll:.3e} 1/yr")

    print(f"\nBest-case measured speedup (IAS15 / fastest NN run): {speedup_best:.2f}x")
    print("Saved plots:")
    print("  - speed_accuracy_tradeoff_curve.png")
    print("  - trajectory_overlay_xy.png")
    print("  - rms_deviation_vs_time.png")
    print("  - lyapunov_divergence_proxy.png")
    print("========================================\n")


if __name__ == "__main__":
    main()
