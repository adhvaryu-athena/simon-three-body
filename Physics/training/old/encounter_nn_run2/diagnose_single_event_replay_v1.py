"""
diagnose_single_event_replay_v1.py

Single-event replay diagnostic for the encounter-level surrogate.

Purpose
-------
The rollout prototype showed that the event-enriched gated surrogate fires at
IC1, dt=0.08, t≈82.56 yr, but worsens the full 100-year position rollout.
This diagnostic isolates that one event and answers:

  1. Does the surrogate improve the local 0.5-year encounter exit state
     relative to an IAS15 reference started from the exact noNN rollout state?
  2. If it improves locally, does it still push the later rollout onto a worse
     chaotic branch?
  3. If it is already worse at the 0.5-year exit, the encounter model itself is
     inaccurate for the real event.

Typical run from C:\\Aarush\\Physics\\training\\encounter_nn:

  python -B diagnose_single_event_replay_v1.py ^
      --model encounter_surrogate_v2_event_enriched_velocitysafe.pt ^
      --event-time 82.56 --dt 0.08 --window-years 0.5 --T 100 ^
      --out-dir single_event_replay_v1_t82p56

Outputs
-------
  <out-dir>/single_event_replay_summary.txt
  <out-dir>/single_event_replay_metrics.csv
  <out-dir>/single_event_replay_arrays.npz
  <out-dir>/local_window_rms.png          (if matplotlib available)
  <out-dir>/post_window_rms.png           (if matplotlib available)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import rebound
except ImportError:
    print("[ERROR] rebound not found. Install it in your environment.")
    raise

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    plt = None


# =============================================================================
# Constants matched to the revised noNN / encounter-surrogate prototype
# =============================================================================
G = 1.0
EPS = 3e-4
R_SOFT_MIN = 5e-4
ZONE1_R_GATE = float(math.sqrt(R_SOFT_MIN * R_SOFT_MIN - EPS * EPS))  # ~4e-4 AU
ADAPT_THRESH = 0.05
NN_THRESH = 500.0 * EPS  # 0.15 AU
MAX_SUBSTEPS = 16
ACTIVE_I, ACTIVE_J, THIRD_K = 1, 2, 0


# =============================================================================
# Encounter residual model -- matches train_encounter_surrogate_v2_velocitysafe_fixed2.py
# =============================================================================
class EncounterResidualMLP(nn.Module):
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


def infer_hidden_from_state_dict(sd: Dict[str, torch.Tensor]) -> int:
    key = "net.0.weight"
    if key not in sd:
        raise KeyError(f"state_dict missing {key}; this is not an EncounterResidualMLP model")
    w = sd[key]
    if w.ndim != 2 or w.shape[1] != 18:
        raise ValueError(f"Expected first layer shape (hidden,18), got {tuple(w.shape)}")
    return int(w.shape[0])


def load_surrogate_model(path: str, device: str) -> EncounterResidualMLP:
    sd = torch.load(path, map_location="cpu")
    hidden = infer_hidden_from_state_dict(sd)
    model = EncounterResidualMLP(hidden=hidden, dropout=0.0)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    print(f"[surrogate] loaded {path} | hidden={hidden} | params={sum(p.numel() for p in model.parameters())}")
    return model


# =============================================================================
# IC and physics helpers
# =============================================================================
def com_center(m: np.ndarray, x: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = float(np.sum(m))
    x = x - np.sum(m[:, None] * x, axis=0) / M
    v = v - np.sum(m[:, None] * v, axis=0) / M
    return x.astype(np.float64), v.astype(np.float64)


def get_ic(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name != "IC1":
        raise ValueError("Only --ic IC1 is implemented in this diagnostic")
    m = np.array([1.0, 0.01, 0.005], dtype=np.float64)
    x0 = np.array([[0, 0, 0], [1, 0, 0], [0, 1.2, 0]], dtype=np.float64)
    v0 = np.array([[0, 0, 0], [0, 1, 0], [-0.9, 0, 0]], dtype=np.float64)
    return (*com_center(m, x0, v0), m)


def all_finite(*arrs) -> bool:
    return all(np.all(np.isfinite(a)) for a in arrs)


def pair_distances(x: np.ndarray) -> np.ndarray:
    return np.array([
        np.linalg.norm(x[1] - x[0]),
        np.linalg.norm(x[2] - x[0]),
        np.linalg.norm(x[2] - x[1]),
    ], dtype=np.float64)


def min_pair_distance(x: np.ndarray) -> float:
    return float(np.min(pair_distances(x)))


def max_radius(x: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(x, axis=1)))


def total_energy(x: np.ndarray, v: np.ndarray, m: np.ndarray, softened_pe: bool = False) -> float:
    ke = 0.5 * float(np.sum(m[:, None] * v * v))
    pe = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            r2 = float(np.dot(x[j] - x[i], x[j] - x[i]))
            r = math.sqrt(r2 + (EPS * EPS if softened_pe else 1e-30))
            pe -= G * float(m[i]) * float(m[j]) / (r + 1e-30)
    return float(ke + pe)


def rel_energy(E: float, E0: float) -> float:
    return float(abs((E - E0) / (abs(E0) + 1e-30)))


def state_rms(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def traj_rms(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    per_body = np.sqrt(np.sum(d * d, axis=-1))
    return np.sqrt(np.mean(per_body * per_body, axis=1))


def compute_pair_velocity_features(x: np.ndarray, v: np.ndarray, m: np.ndarray,
                                   i: int = ACTIVE_I, j: int = ACTIVE_J) -> Tuple[float, float, float]:
    rij = x[j] - x[i]
    vij = v[j] - v[i]
    r = float(np.linalg.norm(rij))
    r_hat = rij / (r + 1e-30)
    v_rad = float(np.dot(vij, r_hat))
    v_tan_vec = vij - v_rad * r_hat
    v_tan = float(np.linalg.norm(v_tan_vec))
    v_scale = math.sqrt(G * (float(m[i]) + float(m[j])) / (r + 1e-30))
    return float(v_rad / (v_scale + 1e-30)), float(v_tan / (v_scale + 1e-30)), r


def make_X_rel18(x: np.ndarray, v: np.ndarray, m: np.ndarray, dt: float) -> Tuple[np.ndarray, float, float, float]:
    i, j, k = ACTIVE_I, ACTIVE_J, THIRD_K
    mi, mj = float(m[i]), float(m[j])
    pair_com_x = (mi * x[i] + mj * x[j]) / (mi + mj)
    pair_com_v = (mi * v[i] + mj * v[j]) / (mi + mj)
    rij = x[j] - x[i]
    vij = v[j] - v[i]
    r3 = x[k] - pair_com_x
    v3 = v[k] - pair_com_v
    vr, vt, r = compute_pair_velocity_features(x, v, m, i, j)
    X = np.concatenate([
        np.log(m + 1e-30),
        rij, vij, r3, v3,
        np.array([math.log(float(dt) + 1e-30), vr, vt], dtype=np.float64),
    ]).astype(np.float32)
    if X.shape != (18,):
        raise RuntimeError(f"X_rel18 bad shape: {X.shape}")
    return X, vr, vt, r


# =============================================================================
# Revised noNN dynamics
# =============================================================================
def revised_no_nn_acc(pos: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, int]]:
    acc = np.zeros((3, 3), dtype=np.float64)
    min_r = float("inf")
    counts = {"zone1": 0, "zone2": 0, "zone3": 0, "zone4": 0}
    for a in range(3):
        for b in range(a + 1, 3):
            rij = pos[b] - pos[a]
            r2 = float(np.dot(rij, rij))
            r = math.sqrt(r2 + 1e-30)
            min_r = min(min_r, r)
            Gmimj = G * float(m[a]) * float(m[b])
            if r < ZONE1_R_GATE:
                scalar = Gmimj / (r2 * r + 1e-30)
                counts["zone1"] += 1
            elif r < ADAPT_THRESH:
                scalar = Gmimj / (r2 * r + 1e-30)
                counts["zone2"] += 1
            elif r < NN_THRESH:
                scalar = Gmimj / ((r2 + EPS * EPS) ** 1.5 + 1e-30)
                counts["zone3"] += 1
            else:
                scalar = Gmimj / (r2 * r + 1e-30)
                counts["zone4"] += 1
            F = scalar * rij
            acc[a] += F / float(m[a])
            acc[b] -= F / float(m[b])
    return acc, min_r, counts


def no_nn_step(x: np.ndarray, v: np.ndarray, m: np.ndarray, step_dt: float,
               a_in: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Dict[str, int], int]:
    x = x.astype(np.float64, copy=True)
    v = v.astype(np.float64, copy=True)
    if a_in is None:
        a, _, c0 = revised_no_nn_acc(x, m)
    else:
        a = a_in.astype(np.float64, copy=True)
        c0 = {"zone1": 0, "zone2": 0, "zone3": 0, "zone4": 0}

    r_now = min_pair_distance(x)
    if r_now < ADAPT_THRESH:
        n_sub = min(MAX_SUBSTEPS, max(2, int(math.ceil(ADAPT_THRESH / max(r_now, 1e-30)))))
    else:
        n_sub = 1
    sub_dt = float(step_dt) / n_sub
    counts = dict(c0)
    min_r_seen = r_now
    for _ in range(n_sub):
        vh = v + 0.5 * sub_dt * a
        x = x + sub_dt * vh
        a, rmin, cc = revised_no_nn_acc(x, m)
        v = vh + 0.5 * sub_dt * a
        min_r_seen = min(min_r_seen, rmin)
        for k in counts:
            counts[k] += int(cc[k])
        if not all_finite(x, v):
            raise FloatingPointError("noNN step produced non-finite state")
    return x, v, a, float(min_r_seen), counts, int(n_sub)


def integrate_nonn(x0: np.ndarray, v0: np.ndarray, m: np.ndarray, dt: float,
                   duration: float, sample_times: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    x = x0.copy(); v = v0.copy(); t = 0.0
    if sample_times is None:
        sample_times = np.array([0.0, float(duration)], dtype=np.float64)
    sample_times = np.asarray(sample_times, dtype=np.float64)
    pos = np.zeros((len(sample_times), 3, 3), dtype=np.float64)
    vel = np.zeros_like(pos)
    si = 0
    stats = {"steps": 0, "substeps": 0, "min_r": min_pair_distance(x),
             "zone1": 0, "zone2": 0, "zone3": 0, "zone4": 0}

    def fill_outputs():
        nonlocal si
        while si < len(sample_times) and t >= sample_times[si] - 1e-12:
            pos[si] = x; vel[si] = v; si += 1

    fill_outputs()
    a, _, c0 = revised_no_nn_acc(x, m)
    for k in ["zone1", "zone2", "zone3", "zone4"]:
        stats[k] += c0[k]
    while t < float(duration) - 1e-14:
        step_dt = min(float(dt), float(duration) - t)
        x, v, a, rmin, cc, n_sub = no_nn_step(x, v, m, step_dt, a_in=a)
        t += step_dt
        stats["steps"] += 1
        stats["substeps"] += n_sub
        stats["min_r"] = min(stats["min_r"], rmin)
        for k in ["zone1", "zone2", "zone3", "zone4"]:
            stats[k] += cc[k]
        fill_outputs()
    while si < len(sample_times):
        pos[si] = x; vel[si] = v; si += 1
    return pos, vel, stats


def advance_nonn_to_time(x0: np.ndarray, v0: np.ndarray, m: np.ndarray, dt: float, t_target: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    pos, vel, stats = integrate_nonn(x0, v0, m, dt, t_target, np.array([0.0, t_target], dtype=np.float64))
    return pos[-1].copy(), vel[-1].copy(), stats


# =============================================================================
# IAS15 helpers with arbitrary output times
# =============================================================================
def simulate_ias15_at_times(x0: np.ndarray, v0: np.ndarray, m: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    times = np.asarray(times, dtype=np.float64)
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G
    for i in range(3):
        sim.add(m=float(m[i]),
                x=float(x0[i, 0]), y=float(x0[i, 1]), z=float(x0[i, 2]),
                vx=float(v0[i, 0]), vy=float(v0[i, 1]), vz=float(v0[i, 2]))
    sim.move_to_com()
    pos = np.zeros((len(times), 3, 3), dtype=np.float64)
    vel = np.zeros_like(pos)
    t0 = time.perf_counter()
    for idx, t in enumerate(times):
        sim.integrate(float(t))
        for i, p in enumerate(sim.particles):
            pos[idx, i] = [p.x, p.y, p.z]
            vel[idx, i] = [p.vx, p.vy, p.vz]
    return pos, vel, time.perf_counter() - t0


# =============================================================================
# Surrogate helpers
# =============================================================================
def predict_residual(model: EncounterResidualMLP, x_rel18: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    xb = torch.tensor(x_rel18.reshape(1, 18), dtype=torch.float32, device=device)
    with torch.no_grad():
        y = model(xb).detach().cpu().numpy().reshape(18).astype(np.float64)
    return y[:9].reshape(3, 3), y[9:].reshape(3, 3)


def project_com_to_reference(x_corr: np.ndarray, v_corr: np.ndarray,
                             x_ref: np.ndarray, v_ref: np.ndarray,
                             m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = float(np.sum(m))
    com_x_corr = np.sum(m[:, None] * x_corr, axis=0) / M
    com_v_corr = np.sum(m[:, None] * v_corr, axis=0) / M
    com_x_ref = np.sum(m[:, None] * x_ref, axis=0) / M
    com_v_ref = np.sum(m[:, None] * v_ref, axis=0) / M
    return x_corr - (com_x_corr - com_x_ref), v_corr - (com_v_corr - com_v_ref)


def maybe_plot(out_dir: str,
               local_times: np.ndarray, pos_ias_w: np.ndarray, vel_ias_w: np.ndarray,
               pos_no_w: np.ndarray, vel_no_w: np.ndarray,
               x_corr_exit: np.ndarray, v_corr_exit: np.ndarray,
               post_times_abs: np.ndarray,
               pos_ias_post_local: np.ndarray, vel_ias_post_local: np.ndarray,
               pos_no_post: np.ndarray, vel_no_post: np.ndarray,
               pos_corr_post: np.ndarray, vel_corr_post: np.ndarray) -> None:
    if not HAS_MPL:
        return
    # Local window: plot noNN trajectory error and single corrected exit point.
    no_pos_rms_w = traj_rms(pos_no_w, pos_ias_w)
    no_vel_rms_w = traj_rms(vel_no_w, vel_ias_w)
    corr_pos_exit = state_rms(x_corr_exit, pos_ias_w[-1])
    corr_vel_exit = state_rms(v_corr_exit, vel_ias_w[-1])

    plt.figure(figsize=(9, 5))
    plt.plot(local_times, no_pos_rms_w, label="noNN window vs local IAS15")
    plt.scatter([local_times[-1]], [corr_pos_exit], marker="x", s=90, label="surrogate-corrected exit")
    plt.xlabel("time from event start (yr)")
    plt.ylabel("RMS position error (AU)")
    plt.title("Single-event local 0.5-year replay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "local_window_rms.png"), dpi=200)
    plt.close()

    no_post_rms = traj_rms(pos_no_post, pos_ias_post_local)
    corr_post_rms = traj_rms(pos_corr_post, pos_ias_post_local)
    plt.figure(figsize=(9, 5))
    plt.plot(post_times_abs, no_post_rms, label="noNN branch vs local IAS15")
    plt.plot(post_times_abs, corr_post_rms, label="surrogate branch vs local IAS15")
    plt.xlabel("absolute time (yr)")
    plt.ylabel("RMS position error (AU)")
    plt.title("Post-window branch replay from same event state")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "post_window_rms.png"), dpi=200)
    plt.close()


# =============================================================================
# Main diagnostic
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Single-event replay diagnostic for the encounter surrogate.")
    ap.add_argument("--model", default="encounter_surrogate_v2_event_enriched_velocitysafe.pt")
    ap.add_argument("--ic", default="IC1", choices=["IC1"])
    ap.add_argument("--dt", type=float, default=0.08)
    ap.add_argument("--event-time", type=float, default=82.56)
    ap.add_argument("--window-years", type=float, default=0.5)
    ap.add_argument("--T", type=float, default=100.0)
    ap.add_argument("--n-window-samples", type=int, default=201)
    ap.add_argument("--n-post-samples", type=int, default=700)
    ap.add_argument("--energy-gate", type=float, default=0.20)
    ap.add_argument("--max-radius-gate", type=float, default=100.0)
    ap.add_argument("--out-dir", default="single_event_replay_v1_t82p56")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-com-project", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    model = load_surrogate_model(args.model, device)
    x0, v0, m = get_ic(args.ic)

    print("=" * 92)
    print("SINGLE-EVENT REPLAY DIAGNOSTIC")
    print(f"  model        : {args.model}")
    print(f"  event_time   : {args.event_time:.6f} yr")
    print(f"  dt/window/T  : {args.dt:.6f} / {args.window_years:.6f} / {args.T:.6f} yr")
    print(f"  device       : {device}")
    print(f"  out_dir      : {args.out_dir}")
    print("=" * 92)

    # 1. Reconstruct exact revised-noNN state at event time.
    print("[1/5] advancing revised noNN to event time...")
    x_evt, v_evt, stats_pre = advance_nonn_to_time(x0, v0, m, args.dt, args.event_time)
    E_evt = total_energy(x_evt, v_evt, m, softened_pe=False)
    X_rel18, vr, vt, r_pair = make_X_rel18(x_evt, v_evt, m, args.dt)
    print(f"      event features: r={r_pair:.8f} AU, vr={vr:+.6f}, vt={vt:.6f}, min_r={min_pair_distance(x_evt):.8f}")

    # 2. Local IAS15 and noNN window from exact event state.
    print("[2/5] local 0.5-year IAS15 and noNN replay...")
    local_times = np.linspace(0.0, float(args.window_years), int(args.n_window_samples))
    pos_ias_w, vel_ias_w, t_ias_w = simulate_ias15_at_times(x_evt, v_evt, m, local_times)
    pos_no_w, vel_no_w, stats_win = integrate_nonn(x_evt, v_evt, m, args.dt, args.window_years, local_times)
    x_no_exit, v_no_exit = pos_no_w[-1].copy(), vel_no_w[-1].copy()
    x_ias_exit, v_ias_exit = pos_ias_w[-1].copy(), vel_ias_w[-1].copy()

    # 3. Surrogate correction at window exit.
    print("[3/5] applying surrogate residual at window exit...")
    pred_rx, pred_rv = predict_residual(model, X_rel18, device)
    x_corr_exit = x_no_exit + pred_rx
    v_corr_exit = v_no_exit + pred_rv
    if not args.no_com_project:
        x_corr_exit, v_corr_exit = project_com_to_reference(x_corr_exit, v_corr_exit, x_no_exit, v_no_exit, m)
    E_no_exit = total_energy(x_no_exit, v_no_exit, m, softened_pe=False)
    E_corr_exit = total_energy(x_corr_exit, v_corr_exit, m, softened_pe=False)
    relE_no = rel_energy(E_no_exit, E_evt)
    relE_corr = rel_energy(E_corr_exit, E_evt)
    pred_pos_norm = state_rms(pred_rx, np.zeros_like(pred_rx))
    pred_vel_norm = state_rms(pred_rv, np.zeros_like(pred_rv))

    use_gate = True
    reason = "used"
    if not all_finite(x_corr_exit, v_corr_exit):
        use_gate = False; reason = "nonfinite"
    elif max_radius(x_corr_exit) > float(args.max_radius_gate):
        use_gate = False; reason = "max_radius_gate"
    elif relE_corr > float(args.energy_gate):
        use_gate = False; reason = "energy_gate"

    # 4. Continue noNN-exit and surrogate-exit branches after the window.
    print("[4/5] continuing post-window branches...")
    t_exit_abs = float(args.event_time) + float(args.window_years)
    post_duration = max(0.0, float(args.T) - t_exit_abs)
    post_times = np.linspace(0.0, post_duration, int(args.n_post_samples)) if post_duration > 0 else np.array([0.0])
    post_times_abs = t_exit_abs + post_times

    # Local IAS15 reference: same exact event state, sampled at window+post_times.
    ias_query_times = float(args.window_years) + post_times
    pos_ias_post_local, vel_ias_post_local, t_ias_post = simulate_ias15_at_times(x_evt, v_evt, m, ias_query_times)

    # Global IAS15 reference: original IC, sampled at absolute post times.
    pos_ias_post_global, vel_ias_post_global, t_ias_global = simulate_ias15_at_times(x0, v0, m, post_times_abs)

    pos_no_post, vel_no_post, stats_no_post = integrate_nonn(x_no_exit, v_no_exit, m, args.dt, post_duration, post_times)
    pos_corr_post, vel_corr_post, stats_corr_post = integrate_nonn(x_corr_exit, v_corr_exit, m, args.dt, post_duration, post_times)

    # 5. Metrics and output.
    print("[5/5] writing diagnostics...")
    no_pos_exit_err = state_rms(x_no_exit, x_ias_exit)
    corr_pos_exit_err = state_rms(x_corr_exit, x_ias_exit)
    no_vel_exit_err = state_rms(v_no_exit, v_ias_exit)
    corr_vel_exit_err = state_rms(v_corr_exit, v_ias_exit)

    true_rx = x_ias_exit - x_no_exit
    true_rv = v_ias_exit - v_no_exit
    pred_resid_pos_err = state_rms(pred_rx, true_rx)
    pred_resid_vel_err = state_rms(pred_rv, true_rv)
    true_pos_norm = state_rms(true_rx, np.zeros_like(true_rx))
    true_vel_norm = state_rms(true_rv, np.zeros_like(true_rv))

    no_post_pos_local = traj_rms(pos_no_post, pos_ias_post_local)
    corr_post_pos_local = traj_rms(pos_corr_post, pos_ias_post_local)
    no_post_vel_local = traj_rms(vel_no_post, vel_ias_post_local)
    corr_post_vel_local = traj_rms(vel_corr_post, vel_ias_post_local)

    no_post_pos_global = traj_rms(pos_no_post, pos_ias_post_global)
    corr_post_pos_global = traj_rms(pos_corr_post, pos_ias_post_global)
    no_post_vel_global = traj_rms(vel_no_post, vel_ias_post_global)
    corr_post_vel_global = traj_rms(vel_corr_post, vel_ias_post_global)

    def timeavg(arr: np.ndarray) -> float:
        return float(np.sqrt(np.mean(arr * arr)))

    metrics = {
        "event_time": float(args.event_time),
        "event_r_pair": float(r_pair),
        "event_v_rad_norm": float(vr),
        "event_v_tan_norm": float(vt),
        "event_min_pair_distance": float(min_pair_distance(x_evt)),
        "window_min_r_ias15": float(np.min([min_pair_distance(p) for p in pos_ias_w])),
        "window_min_r_nonn": float(stats_win["min_r"]),
        "pred_pos_norm": float(pred_pos_norm),
        "pred_vel_norm": float(pred_vel_norm),
        "true_resid_pos_norm": float(true_pos_norm),
        "true_resid_vel_norm": float(true_vel_norm),
        "pred_resid_pos_error": float(pred_resid_pos_err),
        "pred_resid_vel_error": float(pred_resid_vel_err),
        "relE_nonn_exit": float(relE_no),
        "relE_corr_exit": float(relE_corr),
        "accepted_by_energy_gate": int(use_gate),
        "accept_reason": reason,
        "local_exit_noNN_pos_err": float(no_pos_exit_err),
        "local_exit_corr_pos_err": float(corr_pos_exit_err),
        "local_exit_noNN_vel_err": float(no_vel_exit_err),
        "local_exit_corr_vel_err": float(corr_vel_exit_err),
        "local_exit_pos_gain_pct": float(100.0 * (no_pos_exit_err - corr_pos_exit_err) / (no_pos_exit_err + 1e-30)),
        "local_exit_vel_gain_pct": float(100.0 * (no_vel_exit_err - corr_vel_exit_err) / (no_vel_exit_err + 1e-30)),
        "post_local_noNN_pos_timeavg": timeavg(no_post_pos_local),
        "post_local_corr_pos_timeavg": timeavg(corr_post_pos_local),
        "post_local_noNN_vel_timeavg": timeavg(no_post_vel_local),
        "post_local_corr_vel_timeavg": timeavg(corr_post_vel_local),
        "post_global_noNN_pos_timeavg": timeavg(no_post_pos_global),
        "post_global_corr_pos_timeavg": timeavg(corr_post_pos_global),
        "post_global_noNN_vel_timeavg": timeavg(no_post_vel_global),
        "post_global_corr_vel_timeavg": timeavg(corr_post_vel_global),
        "post_local_final_noNN_pos_err": float(no_post_pos_local[-1]),
        "post_local_final_corr_pos_err": float(corr_post_pos_local[-1]),
        "post_global_final_noNN_pos_err": float(no_post_pos_global[-1]),
        "post_global_final_corr_pos_err": float(corr_post_pos_global[-1]),
    }

    metrics_path = os.path.join(args.out_dir, "single_event_replay_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])

    # Human-readable summary.
    local_pos_better = corr_pos_exit_err < no_pos_exit_err
    local_vel_better = corr_vel_exit_err < no_vel_exit_err
    post_local_pos_better = metrics["post_local_corr_pos_timeavg"] < metrics["post_local_noNN_pos_timeavg"]
    post_global_pos_better = metrics["post_global_corr_pos_timeavg"] < metrics["post_global_noNN_pos_timeavg"]

    if (not local_pos_better) or (not local_vel_better):
        verdict = "LOCAL MODEL ISSUE: surrogate is worse already at 0.5-year encounter exit."
    elif local_pos_better and local_vel_better and not post_global_pos_better:
        verdict = "CHAOTIC BRANCH ISSUE: surrogate improves local exit but worsens later global rollout."
    elif local_pos_better and local_vel_better and post_global_pos_better:
        verdict = "PROMISING: surrogate improves local exit and later global branch in this replay."
    else:
        verdict = "MIXED: inspect local/post metrics."

    summary_lines: List[str] = []
    add = summary_lines.append
    add("Single-event replay diagnostic summary")
    add("=" * 88)
    add(f"model              : {args.model}")
    add(f"ic                 : {args.ic}")
    add(f"event_time         : {args.event_time:.6f} yr")
    add(f"dt/window/T        : {args.dt:.6f} / {args.window_years:.6f} / {args.T:.6f} yr")
    add(f"device             : {device}")
    add(f"COM projection     : {not args.no_com_project}")
    add("")
    add("Event features")
    add("-" * 88)
    add(f"r_pair             : {r_pair:.8f} AU")
    add(f"v_rad_norm         : {vr:+.8f}")
    add(f"v_tan_norm         : {vt:.8f}")
    add(f"min_r at event     : {min_pair_distance(x_evt):.8f} AU")
    add(f"window min_r IAS15 : {metrics['window_min_r_ias15']:.8f} AU")
    add(f"window min_r noNN  : {metrics['window_min_r_nonn']:.8f} AU")
    add("")
    add("Surrogate residual and safety")
    add("-" * 88)
    add(f"true residual pos RMS : {true_pos_norm:.8e}")
    add(f"pred residual pos RMS : {pred_pos_norm:.8e}")
    add(f"residual pos error    : {pred_resid_pos_err:.8e}")
    add(f"true residual vel RMS : {true_vel_norm:.8e}")
    add(f"pred residual vel RMS : {pred_vel_norm:.8e}")
    add(f"residual vel error    : {pred_resid_vel_err:.8e}")
    add(f"relE noNN exit        : {relE_no:.8e}")
    add(f"relE corrected exit   : {relE_corr:.8e}")
    add(f"accepted by safety    : {bool(use_gate)} ({reason})")
    add("")
    add("Local 0.5-year window exit vs local IAS15 from exact event state")
    add("-" * 88)
    add(f"noNN pos exit error       : {no_pos_exit_err:.8e}")
    add(f"surrogate pos exit error  : {corr_pos_exit_err:.8e}")
    add(f"local pos gain            : {metrics['local_exit_pos_gain_pct']:+.2f}%")
    add(f"noNN vel exit error       : {no_vel_exit_err:.8e}")
    add(f"surrogate vel exit error  : {corr_vel_exit_err:.8e}")
    add(f"local vel gain            : {metrics['local_exit_vel_gain_pct']:+.2f}%")
    add("")
    add("Post-window branch replay to T")
    add("-" * 88)
    add("Against LOCAL IAS15 branch from exact event state:")
    add(f"  noNN pos timeavg        : {metrics['post_local_noNN_pos_timeavg']:.8e}")
    add(f"  surrogate pos timeavg   : {metrics['post_local_corr_pos_timeavg']:.8e}")
    add(f"  noNN vel timeavg        : {metrics['post_local_noNN_vel_timeavg']:.8e}")
    add(f"  surrogate vel timeavg   : {metrics['post_local_corr_vel_timeavg']:.8e}")
    add("Against GLOBAL IAS15 branch from original IC:")
    add(f"  noNN pos timeavg        : {metrics['post_global_noNN_pos_timeavg']:.8e}")
    add(f"  surrogate pos timeavg   : {metrics['post_global_corr_pos_timeavg']:.8e}")
    add(f"  noNN vel timeavg        : {metrics['post_global_noNN_vel_timeavg']:.8e}")
    add(f"  surrogate vel timeavg   : {metrics['post_global_corr_vel_timeavg']:.8e}")
    add("")
    add("Decision")
    add("-" * 88)
    add(f"local pos improved?       : {local_pos_better}")
    add(f"local vel improved?       : {local_vel_better}")
    add(f"post local pos improved?  : {post_local_pos_better}")
    add(f"post global pos improved? : {post_global_pos_better}")
    add(f"VERDICT: {verdict}")

    summary_path = os.path.join(args.out_dir, "single_event_replay_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))

    np.savez_compressed(
        os.path.join(args.out_dir, "single_event_replay_arrays.npz"),
        local_times=local_times,
        pos_ias_w=pos_ias_w, vel_ias_w=vel_ias_w,
        pos_no_w=pos_no_w, vel_no_w=vel_no_w,
        x_corr_exit=x_corr_exit, v_corr_exit=v_corr_exit,
        post_times_abs=post_times_abs,
        pos_ias_post_local=pos_ias_post_local, vel_ias_post_local=vel_ias_post_local,
        pos_ias_post_global=pos_ias_post_global, vel_ias_post_global=vel_ias_post_global,
        pos_no_post=pos_no_post, vel_no_post=vel_no_post,
        pos_corr_post=pos_corr_post, vel_corr_post=vel_corr_post,
    )

    maybe_plot(args.out_dir,
               local_times, pos_ias_w, vel_ias_w,
               pos_no_w, vel_no_w,
               x_corr_exit, v_corr_exit,
               post_times_abs, pos_ias_post_local, vel_ias_post_local,
               pos_no_post, vel_no_post,
               pos_corr_post, vel_corr_post)

    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
