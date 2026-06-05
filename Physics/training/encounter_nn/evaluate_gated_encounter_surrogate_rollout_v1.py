"""
evaluate_gated_encounter_surrogate_rollout_v1.py

Prototype rollout evaluator for the dt=0.08 encounter-level surrogate.

Purpose
-------
This script integrates the gated encounter-level surrogate into an actual SIMON-style
rollout prototype. It compares three methods on the same IC:

  1. IAS15 reference
  2. revised noNN baseline
       Zone 1/2: direct Newtonian
       Zone 3: softened force with c=1
       Zone 4: direct Newtonian
  3. gated encounter surrogate
       same revised noNN baseline, except when active pair 1-2 is in Zone 3
       and v_rad_norm < gate threshold; then a 0.5 yr noNN encounter window is
       advanced and the trained NN residual is applied at the window exit.

This is intentionally a prototype. The surrogate predicts only the encounter-exit
state, not the intermediate trajectory inside the window. During the window, the
script stores noNN intermediate states for diagnostic sampling and applies the NN
correction only at the window exit.

Typical run from C:\\Aarush\\Physics\\training\\encounter_nn:

  python -B evaluate_gated_encounter_surrogate_rollout_v1.py \
      --model encounter_surrogate_v2_velocitysafe_large1.pt \
      --T 90 --dt 0.08 --window-years 0.5 --vr-thresh -0.40 \
      --out-dir rollout_surrogate_v1_T90

Outputs
-------
  <out-dir>/rollout_summary.txt
  <out-dir>/surrogate_events.csv
  <out-dir>/rms_vs_time.png
  <out-dir>/radius_vs_time.png
  <out-dir>/rollout_arrays.npz
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
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
# Constants matched to the current revised SIMON/encounter-surrogate setup
# =============================================================================
G = 1.0
EPS = 3e-4
R_SOFT_MIN = 5e-4
ZONE1_R_GATE = float(math.sqrt(R_SOFT_MIN * R_SOFT_MIN - EPS * EPS))  # ~4e-4 AU
ADAPT_THRESH = 0.05
NN_THRESH = 500.0 * EPS  # 0.15 AU
MAX_SUBSTEPS = 16
ACTIVE_I, ACTIVE_J, THIRD_K = 1, 2, 0  # the dt=0.08 IC1 audited pair was 1-2


# =============================================================================
# Model definition -- must match train_encounter_surrogate_v2_velocitysafe_fixed2.py
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
# Initial conditions and helper physics
# =============================================================================
def com_center(m: np.ndarray, x: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = float(np.sum(m))
    x = x - np.sum(m[:, None] * x, axis=0) / M
    v = v - np.sum(m[:, None] * v, axis=0) / M
    return x.astype(np.float64), v.astype(np.float64)


def get_ic(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name != "IC1":
        raise ValueError("Only --ic IC1 is implemented in this prototype")
    m = np.array([1.0, 0.01, 0.005], dtype=np.float64)
    x0 = np.array([[0, 0, 0], [1, 0, 0], [0, 1.2, 0]], dtype=np.float64)
    v0 = np.array([[0, 0, 0], [0, 1, 0], [-0.9, 0, 0]], dtype=np.float64)
    x0, v0 = com_center(m, x0, v0)
    return x0, v0, m


def all_finite(*arrs) -> bool:
    for a in arrs:
        if not np.all(np.isfinite(a)):
            return False
    return True


def max_radius(x: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(x, axis=1)))


def pair_distances(x: np.ndarray) -> np.ndarray:
    return np.array([
        np.linalg.norm(x[1] - x[0]),
        np.linalg.norm(x[2] - x[0]),
        np.linalg.norm(x[2] - x[1]),
    ], dtype=np.float64)


def min_pair_distance(x: np.ndarray) -> float:
    return float(np.min(pair_distances(x)))


def total_energy(x: np.ndarray, v: np.ndarray, m: np.ndarray, softened_pe: bool = False) -> float:
    ke = 0.5 * float(np.sum(m[:, None] * v * v))
    pe = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            r2 = float(np.dot(x[j] - x[i], x[j] - x[i]))
            if softened_pe:
                r = math.sqrt(r2 + EPS * EPS)
            else:
                r = math.sqrt(r2 + 1e-30)
            pe -= G * float(m[i]) * float(m[j]) / (r + 1e-30)
    return float(ke + pe)


def rms_sep(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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
# Revised noNN acceleration and stepping
# =============================================================================
def revised_no_nn_acc(pos: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, int]]:
    """
    Revised no-Zone-3-NN acceleration:
      Zone 1/2 r < 0.05: direct Newtonian
      Zone 3 0.05 <= r < 0.15: softened force with c=1
      Zone 4 r >= 0.15: direct Newtonian
    """
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
    """One exact step_dt update, with adaptive substeps if current state is in Zone 2."""
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


# =============================================================================
# IAS15 reference
# =============================================================================
def simulate_ias15(x0: np.ndarray, v0: np.ndarray, m: np.ndarray, T: float, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = G
    for i in range(3):
        sim.add(m=float(m[i]),
                x=float(x0[i, 0]), y=float(x0[i, 1]), z=float(x0[i, 2]),
                vx=float(v0[i, 0]), vy=float(v0[i, 1]), vz=float(v0[i, 2]))
    sim.move_to_com()
    times = np.linspace(0.0, float(T), int(n_samples))
    pos = np.zeros((len(times), 3, 3), dtype=np.float64)
    vel = np.zeros_like(pos)
    t0 = time.perf_counter()
    for k, t in enumerate(times):
        sim.integrate(float(t))
        for i, p in enumerate(sim.particles):
            pos[k, i] = [p.x, p.y, p.z]
            vel[k, i] = [p.vx, p.vy, p.vz]
    return times, pos, vel, time.perf_counter() - t0


# =============================================================================
# Rollout simulators
# =============================================================================
def simulate_revised_nonn_rollout(x0: np.ndarray, v0: np.ndarray, m: np.ndarray,
                                  dt: float, T: float, sample_times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    x = x0.copy(); v = v0.copy(); t = 0.0
    pos = np.zeros((len(sample_times), 3, 3), dtype=np.float64)
    vel = np.zeros_like(pos)
    si = 0
    stats = {"steps": 0, "substeps": 0, "min_r": min_pair_distance(x), "max_radius": max_radius(x),
             "zone1": 0, "zone2": 0, "zone3": 0, "zone4": 0}

    def fill_outputs():
        nonlocal si
        while si < len(sample_times) and t >= sample_times[si] - 1e-12:
            pos[si] = x; vel[si] = v; si += 1

    fill_outputs()
    a, _, c0 = revised_no_nn_acc(x, m)
    for k in ["zone1", "zone2", "zone3", "zone4"]:
        stats[k] += c0[k]

    while t < float(T) - 1e-14:
        step_dt = min(float(dt), float(T) - t)
        x, v, a, rmin, cc, n_sub = no_nn_step(x, v, m, step_dt, a_in=a)
        t += step_dt
        stats["steps"] += 1
        stats["substeps"] += n_sub
        stats["min_r"] = min(stats["min_r"], rmin)
        stats["max_radius"] = max(stats["max_radius"], max_radius(x))
        for k in ["zone1", "zone2", "zone3", "zone4"]:
            stats[k] += cc[k]
        fill_outputs()
        if stats["max_radius"] > 1e4:
            break

    while si < len(sample_times):
        pos[si] = x; vel[si] = v; si += 1
    return pos, vel, stats


def predict_residual(model: EncounterResidualMLP, x: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    xb = torch.tensor(x.reshape(1, 18), dtype=torch.float32, device=device)
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
    x_out = x_corr - (com_x_corr - com_x_ref)
    v_out = v_corr - (com_v_corr - com_v_ref)
    return x_out, v_out


def simulate_gated_surrogate_rollout(x0: np.ndarray, v0: np.ndarray, m: np.ndarray,
                                      model: EncounterResidualMLP,
                                      dt: float, T: float, sample_times: np.ndarray,
                                      window_years: float, vr_thresh: float,
                                      energy_gate: float, max_radius_gate: float,
                                      device: str, com_project: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], List[Dict[str, object]]]:
    x = x0.copy(); v = v0.copy(); t = 0.0
    pos = np.zeros((len(sample_times), 3, 3), dtype=np.float64)
    vel = np.zeros_like(pos)
    si = 0
    events: List[Dict[str, object]] = []
    stats = {"steps": 0, "substeps": 0, "min_r": min_pair_distance(x), "max_radius": max_radius(x),
             "zone1": 0, "zone2": 0, "zone3": 0, "zone4": 0,
             "surrogate_attempts": 0, "surrogate_used": 0, "surrogate_fallback": 0,
             "gate_candidates": 0}

    def fill_outputs():
        nonlocal si
        while si < len(sample_times) and t >= sample_times[si] - 1e-12:
            pos[si] = x; vel[si] = v; si += 1

    fill_outputs()
    a, _, c0 = revised_no_nn_acc(x, m)
    for k in ["zone1", "zone2", "zone3", "zone4"]:
        stats[k] += c0[k]

    while t < float(T) - 1e-14:
        X, vr, vt, r_pair = make_X_rel18(x, v, m, dt)
        in_zone3 = (ADAPT_THRESH <= r_pair < NN_THRESH)
        gate_ok = bool(in_zone3 and (vr < float(vr_thresh)) and (t + float(window_years) <= float(T) + 1e-12))
        if in_zone3:
            stats["gate_candidates"] += 1

        if gate_ok:
            stats["surrogate_attempts"] += 1
            t_start = t
            x_start = x.copy(); v_start = v.copy()
            E_start = total_energy(x_start, v_start, m, softened_pe=False)
            # Advance the noNN backbone across the exact encounter window while storing noNN intermediates.
            remaining = float(window_years)
            min_r_window = min_pair_distance(x)
            zone_window = {"zone1": 0, "zone2": 0, "zone3": 0, "zone4": 0}
            a, _, _ = revised_no_nn_acc(x, m)
            while remaining > 1e-14:
                step_dt = min(float(dt), remaining)
                x, v, a, rmin, cc, n_sub = no_nn_step(x, v, m, step_dt, a_in=a)
                t += step_dt
                remaining -= step_dt
                stats["steps"] += 1
                stats["substeps"] += n_sub
                min_r_window = min(min_r_window, rmin)
                stats["min_r"] = min(stats["min_r"], rmin)
                stats["max_radius"] = max(stats["max_radius"], max_radius(x))
                for k in ["zone1", "zone2", "zone3", "zone4"]:
                    stats[k] += cc[k]
                    zone_window[k] += cc[k]
                # Store intermediate noNN states inside the window.
                while si < len(sample_times) and sample_times[si] < t - 1e-12:
                    pos[si] = x; vel[si] = v; si += 1

            x_no_exit = x.copy(); v_no_exit = v.copy()
            pred_rx, pred_rv = predict_residual(model, X, device=device)
            x_corr = x_no_exit + pred_rx
            v_corr = v_no_exit + pred_rv
            if com_project:
                x_corr, v_corr = project_com_to_reference(x_corr, v_corr, x_no_exit, v_no_exit, m)
            E_corr = total_energy(x_corr, v_corr, m, softened_pe=False)
            relE_corr = abs((E_corr - E_start) / (abs(E_start) + 1e-30))
            pred_pos_norm = float(np.sqrt(np.mean(np.sum(pred_rx * pred_rx, axis=1))))
            pred_vel_norm = float(np.sqrt(np.mean(np.sum(pred_rv * pred_rv, axis=1))))
            reason = "used"
            use = True
            if not all_finite(x_corr, v_corr):
                use = False; reason = "nonfinite"
            elif max_radius(x_corr) > float(max_radius_gate):
                use = False; reason = "max_radius_gate"
            elif relE_corr > float(energy_gate):
                use = False; reason = "energy_gate"
            if use:
                x = x_corr; v = v_corr
                a, _, _ = revised_no_nn_acc(x, m)
                stats["surrogate_used"] += 1
            else:
                x = x_no_exit; v = v_no_exit
                a, _, _ = revised_no_nn_acc(x, m)
                stats["surrogate_fallback"] += 1
            stats["max_radius"] = max(stats["max_radius"], max_radius(x))
            events.append(dict(
                event=len(events) + 1, t_start=t_start, t_end=t, r_pair=r_pair,
                v_rad_norm=vr, v_tan_norm=vt, min_r_window=min_r_window,
                pred_pos_norm=pred_pos_norm, pred_vel_norm=pred_vel_norm,
                relE_corr=relE_corr, used=int(use), reason=reason,
                z1=zone_window["zone1"], z2=zone_window["zone2"],
                z3=zone_window["zone3"], z4=zone_window["zone4"],
            ))
            fill_outputs()
            continue

        # Normal noNN macro step.
        step_dt = min(float(dt), float(T) - t)
        x, v, a, rmin, cc, n_sub = no_nn_step(x, v, m, step_dt, a_in=a)
        t += step_dt
        stats["steps"] += 1
        stats["substeps"] += n_sub
        stats["min_r"] = min(stats["min_r"], rmin)
        stats["max_radius"] = max(stats["max_radius"], max_radius(x))
        for k in ["zone1", "zone2", "zone3", "zone4"]:
            stats[k] += cc[k]
        fill_outputs()
        if stats["max_radius"] > 1e4:
            break

    while si < len(sample_times):
        pos[si] = x; vel[si] = v; si += 1
    return pos, vel, stats, events


# =============================================================================
# Reporting
# =============================================================================
def summarize_errors(times: np.ndarray, pos_ref: np.ndarray, vel_ref: np.ndarray,
                     pos_method: np.ndarray, vel_method: np.ndarray) -> Dict[str, float]:
    pos_rms = rms_sep(pos_method, pos_ref)
    vel_rms = rms_sep(vel_method, vel_ref)
    return {
        "pos_final": float(pos_rms[-1]),
        "pos_timeavg": float(np.sqrt(np.mean(pos_rms * pos_rms))),
        "pos_median": float(np.median(pos_rms)),
        "pos_p95": float(np.percentile(pos_rms, 95)),
        "vel_final": float(vel_rms[-1]),
        "vel_timeavg": float(np.sqrt(np.mean(vel_rms * vel_rms))),
        "vel_median": float(np.median(vel_rms)),
        "vel_p95": float(np.percentile(vel_rms, 95)),
    }


def write_events_csv(path: str, events: List[Dict[str, object]]) -> None:
    fields = ["event", "t_start", "t_end", "r_pair", "v_rad_norm", "v_tan_norm",
              "min_r_window", "pred_pos_norm", "pred_vel_norm", "relE_corr",
              "used", "reason", "z1", "z2", "z3", "z4"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in events:
            w.writerow(row)


def maybe_make_plots(out_dir: str, times: np.ndarray, pos_ref: np.ndarray, vel_ref: np.ndarray,
                     pos_no: np.ndarray, vel_no: np.ndarray,
                     pos_surr: np.ndarray, vel_surr: np.ndarray) -> None:
    if not HAS_MPL:
        return
    pos_no_rms = rms_sep(pos_no, pos_ref)
    pos_s_rms = rms_sep(pos_surr, pos_ref)
    plt.figure(figsize=(9, 5))
    plt.plot(times, pos_no_rms, label="revised noNN")
    plt.plot(times, pos_s_rms, label="gated surrogate")
    plt.xlabel("time (yr)")
    plt.ylabel("RMS position error vs IAS15 (AU)")
    plt.title("Short rollout prototype: position RMS vs IAS15")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rms_vs_time.png"), dpi=200)
    plt.close()

    rad_ref = np.max(np.linalg.norm(pos_ref, axis=2), axis=1)
    rad_no = np.max(np.linalg.norm(pos_no, axis=2), axis=1)
    rad_s = np.max(np.linalg.norm(pos_surr, axis=2), axis=1)
    plt.figure(figsize=(9, 5))
    plt.plot(times, rad_ref, label="IAS15")
    plt.plot(times, rad_no, label="revised noNN")
    plt.plot(times, rad_s, label="gated surrogate")
    plt.xlabel("time (yr)")
    plt.ylabel("max radius from origin (AU)")
    plt.title("Short rollout prototype: boundedness diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "radius_vs_time.png"), dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Prototype rollout evaluator for gated encounter-level surrogate.")
    ap.add_argument("--model", default="encounter_surrogate_v2_velocitysafe_large1.pt",
                    help="Trained encounter surrogate .pt file.")
    ap.add_argument("--ic", default="IC1", choices=["IC1"], help="Initial condition; prototype supports IC1 only.")
    ap.add_argument("--dt", type=float, default=0.08, help="Macro timestep; prototype model was trained for dt=0.08.")
    ap.add_argument("--T", type=float, default=90.0, help="Rollout horizon in years. Use 90 to include the IC1 dt=0.08 event near 82-83 yr.")
    ap.add_argument("--n-samples", type=int, default=1200, help="Number of output samples for comparison.")
    ap.add_argument("--window-years", type=float, default=0.5, help="Surrogate encounter window length.")
    ap.add_argument("--vr-thresh", type=float, default=-0.40, help="Gate: use surrogate iff v_rad_norm < threshold.")
    ap.add_argument("--energy-gate", type=float, default=0.20, help="Fallback if corrected state has relative energy error above this vs window entry.")
    ap.add_argument("--max-radius-gate", type=float, default=100.0, help="Fallback if corrected max radius exceeds this AU.")
    ap.add_argument("--out-dir", default="rollout_surrogate_v1_T90", help="Output directory.")
    ap.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    ap.add_argument("--no-com-project", action="store_true", help="Disable COM projection after applying surrogate residual.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    model = load_surrogate_model(args.model, device=device)
    x0, v0, m = get_ic(args.ic)

    print("=" * 90)
    print("GATED ENCOUNTER-SURROGATE ROLLOUT PROTOTYPE")
    print(f"  model        : {args.model}")
    print(f"  ic           : {args.ic}")
    print(f"  dt/T         : {args.dt} yr / {args.T} yr")
    print(f"  window       : {args.window_years} yr")
    print(f"  gate         : active pair 1-2 in Zone 3 and v_rad_norm < {args.vr_thresh}")
    print(f"  device       : {device}")
    print(f"  out_dir      : {args.out_dir}")
    print("=" * 90)

    print("[1/3] IAS15 reference...")
    times, pos_ref, vel_ref, t_ias = simulate_ias15(x0, v0, m, args.T, args.n_samples)
    print(f"      IAS15 time: {t_ias:.3f}s")

    print("[2/3] revised noNN baseline...")
    t0 = time.perf_counter()
    pos_no, vel_no, stats_no = simulate_revised_nonn_rollout(x0, v0, m, args.dt, args.T, times)
    t_no = time.perf_counter() - t0
    print(f"      noNN time: {t_no:.3f}s")

    print("[3/3] gated encounter surrogate...")
    t0 = time.perf_counter()
    pos_s, vel_s, stats_s, events = simulate_gated_surrogate_rollout(
        x0, v0, m, model, args.dt, args.T, times,
        window_years=args.window_years,
        vr_thresh=args.vr_thresh,
        energy_gate=args.energy_gate,
        max_radius_gate=args.max_radius_gate,
        device=device,
        com_project=(not args.no_com_project),
    )
    t_s = time.perf_counter() - t0
    print(f"      surrogate time: {t_s:.3f}s | attempts={stats_s['surrogate_attempts']} used={stats_s['surrogate_used']} fallback={stats_s['surrogate_fallback']}")

    err_no = summarize_errors(times, pos_ref, vel_ref, pos_no, vel_no)
    err_s = summarize_errors(times, pos_ref, vel_ref, pos_s, vel_s)
    speed_no = t_ias / max(t_no, 1e-12)
    speed_s = t_ias / max(t_s, 1e-12)

    # Save arrays and event log.
    np.savez_compressed(
        os.path.join(args.out_dir, "rollout_arrays.npz"),
        times=times,
        pos_ref=pos_ref, vel_ref=vel_ref,
        pos_nonn=pos_no, vel_nonn=vel_no,
        pos_surrogate=pos_s, vel_surrogate=vel_s,
    )
    write_events_csv(os.path.join(args.out_dir, "surrogate_events.csv"), events)
    maybe_make_plots(args.out_dir, times, pos_ref, vel_ref, pos_no, vel_no, pos_s, vel_s)

    summary_path = os.path.join(args.out_dir, "rollout_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        def w(line: str = ""):
            print(line)
            f.write(line + "\n")

        w("Gated encounter-surrogate rollout prototype summary")
        w("=" * 88)
        w(f"model              : {args.model}")
        w(f"ic                 : {args.ic}")
        w(f"dt                 : {args.dt:.6f} yr")
        w(f"T                  : {args.T:.6f} yr")
        w(f"n_samples          : {args.n_samples}")
        w(f"window_years       : {args.window_years:.6f}")
        w(f"gate               : active pair 1-2 in Zone 3 and v_rad_norm < {args.vr_thresh:.4f}")
        w(f"energy_gate        : {args.energy_gate:.6f}")
        w(f"com_project        : {not args.no_com_project}")
        w("")
        w("Runtime")
        w("-" * 88)
        w(f"IAS15 time          : {t_ias:.6f} s")
        w(f"noNN time           : {t_no:.6f} s   speedup={speed_no:.3f}x")
        w(f"gated surrogate time: {t_s:.6f} s   speedup={speed_s:.3f}x")
        w("")
        w("Surrogate event stats")
        w("-" * 88)
        for k in ["gate_candidates", "surrogate_attempts", "surrogate_used", "surrogate_fallback"]:
            w(f"{k:22s}: {int(stats_s[k])}")
        if events:
            used = [e for e in events if int(e["used"]) == 1]
            if used:
                w(f"first_used_t        : {min(float(e['t_start']) for e in used):.6f} yr")
                w(f"last_used_t         : {max(float(e['t_start']) for e in used):.6f} yr")
                w(f"median_event_vr     : {np.median([float(e['v_rad_norm']) for e in used]):+.6f}")
                w(f"median_pred_pos_norm: {np.median([float(e['pred_pos_norm']) for e in used]):.6e}")
                w(f"median_pred_vel_norm: {np.median([float(e['pred_vel_norm']) for e in used]):.6e}")
        w("")
        w("Comparison against IAS15")
        w("-" * 88)
        w(f"{'method':24s} {'pos_final':>12s} {'pos_timeavg':>12s} {'pos_med':>12s} {'pos_p95':>12s} {'vel_final':>12s} {'vel_timeavg':>12s}")
        w("  " + "-" * 108)
        w(f"{'revised noNN':24s} {err_no['pos_final']:12.4e} {err_no['pos_timeavg']:12.4e} {err_no['pos_median']:12.4e} {err_no['pos_p95']:12.4e} {err_no['vel_final']:12.4e} {err_no['vel_timeavg']:12.4e}")
        w(f"{'gated surrogate':24s} {err_s['pos_final']:12.4e} {err_s['pos_timeavg']:12.4e} {err_s['pos_median']:12.4e} {err_s['pos_p95']:12.4e} {err_s['vel_final']:12.4e} {err_s['vel_timeavg']:12.4e}")
        w("")
        pos_gain = 100.0 * (err_no["pos_timeavg"] - err_s["pos_timeavg"]) / (abs(err_no["pos_timeavg"]) + 1e-30)
        vel_gain = 100.0 * (err_no["vel_timeavg"] - err_s["vel_timeavg"]) / (abs(err_no["vel_timeavg"]) + 1e-30)
        w("Decision diagnostics")
        w("-" * 88)
        w(f"position timeavg gain vs noNN : {pos_gain:+.2f}%")
        w(f"velocity timeavg gain vs noNN : {vel_gain:+.2f}%")
        w(f"surrogate changed rollout?     : {int(stats_s['surrogate_used']) > 0}")
        if int(stats_s["surrogate_used"]) == 0:
            verdict = "NO EVENT: gate did not fire; extend T or inspect active pair."
        elif pos_gain > 0.0 and vel_gain > -5.0:
            verdict = "PASS/CONTINUE: gated surrogate did not break short rollout and improves position time-average."
        elif pos_gain > 0.0:
            verdict = "MIXED: position improves but velocity worsens; inspect event log before full rollout."
        else:
            verdict = "STOP/REASSESS: gated surrogate worsens position rollout versus noNN."
        w(f"VERDICT: {verdict}")

    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
