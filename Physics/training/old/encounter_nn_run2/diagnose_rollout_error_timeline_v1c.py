#!/usr/bin/env python3
"""
diagnose_rollout_error_timeline_v1.py

Diagnostic for the encounter-surrogate rollout.

It compares revised noNN vs gated encounter-surrogate error over time after the
first corrected event, and optionally sweeps alpha values such as 0.25, 0.5,
0.75, 1.0.

Typical run:
python -B diagnose_rollout_error_timeline_v1.py ^
  --model encounter_surrogate_v3_rollout_local_velocitysafe.pt ^
  --T 100 --dt 0.08 --window-years 0.5 ^
  --vr-thresh -0.40 ^
  --alphas 0.25,0.5,0.75,1.0 ^
  --out-dir rollout_error_timeline_v1
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
import rebound


@dataclass
class Config:
    G: float = 1.0
    eps: float = 3e-4
    zone1_r_gate: float = 4e-4
    adapt_thresh: float = 0.05
    nn_thresh: float = 0.15
    max_substeps: int = 16
    energy_gate: float = 0.20


def ic1_com_centered() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = np.array([1.0, 0.01, 0.005], dtype=np.float64)
    x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.2, 0.0]], dtype=np.float64)
    v0 = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.9, 0.0, 0.0]], dtype=np.float64)
    M = float(np.sum(m))
    x0 = x0 - np.sum(m[:, None] * x0, axis=0) / M
    v0 = v0 - np.sum(m[:, None] * v0, axis=0) / M
    return m, x0, v0


class ResidualMLP(nn.Module):
    """Residual surrogate reconstructed directly from checkpoint Linear layers.

    The velocity-safe checkpoints used in this project are not always exactly
    18 -> 128 -> 128 -> 128 -> 18. The current checkpoint is:

        net.0.weight  : (128, 18)
        net.3.weight  : (128, 128)
        net.6.weight  : (128, 128)
        net.9.weight  : (64, 128)
        net.11.weight : (18, 64)

    So this diagnostic reconstructs the Linear stack from the checkpoint
    instead of assuming a fixed Sequential index for the output layer.
    """
    def __init__(self, layer_shapes: List[Tuple[int, int]], out_dim: int = 18):
        super().__init__()
        if not layer_shapes:
            raise ValueError("No Linear layer shapes were supplied")
        self.in_dim = int(layer_shapes[0][1])
        self.out_dim = int(out_dim)
        self.linears = nn.ModuleList([
            nn.Linear(int(in_features), int(out_features))
            for out_features, in_features in layer_shapes
        ])
        self.register_buffer("input_mean", torch.zeros(self.in_dim))
        self.register_buffer("input_std", torch.ones(self.in_dim))
        self.register_buffer("target_mean", torch.zeros(self.out_dim))
        self.register_buffer("target_std", torch.ones(self.out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.input_mean) / (self.input_std + 1e-8)
        for layer in self.linears[:-1]:
            z = torch.nn.functional.silu(layer(z))
        y = self.linears[-1](z)
        return y * (self.target_std + 1e-8) + self.target_mean


def _state_dict(obj):
    if isinstance(obj, dict):
        for k in ["model_state_dict", "state_dict", "model"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _copy_buffer(model, sd, dst, names):
    buf = getattr(model, dst)
    for name in names:
        if name in sd and tuple(sd[name].shape) == tuple(buf.shape):
            with torch.no_grad():
                buf.copy_(sd[name].detach().float())
            return name
    return None


def _find_buffer_key(sd, names, length=None):
    for name in names:
        if name in sd:
            v = sd[name]
            if length is None or (hasattr(v, "numel") and int(v.numel()) == int(length)):
                return name
    return None


def _net_linear_weight_keys(sd: Dict[str, torch.Tensor]) -> List[str]:
    """Return checkpoint Linear weight keys in forward order.

    Handles keys like net.0.weight, net.3.weight, net.11.weight. If the
    checkpoint uses a prefixed naming convention, this still keeps only the
    numeric order under '.net.'.
    """
    pairs = []
    for k, v in sd.items():
        if not (k.endswith(".weight") and getattr(v, "ndim", None) == 2):
            continue
        parts = k.split(".")
        # Prefer standard net.<integer>.weight keys.
        idx = None
        for a, b in zip(parts[:-1], parts[1:]):
            if a == "net" and b.isdigit():
                idx = int(b)
                break
        if idx is not None:
            pairs.append((idx, k))
    if not pairs:
        raise ValueError("Could not find checkpoint layers named like net.<idx>.weight")
    return [k for _, k in sorted(pairs)]


def _bias_key_for_weight(weight_key: str) -> str:
    return weight_key[:-len(".weight")] + ".bias" if weight_key.endswith(".weight") else weight_key + ".bias"


def _copy_param(dst_param: torch.nn.Parameter, src_tensor: torch.Tensor):
    with torch.no_grad():
        dst_param.copy_(src_tensor.detach().float())


def load_model(path: str, device: str):
    raw = torch.load(path, map_location="cpu")
    sd = _state_dict(raw)
    if not isinstance(sd, dict):
        raise ValueError("Could not read a PyTorch state_dict from the model file")

    weight_keys = _net_linear_weight_keys(sd)
    layer_shapes = [tuple(sd[k].shape) for k in weight_keys]
    in_dim = int(layer_shapes[0][1])
    out_dim = int(layer_shapes[-1][0])

    # Residual target must be 18 = 9 position residuals + 9 velocity residuals.
    target_key_for_dim = _find_buffer_key(sd, ["target_mean", "y_mean", "Y_mean", "output_mean"], None)
    target_dim = int(sd[target_key_for_dim].numel()) if target_key_for_dim is not None else out_dim
    if out_dim != 18 or target_dim != 18:
        shapes = ", ".join(f"{k}:{tuple(sd[k].shape)}" for k in weight_keys)
        raise ValueError(
            f"Expected final residual layer and target dimension to be 18, got out_dim={out_dim}, "
            f"target_dim={target_dim}. Linear stack: {shapes}"
        )

    model = ResidualMLP(layer_shapes=layer_shapes, out_dim=out_dim)

    # Manually copy every Linear layer, preserving the checkpoint architecture.
    for layer, wkey in zip(model.linears, weight_keys):
        _copy_param(layer.weight, sd[wkey])
        bkey = _bias_key_for_weight(wkey)
        if bkey in sd and tuple(sd[bkey].shape) == tuple(layer.bias.shape):
            _copy_param(layer.bias, sd[bkey])
        else:
            raise ValueError(f"Missing or incompatible bias for {wkey}; expected {bkey}")

    info = {
        "in_dim": str(in_dim),
        "hidden_stack": " -> ".join(str(s[0]) for s in layer_shapes[:-1]),
        "out_dim": str(out_dim),
        "linear_keys": ",".join(weight_keys),
        "input_mean": _copy_buffer(model, sd, "input_mean", ["input_mean", "x_mean", "X_mean", "feature_mean"]),
        "input_std": _copy_buffer(model, sd, "input_std", ["input_std", "x_std", "X_std", "feature_std"]),
        "target_mean": _copy_buffer(model, sd, "target_mean", ["target_mean", "y_mean", "Y_mean", "output_mean"]),
        "target_std": _copy_buffer(model, sd, "target_std", ["target_std", "y_std", "Y_std", "output_std"]),
    }

    # Final safety check: the model must emit exactly 18 residual values.
    with torch.no_grad():
        probe = torch.zeros((1, in_dim), dtype=torch.float32)
        y_probe = model(probe).reshape(-1)
    if int(y_probe.numel()) != 18:
        raise RuntimeError(f"Model loader error: model emits {int(y_probe.numel())} values, expected 18")

    model.to(device).eval()
    return model, info


def project_com(x, v, m):
    M = float(np.sum(m))
    return x - np.sum(m[:, None] * x, axis=0) / M, v - np.sum(m[:, None] * v, axis=0) / M


def compute_energy(x, v, m, cfg):
    ke = 0.5 * float(np.sum(m[:, None] * v * v))
    pe = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            pe -= cfg.G * float(m[i] * m[j]) / (float(np.linalg.norm(x[j] - x[i])) + 1e-30)
    return ke + pe


def pair_geom(x, v, m, cfg, i=1, j=2):
    rij = x[j] - x[i]
    vij = v[j] - v[i]
    r = float(np.linalg.norm(rij) + 1e-30)
    rhat = rij / r
    vr = float(np.dot(vij, rhat))
    vt = float(np.linalg.norm(vij - vr * rhat))
    vscale = math.sqrt(cfg.G * float(m[i] + m[j]) / (r + 1e-30))
    return r, vr / (vscale + 1e-30), vt / (vscale + 1e-30)


def min_pair_distance(x):
    return min(float(np.linalg.norm(x[j] - x[i])) for i in range(3) for j in range(i + 1, 3))


def accelerations_nonn(x, m, cfg):
    acc = np.zeros_like(x, dtype=np.float64)
    for i in range(3):
        for j in range(i + 1, 3):
            rij = x[j] - x[i]
            r2 = float(np.dot(rij, rij))
            r = math.sqrt(r2 + 1e-30)
            if cfg.adapt_thresh <= r < cfg.nn_thresh:
                fsc = cfg.G * m[i] * m[j] / ((r2 + cfg.eps * cfg.eps) ** 1.5 + 1e-30)
            else:
                fsc = cfg.G * m[i] * m[j] / (r2 * r + 1e-30)
            fvec = fsc * rij
            acc[i] += fvec / (m[i] + 1e-30)
            acc[j] -= fvec / (m[j] + 1e-30)
    return acc


def leapfrog_step_nonn(x, v, m, cfg, dt):
    rmin = min_pair_distance(x)
    nsub = 1
    if rmin < cfg.adapt_thresh:
        nsub = min(cfg.max_substeps, max(2, int(math.ceil(cfg.adapt_thresh / max(rmin, 1e-30)))))
    h = float(dt) / nsub
    for _ in range(nsub):
        a0 = accelerations_nonn(x, m, cfg)
        vh = v + 0.5 * h * a0
        x = x + h * vh
        a1 = accelerations_nonn(x, m, cfg)
        v = vh + 0.5 * h * a1
    return x, v


def simulate_ias15(x0, v0, m, cfg, T, n_samples):
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.G = cfg.G
    for i in range(3):
        sim.add(m=float(m[i]), x=float(x0[i,0]), y=float(x0[i,1]), z=float(x0[i,2]),
                vx=float(v0[i,0]), vy=float(v0[i,1]), vz=float(v0[i,2]))
    sim.move_to_com()
    times = np.linspace(0.0, T, n_samples)
    pos = np.zeros((n_samples, 3, 3), dtype=np.float64)
    vel = np.zeros((n_samples, 3, 3), dtype=np.float64)
    t0 = time.perf_counter()
    for k, t in enumerate(times):
        sim.integrate(float(t))
        for i, p in enumerate(sim.particles):
            pos[k, i] = [p.x, p.y, p.z]
            vel[k, i] = [p.vx, p.vy, p.vz]
    return times, pos, vel, time.perf_counter() - t0


def rms_error(a, b):
    d = a - b
    per_body = np.sqrt(np.sum(d * d, axis=-1))
    return np.sqrt(np.mean(per_body * per_body, axis=1))


def build_xrel18(x, v, m, dt, cfg):
    # active pair is 1-2; third body is 0
    i, j, k = 1, 2, 0
    pair_M = m[i] + m[j]
    r_pair = x[j] - x[i]
    v_pair = v[j] - v[i]
    x_com_pair = (m[i] * x[i] + m[j] * x[j]) / (pair_M + 1e-30)
    v_com_pair = (m[i] * v[i] + m[j] * v[j]) / (pair_M + 1e-30)
    r_third = x[k] - x_com_pair
    v_third = v[k] - v_com_pair
    _, vrn, vtn = pair_geom(x, v, m, cfg, i, j)
    return np.concatenate([
        np.log(m + 1e-30), r_pair, v_pair, r_third, v_third,
        np.array([math.log(float(dt) + 1e-30), vrn, vtn], dtype=np.float64)
    ]).astype(np.float32)


def predict_residual(model, x, v, m, dt, cfg, device):
    feat = build_xrel18(x, v, m, dt, cfg)
    with torch.no_grad():
        y = model(torch.tensor(feat[None, :], dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
    if y.size != 18:
        raise RuntimeError(
            f"Residual model returned {y.size} values, expected 18. "
            "The output layer was probably loaded incorrectly. Use the fixed v1b script."
        )
    dx = y[:9].reshape(3, 3).astype(np.float64)
    dv = y[9:18].reshape(3, 3).astype(np.float64)
    pred_pos_norm = float(np.sqrt(np.mean(np.sum(dx * dx, axis=1))))
    pred_vel_norm = float(np.sqrt(np.mean(np.sum(dv * dv, axis=1))))
    return dx, dv, pred_pos_norm, pred_vel_norm


def simulate_nonn_global(x0, v0, m, cfg, dt, T, n_samples):
    times = np.linspace(0.0, T, n_samples)
    pos = np.zeros((n_samples, 3, 3), dtype=np.float64)
    vel = np.zeros((n_samples, 3, 3), dtype=np.float64)
    x, v = x0.copy(), v0.copy()
    t = 0.0
    si = 0
    while si < n_samples and times[si] <= t + 1e-12:
        pos[si], vel[si] = x, v
        si += 1
    t0 = time.perf_counter()
    for _ in range(int(math.ceil(T / dt))):
        h = min(float(dt), T - t)
        if h <= 1e-12:
            break
        x, v = leapfrog_step_nonn(x, v, m, cfg, h)
        t += h
        while si < n_samples and times[si] <= t + 1e-12:
            pos[si], vel[si] = x, v
            si += 1
    while si < n_samples:
        pos[si], vel[si] = x, v
        si += 1
    return times, pos, vel, time.perf_counter() - t0


def run_nonn_window(x, v, m, cfg, dt, window_years):
    t = 0.0
    for _ in range(int(math.ceil(window_years / dt))):
        h = min(float(dt), window_years - t)
        if h <= 1e-12:
            break
        x, v = leapfrog_step_nonn(x, v, m, cfg, h)
        t += h
    return x, v


def simulate_surrogate_global(x0, v0, m, cfg, model, device, dt, T, n_samples, window_years, vr_thresh, alpha, max_events=1):
    times = np.linspace(0.0, T, n_samples)
    pos = np.zeros((n_samples, 3, 3), dtype=np.float64)
    vel = np.zeros((n_samples, 3, 3), dtype=np.float64)
    x, v = x0.copy(), v0.copy()
    t = 0.0
    si = 0
    events = []
    while si < n_samples and times[si] <= t + 1e-12:
        pos[si], vel[si] = x, v
        si += 1
    t0 = time.perf_counter()
    guard = int(math.ceil(T / min(dt, window_years))) + 10000
    for _ in range(guard):
        if t >= T - 1e-12:
            break
        r, vrn, vtn = pair_geom(x, v, m, cfg, 1, 2)
        gate = (cfg.adapt_thresh <= r < cfg.nn_thresh) and (vrn < vr_thresh) and (len(events) < max_events)
        if gate:
            event_t = t
            dx, dv, pred_pos_norm, pred_vel_norm = predict_residual(model, x, v, m, dt, cfg, device)
            E_before = compute_energy(x, v, m, cfg)
            exit_t = min(T, t + window_years)
            x_exit, v_exit = run_nonn_window(x.copy(), v.copy(), m, cfg, dt, exit_t - t)
            x_corr = x_exit + float(alpha) * dx
            v_corr = v_exit + float(alpha) * dv
            x_corr, v_corr = project_com(x_corr, v_corr, m)
            E_after = compute_energy(x_corr, v_corr, m, cfg)
            relE = abs((E_after - E_before) / (abs(E_before) + 1e-30))
            accepted = np.isfinite(x_corr).all() and np.isfinite(v_corr).all() and relE <= cfg.energy_gate
            # Fill skipped samples up to exit with noNN exit state (not used for final decision; analysis starts at exit)
            while si < n_samples and times[si] <= exit_t + 1e-12:
                pos[si], vel[si] = x_exit, v_exit
                si += 1
            if accepted:
                x, v = x_corr, v_corr
                # overwrite exact exit sample if close enough
                idx = int(np.argmin(np.abs(times - exit_t)))
                if abs(times[idx] - exit_t) <= max(dt, times[1]-times[0]):
                    pos[idx], vel[idx] = x, v
                status = "used"
            else:
                x, v = x_exit, v_exit
                status = "fallback_energy"
            t = exit_t
            events.append({
                "alpha": float(alpha), "event_t": float(event_t), "window_exit_t": float(exit_t),
                "r_pair": float(r), "v_rad_norm": float(vrn), "v_tan_norm": float(vtn),
                "pred_pos_norm": pred_pos_norm, "pred_vel_norm": pred_vel_norm,
                "relE_corr": float(relE), "accepted": int(accepted), "status": status,
            })
            continue
        h = min(float(dt), T - t)
        if h <= 1e-12:
            break
        x, v = leapfrog_step_nonn(x, v, m, cfg, h)
        t += h
        while si < n_samples and times[si] <= t + 1e-12:
            pos[si], vel[si] = x, v
            si += 1
    while si < n_samples:
        pos[si], vel[si] = x, v
        si += 1
    return times, pos, vel, time.perf_counter() - t0, events


def intervals(times, mask):
    out = []
    s = None
    for i, val in enumerate(mask):
        if val and s is None:
            s = i
        elif (not val) and s is not None:
            out.append((float(times[s]), float(times[i-1]), i-s))
            s = None
    if s is not None:
        out.append((float(times[s]), float(times[-1]), len(times)-s))
    return out


def analyse(times, err_no, err_s, event_exit_t, tol):
    mask = times >= event_exit_t - 1e-12
    t = times[mask]
    diff = err_s[mask] - err_no[mask]
    worse = diff > tol
    first_worse_t = float(t[np.argmax(worse)]) if np.any(worse) else float('nan')
    imax = int(np.argmax(diff)) if len(diff) else 0
    max_excess = float(diff[imax]) if len(diff) else float('nan')
    max_excess_t = float(t[imax]) if len(diff) else float('nan')
    ivals = intervals(t, worse)
    if ivals:
        containing = [iv for iv in ivals if iv[0] <= max_excess_t <= iv[1]]
        iv = containing[0] if containing else max(ivals, key=lambda z: z[2])
        istart, iend, icount = iv
    else:
        istart, iend, icount = float('nan'), float('nan'), 0
    frac = float(np.mean(worse)) if len(worse) else float('nan')
    tail = t >= (event_exit_t + 0.5 * (times[-1] - event_exit_t))
    tail_frac = float(np.mean(worse[tail])) if np.any(tail) else float('nan')
    if not np.any(worse):
        pattern = "never worse after correction exit"
    elif frac > 0.5 or tail_frac > 0.5:
        pattern = "persistent branch divergence / phase shift"
    elif icount <= max(3, int(0.10 * len(t))):
        pattern = "short transient excess"
    else:
        pattern = "mixed/intermittent phase-shift behaviour"
    return dict(first_worse_t=first_worse_t, max_excess=max_excess, max_excess_t=max_excess_t,
                max_interval_start=istart, max_interval_end=iend, max_interval_samples=icount,
                frac_worse=frac, tail_frac_worse=tail_frac, pattern=pattern)


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def parse_alphas(s):
    return [float(x.strip()) for x in str(s).split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--T', type=float, default=100.0)
    ap.add_argument('--dt', type=float, default=0.08)
    ap.add_argument('--window-years', type=float, default=0.5)
    ap.add_argument('--n-samples', type=int, default=1200)
    ap.add_argument('--vr-thresh', type=float, default=-0.40)
    ap.add_argument('--alphas', type=parse_alphas, default=[1.0])
    ap.add_argument('--out-dir', default='rollout_error_timeline_v1')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--energy-gate', type=float, default=0.20)
    ap.add_argument('--max-events', type=int, default=1)
    ap.add_argument('--worse-tolerance', type=float, default=1e-9)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = Config(energy_gate=args.energy_gate)
    m, x0, v0 = ic1_com_centered()

    print('='*88)
    print('ROLLOUT ERROR TIMELINE DIAGNOSTIC v1')
    print(f'model   : {args.model}')
    print(f'dt/T    : {args.dt:.6f} / {args.T:.3f}')
    print(f'alphas  : {args.alphas}')
    print(f'out_dir : {args.out_dir}')
    print('='*88)

    model, info = load_model(args.model, args.device)
    print('[1/3] IAS15 reference...')
    times, pos_ias, vel_ias, t_ias = simulate_ias15(x0, v0, m, cfg, args.T, args.n_samples)
    print(f'      IAS15 time={t_ias:.3f}s')
    print('[2/3] revised noNN baseline...')
    _, pos_no, vel_no, t_no = simulate_nonn_global(x0, v0, m, cfg, args.dt, args.T, args.n_samples)
    print(f'      noNN time={t_no:.3f}s')
    err_no_pos = rms_error(pos_no, pos_ias)
    err_no_vel = rms_error(vel_no, vel_ias)

    alpha_rows, timeline_rows, event_rows = [], [], []
    best_alpha, best_score = None, float('inf')
    print('[3/3] alpha sweep...')
    for alpha in args.alphas:
        _, pos_s, vel_s, t_s, events = simulate_surrogate_global(
            x0, v0, m, cfg, model, args.device, args.dt, args.T, args.n_samples,
            args.window_years, args.vr_thresh, alpha, max_events=args.max_events)
        err_s_pos = rms_error(pos_s, pos_ias)
        err_s_vel = rms_error(vel_s, vel_ias)
        if events:
            event_t = events[0]['event_t']; exit_t = events[0]['window_exit_t']; used = events[0]['accepted']
        else:
            event_t = float('nan'); exit_t = 0.0; used = 0
        pa = analyse(times, err_no_pos, err_s_pos, exit_t, args.worse_tolerance)
        va = analyse(times, err_no_vel, err_s_vel, exit_t, args.worse_tolerance)
        pos_no_ta, pos_s_ta = float(np.mean(err_no_pos)), float(np.mean(err_s_pos))
        vel_no_ta, vel_s_ta = float(np.mean(err_no_vel)), float(np.mean(err_s_vel))
        pos_gain = 100.0 * (pos_no_ta - pos_s_ta) / (pos_no_ta + 1e-30)
        vel_gain = 100.0 * (vel_no_ta - vel_s_ta) / (vel_no_ta + 1e-30)
        row = {
            'alpha': alpha, 'event_used': used, 'event_t': event_t, 'event_exit_t': exit_t,
            'runtime_sec': t_s,
            'pos_final_noNN': float(err_no_pos[-1]), 'pos_final_surrogate': float(err_s_pos[-1]),
            'pos_timeavg_noNN': pos_no_ta, 'pos_timeavg_surrogate': pos_s_ta, 'pos_timeavg_gain_pct': pos_gain,
            'vel_final_noNN': float(err_no_vel[-1]), 'vel_final_surrogate': float(err_s_vel[-1]),
            'vel_timeavg_noNN': vel_no_ta, 'vel_timeavg_surrogate': vel_s_ta, 'vel_timeavg_gain_pct': vel_gain,
            'pos_first_worse_t': pa['first_worse_t'], 'pos_max_excess': pa['max_excess'],
            'pos_max_excess_t': pa['max_excess_t'], 'pos_max_excess_interval_start': pa['max_interval_start'],
            'pos_max_excess_interval_end': pa['max_interval_end'], 'pos_frac_post_window_worse': pa['frac_worse'],
            'pos_tail_frac_worse': pa['tail_frac_worse'], 'pos_pattern': pa['pattern'],
            'vel_first_worse_t': va['first_worse_t'], 'vel_max_excess': va['max_excess'],
            'vel_max_excess_t': va['max_excess_t'], 'vel_frac_post_window_worse': va['frac_worse'],
            'vel_tail_frac_worse': va['tail_frac_worse'], 'vel_pattern': va['pattern'],
        }
        alpha_rows.append(row)
        for ev in events:
            event_rows.append(ev)
        for k, t in enumerate(times):
            timeline_rows.append({
                'alpha': alpha, 'time': float(t),
                'pos_err_noNN': float(err_no_pos[k]), 'pos_err_surrogate': float(err_s_pos[k]),
                'pos_excess_surrogate_minus_noNN': float(err_s_pos[k] - err_no_pos[k]),
                'vel_err_noNN': float(err_no_vel[k]), 'vel_err_surrogate': float(err_s_vel[k]),
                'vel_excess_surrogate_minus_noNN': float(err_s_vel[k] - err_no_vel[k]),
                'after_event_exit': int(t >= exit_t - 1e-12),
            })
        score = (pos_s_ta - pos_no_ta) + 0.25 * (vel_s_ta - vel_no_ta)
        if score < best_score:
            best_score, best_alpha = score, alpha
        print(f'      alpha={alpha:.2f}: pos_timeavg {pos_no_ta:.6e}->{pos_s_ta:.6e} ({pos_gain:+.2f}%), '
              f'vel_timeavg {vel_no_ta:.6e}->{vel_s_ta:.6e} ({vel_gain:+.2f}%), pattern={pa["pattern"]}')

    alpha_csv = os.path.join(args.out_dir, 'alpha_sweep_summary.csv')
    timeline_csv = os.path.join(args.out_dir, 'rollout_error_timeline.csv')
    events_csv = os.path.join(args.out_dir, 'surrogate_events_timeline_diag.csv')
    write_csv(alpha_csv, alpha_rows)
    write_csv(timeline_csv, timeline_rows)
    write_csv(events_csv, event_rows)

    summary_path = os.path.join(args.out_dir, 'rollout_error_timeline_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('Rollout error timeline diagnostic v1\n')
        f.write('='*88 + '\n')
        f.write(f'model              : {args.model}\n')
        f.write(f'dt                 : {args.dt:.8f} yr\n')
        f.write(f'T                  : {args.T:.8f} yr\n')
        f.write(f'window_years       : {args.window_years:.8f}\n')
        f.write(f'n_samples          : {args.n_samples}\n')
        f.write(f'gate               : pair 1-2 in Zone 3 and v_rad_norm < {args.vr_thresh:.4f}\n')
        f.write(f'alphas             : {args.alphas}\n')
        f.write(f'energy_gate        : {args.energy_gate:.6f}\n')
        f.write(f'device             : {args.device}\n\n')
        f.write('Model load info\n' + '-'*88 + '\n')
        for k, v in info.items():
            f.write(f'{k:20s}: {v}\n')
        f.write('\nBaseline\n' + '-'*88 + '\n')
        f.write(f'IAS15 time         : {t_ias:.6f} s\n')
        f.write(f'noNN time          : {t_no:.6f} s\n')
        f.write(f'noNN pos final     : {err_no_pos[-1]:.8e}\n')
        f.write(f'noNN pos timeavg   : {np.mean(err_no_pos):.8e}\n')
        f.write(f'noNN vel final     : {err_no_vel[-1]:.8e}\n')
        f.write(f'noNN vel timeavg   : {np.mean(err_no_vel):.8e}\n')
        f.write('\nAlpha sweep\n' + '-'*88 + '\n')
        f.write('alpha  used event_t  exit_t   pos_final  pos_timeavg pos_gain  vel_final  vel_timeavg vel_gain  first_worse  pos_pattern\n')
        for r in alpha_rows:
            f.write(f"{r['alpha']:5.2f} {r['event_used']:5d} {r['event_t']:7.3f} {r['event_exit_t']:7.3f} "
                    f"{r['pos_final_surrogate']:.4e} {r['pos_timeavg_surrogate']:.4e} {r['pos_timeavg_gain_pct']:+7.2f}% "
                    f"{r['vel_final_surrogate']:.4e} {r['vel_timeavg_surrogate']:.4e} {r['vel_timeavg_gain_pct']:+7.2f}% "
                    f"{r['pos_first_worse_t']:10.3f}  {r['pos_pattern']}\n")
        f.write('\nDetailed position timeline interpretation\n' + '-'*88 + '\n')
        for r in alpha_rows:
            f.write(f"alpha={r['alpha']:.4f}\n")
            f.write(f"  first time surrogate position error becomes worse : {r['pos_first_worse_t']}\n")
            f.write(f"  max excess position error                         : {r['pos_max_excess']:.8e}\n")
            f.write(f"  max excess time                                   : {r['pos_max_excess_t']}\n")
            f.write(f"  max excess interval                               : {r['pos_max_excess_interval_start']} to {r['pos_max_excess_interval_end']}\n")
            f.write(f"  fraction worse after event exit                   : {r['pos_frac_post_window_worse']:.3f}\n")
            f.write(f"  tail fraction worse                               : {r['pos_tail_frac_worse']:.3f}\n")
            f.write(f"  pattern                                           : {r['pos_pattern']}\n")
        f.write('\nDecision hint\n' + '-'*88 + '\n')
        f.write(f'Best alpha by simple position-first score: {best_alpha}\n')
        f.write('Prefer the smallest alpha that keeps local/event gain while avoiding persistent post-event position-timeavg worsening.\n')
        f.write('\nFiles written\n' + '-'*88 + '\n')
        f.write(alpha_csv + '\n')
        f.write(timeline_csv + '\n')
        f.write(events_csv + '\n')

    print('='*88)
    print(f'[done] wrote {summary_path}')
    print(f'[done] wrote {alpha_csv}')
    print(f'[done] wrote {timeline_csv}')
    print(f'[done] wrote {events_csv}')
    print(f'[decision hint] best alpha by simple score: {best_alpha}')
    print('='*88)


if __name__ == '__main__':
    main()
