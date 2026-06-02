"""
integrators_symplectic.py — Track 2 candidates, each verified before use.

(A) tsalf  : time-symmetric adaptive leapfrog (Hut-Makino-McMillan 1995 style).
             Stepsize h solved implicitly as h = 0.5*(tau(start)+tau(end)) by a
             few fixed-point iterations -> time-reversible -> energy stays bounded
             even with adaptive steps. tau = eta * (min pairwise dynamical time).
             Target: IC1-style close-encounter chaos.
(B) yoshida4: 4th-order symplectic (Yoshida triple-jump of velocity-Verlet),
             fixed step -> strictly symplectic, error ~ds^4. 3 force-evals/step.
             Target: IC4-style smooth under-resolution at low cost.

Verification (run this file directly): Kepler e=0.9 energy boundedness for both,
plus a reversibility round-trip for tsalf.
"""
import math
import numpy as np
import simon_core as sc

# Yoshida 4th-order triple-jump coefficients
W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
W0 = 1.0 - 2.0 * W1

def _acc_fn(m, G, mode, w):
    P = sc._prep(m, G); ii, jj = P['ii'], P['jj']; eps2 = sc.EPS * sc.EPS
    cnt = [0]
    def acc(x):
        cnt[0] += 1
        return sc.compute_acc(x, ii, jj, P['Gmimj'], P['inv_mi'], P['inv_mj'],
                              P['log_mi'], P['log_mj'], w, eps2, mode)
    return acc, cnt, P

def _verlet(x, v, a, h, acc):
    """One velocity-Verlet (KDK) step; 1 force eval. Returns x1,v1,a1."""
    vh = v + 0.5 * h * a
    x1 = x + h * vh
    a1 = acc(x1)
    v1 = vh + 0.5 * h * a1
    return x1, v1, a1

def t_dyn_min(x, v, m, ii, jj, G):
    """Min over pairs of min(orbital_time, flyby_time) — the tightest timescale."""
    rij = x[jj] - x[ii]; vij = v[jj] - v[ii]
    r2 = np.einsum('ij,ij->i', rij, rij); r = np.sqrt(r2 + 1e-30)
    vmag = np.sqrt(np.einsum('ij,ij->i', vij, vij) + 1e-30)
    GM = G * (m[ii] + m[jj])
    t_orb = 2.0 * np.pi * np.sqrt(r2 * r / (GM + 1e-30))
    t_fly = r / vmag
    return float(np.min(np.minimum(t_orb, t_fly)))

# ---------------------------------------------------------------------------
def yoshida4_simulate(m, x0, v0, ds, T, n_samples, G, mode='soft', w=None):
    acc, cnt, P = _acc_fn(m, G, mode, w)
    x = x0.astype(np.float64).copy(); v = v0.astype(np.float64).copy()
    N = len(m); times = np.linspace(0.0, T, n_samples)
    n_steps = int(round(T / ds))
    pos = np.full((n_samples, N, 3), np.nan); vel = np.full((n_samples, N, 3), np.nan)
    a = acc(x); pos[0] = x; vel[0] = v; si = 0
    for step in range(n_steps):
        for c in (W1, W0, W1):
            x, v, a = _verlet(x, v, a, c * ds, acc)
        t_cur = (step + 1) * ds
        while si < n_samples - 1 and times[si + 1] <= t_cur + 1e-9:
            si += 1; pos[si] = x; vel[si] = v
    while si < n_samples - 1:
        si += 1; pos[si] = x; vel[si] = v
    assert not np.isnan(pos).any()
    assert not np.all(pos == 0.0, axis=2).all(axis=1).any()
    return times, pos, vel, dict(force_evals=cnt[0], n_steps=n_steps, scheme='yoshida4', ds=ds)

# ---------------------------------------------------------------------------
def tsalf_simulate(m, x0, v0, eta, T, n_samples, G, mode='soft', w=None,
                   n_iter=2, max_steps=3_000_000):
    acc, cnt, P = _acc_fn(m, G, mode, w); ii, jj = P['ii'], P['jj']
    x = x0.astype(np.float64).copy(); v = v0.astype(np.float64).copy(); a = acc(x)
    ts = [0.0]; xs = [x.copy()]; vs = [v.copy()]
    t = 0.0; nstep = 0
    while t < T and nstep < max_steps:
        tau0 = eta * t_dyn_min(x, v, m, ii, jj, G)
        h = tau0
        for _ in range(n_iter):                      # implicit symmetric stepsize
            x1, v1, a1 = _verlet(x, v, a, h, acc)
            tau1 = eta * t_dyn_min(x1, v1, m, ii, jj, G)
            h = 0.5 * (tau0 + tau1)
        if t + h > T:
            h = T - t                                # land exactly on T (last step only)
        x, v, a = _verlet(x, v, a, h, acc)
        t += h; nstep += 1
        ts.append(t); xs.append(x.copy()); vs.append(v.copy())
    ts = np.array(ts); xs = np.array(xs); vs = np.array(vs)
    # energy on the ACTUAL step states (no interpolation smoothing)
    maxdE = sc.max_dE(xs, vs, m, G)
    # interpolate to the fixed sample grid for RMS-vs-IAS15
    times = np.linspace(0.0, T, n_samples)
    pos = np.empty((n_samples, len(m), 3)); vel = np.empty((n_samples, len(m), 3))
    for i in range(len(m)):
        for d in range(3):
            pos[:, i, d] = np.interp(times, ts, xs[:, i, d])
            vel[:, i, d] = np.interp(times, ts, vs[:, i, d])
    return times, pos, vel, dict(force_evals=cnt[0], n_steps=nstep, scheme='tsalf',
                                 eta=eta, max_dE_steps=maxdE, completed=(t >= T - 1e-9),
                                 step_states=(ts, xs, vs))

# ---------------------------------------------------------------------------
# Verification: Kepler e=0.9, and tsalf reversibility round-trip.
# ---------------------------------------------------------------------------
def _kepler_ic(a=1.0, e=0.9, m0=1.0, m1=1e-3, G=1.0):
    M = m0 + m1
    rp = a * (1 - e); vp = math.sqrt(G * M * (1 + e) / rp)
    x = np.array([[0.0, 0, 0], [rp, 0, 0]]); v = np.array([[0.0, 0, 0], [0, vp, 0]])
    m = np.array([m0, m1])
    Mt = m.sum(); x -= (m[:, None] * x).sum(0) / Mt; v -= (m[:, None] * v).sum(0) / Mt
    P_orb = 2 * math.pi * math.sqrt(a**3 / (G * M))
    return m, x, v, P_orb

if __name__ == "__main__":
    G = 1.0
    m, x0, v0, Porb = _kepler_ic(a=1.0, e=0.9, G=G)
    T = 20 * Porb; NS = 4000
    print(f"=== KEPLER VERIFICATION (e=0.9, {20} orbits, Porb={Porb:.3f}) ===")
    for ds in [0.05, 0.02]:
        _, p, vv, info = yoshida4_simulate(m, x0, v0, ds, T, NS, G, mode='newton')
        print(f"  yoshida4 ds={ds:<5}: |dE/E0|max={sc.max_dE(p,vv,m,G):.3e}%  fe={info['force_evals']:7d}  bounded={sc.bounded(p,m)}")
    for eta in [0.05, 0.02]:
        _, p, vv, info = tsalf_simulate(m, x0, v0, eta, T, NS, G, mode='newton')
        print(f"  tsalf    eta={eta:<5}: |dE/E0|max={info['max_dE_steps']:.3e}%  fe={info['force_evals']:7d}  steps={info['n_steps']}  bounded={sc.bounded(p,m)}")
    # reversibility: forward T then backward T (negate v), compare to start
    print("=== TSALF REVERSIBILITY (forward then time-reversed) ===")
    ts, xs, vs = tsalf_simulate(m, x0, v0, 0.05, 5*Porb, 2000, G, mode='newton')[3]['step_states']
    xf, vf = xs[-1].copy(), vs[-1].copy()
    _, _, _, infob = tsalf_simulate(m, xf, -vf, 0.05, 5*Porb, 2000, G, mode='newton')
    tb, xb, vb = infob['step_states']
    err = np.linalg.norm(xb[-1] - x0)
    print(f"  round-trip position error ||x_back - x_0|| = {err:.3e}  (small => reversible)")
    print("KEPLER_SELFTEST_DONE")
