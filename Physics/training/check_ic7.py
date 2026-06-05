# check_ic7.py
#
# Verifies whether IC7 (high-eccentricity redesign) is physically unstable
# by running ias15 alone and checking whether it also ejects a body.
#
# IC7 coordinates (identical to multi_ic_eval_v3.py):
#   m  = [1.0, 0.01, 0.005]
#   x0 = [[0,0,0], [0.6,0,0], [0,4.0,0]]
#   v0 = [[0,0,0], [0,1.6,0], [-0.15,0,0]]
#   Design: Body 1 at 0.6 AU, v=1.6 (1.24x v_circ, e~0.52), Body 2 at 4 AU
#
# QUESTION: does ias15 also eject a body from IC7?
#   YES -> IC7 is physically unstable. Accept 4/7 bounded, move on.
#   NO  -> leapfrog at dt=0.04 is the cause. Try smaller dt or redesign.
#
# Run: python check_ic7.py

import time
import numpy as np

try:
    import rebound
except ImportError:
    print("[ERROR] rebound not found.")
    raise

G                  = 1.0
T                  = 100.0
N_SAMPLES          = 5000
EJECTION_THRESHOLD = 10.0   # AU

# ── IC7 coordinates (CoM-centered, identical to multi_ic_eval_v3.py) ─────────
m  = np.array([1.0, 0.01, 0.005])
x0 = np.array([[0,0,0],[0.6,0,0],[0,4.0,0]], dtype=np.float64)
v0 = np.array([[0,0,0],[0,1.6,0],[-0.15,0,0]], dtype=np.float64)

M  = m.sum()
x0 = x0 - (m[:, None] * x0).sum(0) / M
v0 = v0 - (m[:, None] * v0).sum(0) / M

# Energy + orbital properties
KE = 0.5 * np.sum(m[:, None] * v0**2)
PE = sum(-G*m[i]*m[j]/np.linalg.norm(x0[i]-x0[j])
         for i in range(3) for j in range(i+1,3))
E  = KE + PE

r01    = np.linalg.norm(x0[1]-x0[0])
v01    = np.linalg.norm(v0[1]-v0[0])
v_circ = np.sqrt(G*m[0]/r01)
v_esc  = np.sqrt(2*G*m[0]/r01)
mu     = G*(m[0]+m[1])
a_orb  = 1.0/(2.0/r01 - v01**2/mu)
L      = r01*v01
ecc    = np.sqrt(max(0.0, 1.0 - L**2/(mu*a_orb)))

print("IC7 properties:")
print(f"  E_total  = {E:.5f}  (BOUND)")
print(f"  r01      = {r01:.3f} AU  (Body 1 start)")
print(f"  v_circ   = {v_circ:.3f}  v_esc = {v_esc:.3f}")
print(f"  Body 1 speed: {v01:.3f} ({v01/v_circ:.2f}x v_circ, {v01/v_esc:.2f}x v_esc)")
print(f"  Estimated eccentricity e ~ {ecc:.3f}  (semi-major axis a ~ {a_orb:.3f} AU)")
print(f"  Body 2 distance from Body 1: {np.linalg.norm(x0[2]-x0[1]):.2f} AU")

# ── Run ias15 ────────────────────────────────────────────────────────────────
print(f"\n[check_ic7] Running ias15 alone on IC7  (T={T}yr, n_samples={N_SAMPLES}) ...")
sim = rebound.Simulation()
sim.integrator = "ias15"
sim.G = G
for i in range(len(m)):
    sim.add(m=float(m[i]),
            x=float(x0[i,0]), y=float(x0[i,1]), z=float(x0[i,2]),
            vx=float(v0[i,0]), vy=float(v0[i,1]), vz=float(v0[i,2]))
sim.move_to_com()

times = np.linspace(0.0, T, N_SAMPLES)
pos   = np.zeros((N_SAMPLES, 3, 3))

t_start = time.perf_counter()
for k, t in enumerate(times):
    sim.integrate(t)
    for i, p in enumerate(sim.particles):
        pos[k, i] = [p.x, p.y, p.z]
elapsed = time.perf_counter() - t_start
print(f"  ias15 completed in {elapsed:.3f}s")

# ── Analyse ───────────────────────────────────────────────────────────────────
separations   = np.linalg.norm(pos[-1], axis=1)
max_sep_final = separations.max()
ejected_body  = int(np.argmax(separations))

max_sep_vs_time = np.max(np.linalg.norm(pos, axis=2), axis=1)
ejection_times  = times[max_sep_vs_time > EJECTION_THRESHOLD]
t_ejection      = ejection_times[0] if len(ejection_times) > 0 else None

print(f"\n[check_ic7] RESULTS -- ias15 on IC7:")
print(f"  Final body separations from origin at T=100yr:")
for i in range(3):
    print(f"    Body {i} (m={m[i]:.3f}): {separations[i]:.4f} AU")
print(f"  Max separation: {max_sep_final:.4f} AU  (threshold = {EJECTION_THRESHOLD} AU)")
if t_ejection is not None:
    print(f"  First ejection (>{EJECTION_THRESHOLD} AU) at: t = {t_ejection:.1f} yr")
    print(f"  Body furthest: Body {ejected_body} at {separations[ejected_body]:.2f} AU")

print(f"\n{'='*60}")
if max_sep_final > EJECTION_THRESHOLD:
    ratio = 11.44 / max_sep_final
    print("VERDICT: ias15 ALSO EJECTS a body from IC7.")
    print(f"  ias15 final max separation: {max_sep_final:.2f} AU")
    print(f"  SIMON final max separation: ~11.44 AU (from multi_ic_eval_v3.py)")
    print(f"  Ratio SIMON/ias15: {ratio:.2f}x")
    print()
    print("CONCLUSION: IC7 is physically unstable.")
    print("  Both integrators predict ejection. Not a SIMON failure.")
    print()
    print("IMPLICATION: High-eccentricity configurations with these mass")
    print("  ratios are intrinsically prone to instability via three-body")
    print("  energy exchange. The eccentric inner body transfers energy to")
    print("  the outer body over repeated close passes.")
    print()
    print("ACTION: Accept 4/7 bounded. Stop redesigning high-ecc ICs.")
    print("  Bounded ICs cover the full stable spectrum:")
    print("    IC6: near-circular  (lambda=-0.019, RMS=0.008 AU)")
    print("    IC4: hierarchical   (lambda=0.099,  RMS=1.60 AU)")
    print("    IC1: moderate       (lambda=0.166,  RMS=1.54 AU)")
    print("    IC3: tight binary   (lambda=0.121,  RMS=3.45 AU)")
    print("  Unstable: IC2, IC5, IC7 -- all physically confirmed.")
else:
    print("VERDICT: ias15 STAYS BOUNDED on IC7.")
    print(f"  ias15 final max separation: {max_sep_final:.4f} AU  (< {EJECTION_THRESHOLD} AU)")
    print(f"  SIMON final max separation: ~11.44 AU (ejection)")
    print()
    print("CONCLUSION: IC7 ejection is a SIMON / leapfrog issue, not physical.")
    print("  ias15 stays bounded but SIMON's leapfrog at dt=0.04 accumulates")
    print("  sufficient phase error to eventually transfer orbital energy.")
    print()
    print("NEXT STEP: Run SIMON on IC7 at smaller dt (e.g. dt=0.01) to check")
    print("  whether the ejection disappears. If yes, IC7 is bounded at finer")
    print("  dt and can be included with a note on timestep sensitivity.")
print(f"{'='*60}")
