"""
weekend_phaseC: binary-single operating-envelope sweep.
6 ICs parameterised by binary separation r_bin (0.01..0.40 AU).
Masses [1.0,0.5,0.1] Msun, G=4pi^2. Binary = bodies 0,1 (circular, sep r_bin).
Perturber = body 2 at 3.0 AU circular around binary COM. Coplanar prograde.

This pass: diagnostic (binary period vs dt) + variant (a) SIMON heuristic no-NN
+ (c) ias15. Metrics: gate-firing %, time_avg_RMS(a-c), max|dE| (softened PE),
bounded, and the binary pair-range metric (catches binary unbinding).
"""
import sys, os, json, time, math
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT + "/src")
import numpy as np
import simon_core as sc
import integrators_symplectic as si

OUT = ROOT + "/experiments/weekend_phaseC"; os.makedirs(OUT, exist_ok=True)
G = sc.G_REAL; T, dt, NS = 100.0, 0.04, 5000
R_BINS = [0.01, 0.03, 0.05, 0.10, 0.20, 0.40]
m = np.array([1.0, 0.5, 0.1]); mb = 1.5
L=[]; log=lambda s:(print(s,flush=True),L.append(s))

def binary_single_ic(r_bin):
    x = np.array([[-r_bin*0.5/mb, 0, 0],   # body0 (primary)
                  [ r_bin*1.0/mb, 0, 0],   # body1 (secondary): separation = r_bin
                  [ 3.0,          0, 0]], float)
    vrel = math.sqrt(G*mb/r_bin)           # circular binary
    v2   = math.sqrt(G*mb/3.0)             # circular perturber around binary
    v = np.array([[0, -vrel*0.5/mb, 0],
                  [0,  vrel*1.0/mb, 0],
                  [0,  v2,          0]], float)
    M = m.sum(); x -= (m[:,None]*x).sum(0)/M; v -= (m[:,None]*v).sum(0)/M
    return x, v

def binary_range(pos):
    d = np.linalg.norm(pos[:,0,:]-pos[:,1,:], axis=1); return float(d.min()), float(d.max())

log("="*100)
log("BINARY-SINGLE OPERATING ENVELOPE — diagnostic + (a) heuristic no-NN + (c) ias15")
log(f"masses={m.tolist()} G=4pi^2  T={T} dt={dt}  binary=bodies(0,1) perturber=body2@3AU circular")
log("="*100)
log(f"{'r_bin':>7}{'t_orb_bin':>11}{'orbits/macrostep':>17}{'heur n_sub':>11}{'sub/orbit':>11}")
for rb in R_BINS:
    t_orb = 2*math.pi*math.sqrt(rb**3/(G*mb))
    nsub = min(16, max(2, math.ceil(0.05/rb))) if rb < 0.05 else 1
    sub_dt = dt/nsub
    log(f"{rb:>7}{t_orb:>11.2e}{dt/t_orb:>17.1f}{nsub:>11}{t_orb/sub_dt:>11.3f}")
log("(need ~30 sub-steps/orbit for leapfrog; <1 means catastrophically under-resolved)")
log("")

rows=[]
log(f"{'r_bin':>7}{'gate%':>7}{'RMS(a-c)':>11}{'|dE|max_a%':>12}{'bnd_a':>7}{'bin_range_a(AU)':>22}{'bin_range_ias15':>22}{'t_a':>7}{'t_c':>8}")
for rb in R_BINS:
    x0, v0 = binary_single_ic(rb)
    # (c) ias15
    t0=time.time(); _, pc, vc = sc.ias15_reference(m, x0, v0, T, NS, G); tc=time.time()-t0
    # (a) SIMON heuristic, no NN  (mode='soft' = softened force, c=1; adaptive sub-stepping)
    t0=time.time(); _, pa, va, ia = sc.simulate_leapfrog(m, x0, v0, dt, T, NS, G, mode='soft', adaptive=True); ta=time.time()-t0
    nsh = np.asarray(ia['n_sub_history']); gate = 100.0*np.mean(nsh>1)
    rms_ac = float(np.sqrt(np.mean(sc.rms_sep(pa, pc)**2)))
    dEa = sc.max_dE(pa, va, m, G, eps=sc.EPS)
    bnd = sc.bounded(pa, m)
    bmn,bmx = binary_range(pa); imn,imx = binary_range(pc)
    log(f"{rb:>7}{gate:>6.0f}%{rms_ac:>11.4f}{dEa:>11.3f}%{str(bnd):>7}"
        f"{('['+f'{bmn:.4f},{bmx:.4f}'+']'):>22}{('['+f'{imn:.4f},{imx:.4f}'+']'):>22}{ta:>7.1f}{tc:>8.1f}")
    rows.append(dict(r_bin=rb, gate_pct=gate, rms_ac=rms_ac, maxdE_a=dEa, bounded_a=bnd,
                     bin_range_a=[bmn,bmx], bin_range_ias15=[imn,imx], t_a=ta, t_c=tc))

json.dump(dict(rows=rows), open(OUT+"/phaseC_ac.json","w"), indent=2, default=str)
open(OUT+"/phaseC_ac.txt","w").write("\n".join(L)+"\n")
print("\nAC_DONE")
