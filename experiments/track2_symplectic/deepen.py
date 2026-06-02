"""
Track 2 deepening:
 (1) synthetic Sun-Earth-Moon tight-pair (Earth-Moon ~0.00257 AU) -- the regime
     that forces continuous fine resolution; mode='newton' (physical point masses).
 (2) speed-accuracy frontier plot from track2_results.json.
 (3) energy(t) boundedness plot for IC1 (tsalf vs leapfrog) -- the symplectic story.
"""
import sys, os, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT + "/src")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import simon_core as sc
import integrators_symplectic as si

OUT = ROOT + "/experiments/track2_symplectic"
L = []; log = lambda s: (print(s), L.append(s))

# ---------- (1) synthetic Sun-Earth-Moon ----------
def sem_synthetic():
    G = sc.G_REAL
    mS, mE, mM = 1.0, 3.003e-6, 3.694e-8
    aE, aM = 1.0, 0.00257
    vE = np.sqrt(G*mS/aE)
    vM = np.sqrt(G*(mE+mM)/aM)
    m = np.array([mS, mE, mM])
    x0 = np.array([[0,0,0],[aE,0,0],[aE+aM,0,0]], float)
    v0 = np.array([[0,0,0],[0,vE,0],[0,vE+vM,0]], float)
    M = m.sum(); x0 -= (m[:,None]*x0).sum(0)/M; v0 -= (m[:,None]*v0).sum(0)/M
    return m, x0, v0, G

log("="*92); log("SYNTHETIC SUN-EARTH-MOON (Earth-Moon=0.00257 AU, mode=newton, G=4pi^2)"); log("="*92)
m, x0, v0, G = sem_synthetic()
T_SEM, NS = 30.0, 4000          # ~400 lunar orbits
EPS_N = 1e-9                    # effectively-Newtonian energy for point masses
_, pref, vref = sc.ias15_reference(m, x0, v0, T_SEM, NS, G)
def em_range(pos):
    d = np.linalg.norm(pos[:,1,:]-pos[:,2,:], axis=1); return float(d.min()), float(d.max())
log(f"IAS15 ref Earth-Moon range: {em_range(pref)} AU")
hdr = f"{'method':<24}{'param':>9}{'|dE/E0|max':>13}{'EM_min':>10}{'EM_max':>10}{'bnd':>6}{'force_evals':>12}"
log(hdr); log("-"*len(hdr))
sem_rows = []
def run_sem(name, param, t, p, vv, fe):
    dE = sc.max_dE(p, vv, m, G, eps=EPS_N); emn, emx = em_range(p); bnd = sc.bounded(p, m)
    dEs = "inf" if not np.isfinite(dE) else f"{dE:.4f}%"
    log(f"{name:<24}{param:>9}{dEs:>13}{emn:>10.5f}{emx:>10.5f}{str(bnd):>6}{fe:>12d}")
    sem_rows.append(dict(method=name,param=param,max_dE_pct=dE,em_min=emn,em_max=emx,bounded=bnd,force_evals=fe))
for dt in [0.04, 0.01, 0.002]:
    t,p,vv,info = sc.simulate_leapfrog(m,x0,v0,dt,T_SEM,NS,G,mode='newton',adaptive=False)
    run_sem("leapfrog_fixed", f"dt={dt}", t,p,vv,info['force_evals'])
t,p,vv,info = sc.simulate_leapfrog(m,x0,v0,0.04,T_SEM,NS,G,mode='newton',adaptive=True)
run_sem("leapfrog_adaptive", "dt=0.04", t,p,vv,info['force_evals'])
for eta in [0.1, 0.05]:
    t,p,vv,info = si.tsalf_simulate(m,x0,v0,eta,T_SEM,NS,G,mode='newton')
    run_sem("tsalf", f"eta={eta}", t,p,vv,info['force_evals'])
    sem_rows[-1]['max_dE_pct'] = info['max_dE_steps']; sem_rows[-1]['max_dE_steps']=info['max_dE_steps']
log("")
# verdict for SEM
ts_sem = [r for r in sem_rows if r['method']=='tsalf' and r['bounded']]
lf_sem = [r for r in sem_rows if r['method']=='leapfrog_fixed']
log("SEM verdict:")
for r in lf_sem:
    log(f"  leapfrog {r['param']}: bnd={r['bounded']} dE~{r['max_dE_pct']:.3g}% fe={r['force_evals']} EM=[{r['em_min']:.5f},{r['em_max']:.5f}]")
for r in ts_sem:
    log(f"  tsalf {r['param']}: bnd={r['bounded']} dE={r['max_dE_pct']:.4g}% fe={r['force_evals']} EM=[{r['em_min']:.5f},{r['em_max']:.5f}]")

# ---------- (2) frontier plot from json ----------
rows = json.load(open(OUT+"/track2_results.json"))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, ic in zip(axes, ["IC1","IC4","IC6"]):
    rr = [r for r in rows if r['ic']==ic]
    for r in rr:
        dE = r['max_dE_pct']
        if not np.isfinite(dE) or not r['bounded']:
            ax.scatter(r['force_evals'], max(dE if np.isfinite(dE) else 1e4, 1e-4), marker='x', s=70, color='red')
            ax.annotate(r['method'].split('_')[0]+("!" if not r['bounded'] else ""), (r['force_evals'], max(dE if np.isfinite(dE) else 1e4,1e-4)), fontsize=6, color='red')
        else:
            c = '#16A34A' if r['method']=='tsalf' else ('#9333EA' if r['method']=='yoshida4' else '#2563A6')
            ax.scatter(r['force_evals'], max(dE,1e-5), s=60, color=c)
            ax.annotate(r['method'].split('_')[0]+":"+r['param'].split('=')[-1], (r['force_evals'], max(dE,1e-5)), fontsize=6)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(ic)
    ax.set_xlabel('force evals (cost)'); ax.set_ylabel('|dE/E0|max %'); ax.grid(True, which='both', alpha=0.3)
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
fig.suptitle("Track 2 speed-accuracy frontier (green=tsalf, purple=yoshida4, blue=leapfrog; red x=ejected/unbounded)")
fig.tight_layout(); fig.savefig(OUT+"/frontier.png", dpi=170); plt.close()
log("wrote frontier.png")

# ---------- (3) energy(t) boundedness for IC1 ----------
m1,x1,v1 = sc.get_ic("IC1")
fig, ax = plt.subplots(figsize=(8,4.8))
# leapfrog dt=0.04 and dt=0.005 (grid energy)
for dt,col in [(0.04,'#2563A6'),(0.005,'#DC2626')]:
    t,p,vv,info = sc.simulate_leapfrog(m1,x1,v1,dt,100.0,5000,sc.G_TOY,mode='soft',adaptive=False)
    E = sc.energy_series(p,vv,m1,sc.G_TOY); dEt = np.abs((E-E[0])/abs(E[0]))*100
    ax.semilogy(t, np.maximum(dEt,1e-6), color=col, lw=1.3, label=f"leapfrog dt={dt}")
# tsalf eta=0.05 (step states)
ts,xs,vs = si.tsalf_simulate(m1,x1,v1,0.05,100.0,5000,sc.G_TOY,mode='soft')[3]['step_states']
E = sc.energy_series(xs,vs,m1,sc.G_TOY); dEt = np.abs((E-E[0])/abs(E[0]))*100
ax.semilogy(ts, np.maximum(dEt,1e-6), color='#16A34A', lw=1.5, label="tsalf eta=0.05")
ax.set_xlabel("time (yr)"); ax.set_ylabel("|dE/E0|(t)  %"); ax.set_title("IC1 energy error vs time (symplectic story)")
ax.grid(True, which='both', alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT+"/energy_vs_time_IC1.png", dpi=170); plt.close()
log("wrote energy_vs_time_IC1.png")

open(OUT+"/sem_synthetic.txt","w").write("\n".join(L)+"\n")
json.dump(sem_rows, open(OUT+"/sem_synthetic.json","w"), indent=2, default=str)
print("DEEPEN_DONE")
