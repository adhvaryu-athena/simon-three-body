"""
Track 5: population sweep mapping tsalf's operating envelope vs baselines.
N random bound 3-body configs (Sun + 2 small bodies, varied radii/eccentricity).
Per config, per method (leapfrog dt=0.04, leapfrog dt=0.005, heuristic adaptive,
tsalf eta=0.05): panel (|dE/E0|max, bounded, force_evals). Classify regime by the
minimum pairwise separation over the run. Incremental JSON checkpoint every 25.
"""
import sys, os, json, time, math
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT + "/src")
import numpy as np
import simon_core as sc
import integrators_symplectic as si

OUT = ROOT + "/experiments/track5_population"; os.makedirs(OUT, exist_ok=True)
G = sc.G_TOY; T, NS = 20.0, 1000
N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SEED = 12345

def gen_config(rng):
    mS = 1.0
    m1 = 10 ** rng.uniform(-3.0, -1.3); m2 = 10 ** rng.uniform(-3.3, -1.3)
    r1 = rng.uniform(0.5, 3.0); r2 = rng.uniform(1.2, 6.0)
    th1 = rng.uniform(0, 2*np.pi); th2 = rng.uniform(0, 2*np.pi)
    f1 = rng.uniform(0.65, 1.15); f2 = rng.uniform(0.65, 1.15)   # ecc via speed factor
    vc1 = math.sqrt(G*mS/r1); vc2 = math.sqrt(G*mS/r2)
    m = np.array([mS, m1, m2])
    x = np.array([[0,0,0],
                  [r1*np.cos(th1), r1*np.sin(th1), 0],
                  [r2*np.cos(th2), r2*np.sin(th2), 0]], float)
    v = np.array([[0,0,0],
                  [-vc1*f1*np.sin(th1), vc1*f1*np.cos(th1), 0],
                  [-vc2*f2*np.sin(th2), vc2*f2*np.cos(th2), 0]], float)
    M = m.sum(); x -= (m[:,None]*x).sum(0)/M; v -= (m[:,None]*v).sum(0)/M
    # bound check
    KE = 0.5*np.sum(m[:,None]*v**2); PE = 0.0
    for i in range(3):
        for j in range(i+1,3):
            PE -= G*m[i]*m[j]/np.linalg.norm(x[i]-x[j])
    if KE+PE >= 0: return None
    return m, x, v

def panel(method, m, x0, v0):
    t0 = time.time()
    if method == "lf04":
        t,p,vv,info = sc.simulate_leapfrog(m,x0,v0,0.04,T,NS,G,mode='newton',adaptive=False)
        dE = sc.max_dE(p,vv,m,G,eps=1e-9); fe=info['force_evals']
    elif method == "lf005":
        t,p,vv,info = sc.simulate_leapfrog(m,x0,v0,0.005,T,NS,G,mode='newton',adaptive=False)
        dE = sc.max_dE(p,vv,m,G,eps=1e-9); fe=info['force_evals']
    elif method == "adapt":
        t,p,vv,info = sc.simulate_leapfrog(m,x0,v0,0.04,T,NS,G,mode='newton',adaptive=True)
        dE = sc.max_dE(p,vv,m,G,eps=1e-9); fe=info['force_evals']
    elif method == "tsalf":
        t,p,vv,info = si.tsalf_simulate(m,x0,v0,0.05,T,NS,G,mode='newton')
        dE = info['max_dE_steps']; fe=info['force_evals']
    bnd = sc.bounded(p,m)
    # regime: min pairwise sep over run
    minsep = 1e9
    for k in range(0, NS, 5):
        d = p[k];
        for i in range(3):
            for j in range(i+1,3):
                minsep = min(minsep, float(np.linalg.norm(d[i]-d[j])))
    return dict(method=method, max_dE_pct=float(dE) if np.isfinite(dE) else None,
                bounded=bool(bnd), force_evals=int(fe), minsep=minsep, wall=time.time()-t0)

rng = np.random.RandomState(SEED)
results = []
ckpt = OUT + "/popsweep.json"
t_all = time.time(); ngen = 0; nkept = 0
while nkept < N:
    cfg = gen_config(rng); ngen += 1
    if cfg is None: continue
    m, x0, v0 = cfg
    rec = dict(cfg_id=nkept, m=m.tolist())
    ok = True
    for meth in ["lf04","lf005","adapt","tsalf"]:
        try:
            rec[meth] = panel(meth, m, x0, v0)
        except Exception as e:
            rec[meth] = dict(method=meth, error=str(e)); ok=False
    # min separation regime from the fine reference (lf005)
    rec['minsep'] = rec.get('lf005',{}).get('minsep', None)
    results.append(rec); nkept += 1
    if nkept % 25 == 0:
        json.dump(dict(N=nkept, n_generated=ngen, T=T, results=results),
                  open(ckpt, "w"), default=str)
        print(f"[{time.time()-t_all:.0f}s] kept {nkept}/{N} (gen {ngen})", flush=True)
json.dump(dict(N=nkept, n_generated=ngen, T=T, results=results), open(ckpt,"w"), default=str)
print(f"POPSWEEP_DONE kept={nkept} gen={ngen} wall={time.time()-t_all:.0f}s")
