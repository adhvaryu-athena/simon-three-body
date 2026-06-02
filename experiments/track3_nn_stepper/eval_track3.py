"""
Track 3 eval (uses saved model): equal-ACCURACY comparison.
NN-tsalf targets uniform local-error (learned mu* on eta*t_dyn). Analytic-tsalf
targets uniform eta*t_dyn. Sweep analytic eta to build its accuracy-cost frontier;
place NN-tsalf on it. CLICK iff NN is materially BELOW the analytic frontier
(cheaper at equal energy accuracy) on held-out IC1/IC4/IC6.
"""
import sys, os, json, math
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT + "/src")
import numpy as np, torch
import simon_core as sc
import integrators_symplectic as si
OUT = ROOT + "/experiments/track3_nn_stepper"; G = sc.G_TOY; ETA0 = 0.1
L=[]; log=lambda s:(print(s,flush=True),L.append(s))

d = np.load(OUT+"/stepper_norm.npz"); MU, SD = d['mu'], d['sd']
sdt = torch.load(OUT+"/stepper_nn.pt", map_location="cpu")
W0,b0=sdt['0.weight'].numpy(),sdt['0.bias'].numpy()
W2,b2=sdt['2.weight'].numpy(),sdt['2.bias'].numpy()
W4,b4=sdt['4.weight'].numpy(),sdt['4.bias'].numpy()
silu=lambda z: z/(1+np.exp(-z))
def nn_predict(feat):
    h=(feat-MU)/SD
    h=silu(h@W0.T+b0); h=silu(h@W2.T+b2)
    return float((h@W4.T+b4)[0])           # fixed: shape-(1,) -> scalar

def acc_factory(m):
    P=sc._prep(m,G); ii,jj=P['ii'],P['jj']; eps2=sc.EPS**2
    return (lambda x: sc.compute_acc(x,ii,jj,P['Gmimj'],P['inv_mi'],P['inv_mj'],P['log_mi'],P['log_mj'],None,eps2,'newton')), ii, jj

def features(x,v,m,ii,jj):
    rij=x[jj]-x[ii]; vij=v[jj]-v[ii]
    r=np.sqrt(np.einsum('ij,ij->i',rij,rij)+1e-30); k=int(np.argmin(r)); i,j=ii[k],jj[k]
    rv=r[k]; vrel=math.sqrt(float(vij[k]@vij[k])+1e-30); eproxy=abs(float(rij[k]@vij[k])/(rv*vrel+1e-30))
    GM=G*(m[i]+m[j]); t_orb=2*np.pi*math.sqrt(rv**3/(GM+1e-30)); t_fly=rv/vrel
    kk=[b for b in range(len(m)) if b not in (i,j)][0]; a_int=GM/rv**2
    a_tid=G*m[kk]*(1.0/(np.linalg.norm(x[i]-x[kk])**2+1e-9)+1.0/(np.linalg.norm(x[j]-x[kk])**2+1e-9))
    return np.array([math.log(rv),math.log(vrel+1e-12),eproxy,math.log(GM+1e-30),
                     math.log(t_orb/(t_fly+1e-30)+1e-30),math.log(a_tid/(a_int+1e-30)+1e-30),
                     math.log(ETA0*min(t_orb,t_fly)+1e-30)],dtype=np.float32)

def run(m,x0,v0,T,NS,eta,use_nn):
    acc,ii,jj=acc_factory(m); x=x0.copy(); v=v0.copy(); a=acc(x)
    ts=[0.0]; xs=[x.copy()]; vs=[v.copy()]; t=0.0; ns=0
    while t<T and ns<3_000_000:
        td=si.t_dyn_min(x,v,m,ii,jj,G); h0=eta*td
        if use_nn: h0*=min(8.0,max(0.25,2.0**nn_predict(features(x,v,m,ii,jj))))
        vh=v+0.5*h0*a; x1=x+h0*vh; a1=acc(x1); v1=vh+0.5*h0*a1
        td1=si.t_dyn_min(x1,v1,m,ii,jj,G); h1=eta*td1
        if use_nn: h1*=min(8.0,max(0.25,2.0**nn_predict(features(x1,v1,m,ii,jj))))
        h=0.5*(h0+h1)
        if t+h>T: h=T-t
        vh=v+0.5*h*a; x=x+h*vh; a=acc(x); v=vh+0.5*h*a; t+=h; ns+=1
        ts.append(t); xs.append(x.copy()); vs.append(v.copy())
        if not np.all(np.isfinite(x)): break
    ts=np.array(ts); xs=np.array(xs); vs=np.array(vs)
    return sc.max_dE(xs,vs,m,G,eps=1e-9), ns, (t>=T-1e-6)

def fe_at_accuracy(front, dEn):
    """Force-evals the analytic frontier needs to reach accuracy dEn, by log-log
    interpolation on the Pareto lower-envelope (fe asc, dE desc). Equal-accuracy."""
    pts = sorted((fe, dE) for (eta, dE, fe, ok) in front if ok and np.isfinite(dE) and dE > 0)
    pareto = []; best = 1e18
    for fe, dE in pts:
        if dE < best - 1e-15: pareto.append((fe, dE)); best = dE
    if not pareto: return None
    if dEn >= pareto[0][1]: return pareto[0][0]      # NN no more accurate than coarsest analytic
    if dEn <= pareto[-1][1]: return pareto[-1][0]     # NN beyond finest analytic tested
    for k in range(len(pareto) - 1):
        fe0, d0 = pareto[k]; fe1, d1 = pareto[k+1]    # d0 > d1
        if d1 <= dEn <= d0:
            f = (math.log(dEn) - math.log(d0)) / (math.log(d1) - math.log(d0))
            return math.exp(math.log(fe0) + f * (math.log(fe1) - math.log(fe0)))
    return pareto[-1][0]

log("HELD-OUT EVAL — equal-accuracy frontier (analytic eta sweep) vs NN-tsalf")
rows=[]
for ic in ["IC1","IC4","IC6"]:
    m,x0,v0=sc.get_ic(ic)
    # analytic frontier
    front=[]
    for eta in [0.2,0.1,0.05,0.025,0.0125]:
        dE,fe,ok=run(m,x0,v0,100.0,5000,eta,use_nn=False); front.append((eta,dE,fe,ok))
    dEn,fen,okn=run(m,x0,v0,100.0,5000,ETA0,use_nn=True)
    # analytic fe needed to MATCH the NN's accuracy (interp on bounded points, dE vs fe)
    fe_eq=fe_at_accuracy(front, dEn)         # analytic fe at EQUAL accuracy (interp)
    log(f"\n{ic}: NN-tsalf dE={dEn:.4f}% fe={fen} bnd={okn}")
    log(f"   analytic frontier (eta,dE%,fe,bnd): "+", ".join(f"({e},{d:.3f},{f},{o})" for e,d,f,o in front))
    if fe_eq:
        log(f"   cheapest analytic reaching NN's accuracy: fe={fe_eq} -> NN/analytic cost = {fen/fe_eq:.2f}x")
    rows.append(dict(ic=ic,nn_dE=dEn,nn_fe=fen,nn_bounded=okn,
                     analytic_front=[(e,d,f,o) for e,d,f,o in front], fe_equal_acc=fe_eq))

# verdict
def cost_ratio(r): return (r['nn_fe']/r['fe_equal_acc']) if r['fe_equal_acc'] else None
ratios={r['ic']:cost_ratio(r) for r in rows}
wins=[ic for ic,x in ratios.items() if x is not None and x<0.85]
ties=[ic for ic,x in ratios.items() if x is not None and 0.85<=x<=1.18]
log("\n"+"="*70)
log(f"cost ratios NN/analytic@equal-accuracy: {ratios}")
if len(wins)>=2: verdict="CLICK (NN materially cheaper on held-out)"
elif wins: verdict="PARTIAL (NN cheaper on some ICs)"
elif len(ties)>=2: verdict="NULL — NN ~matches analytic t_dyn criterion; ship analytic (Occam)"
else: verdict="PARTIAL/MIXED"
log(f"TRACK 3 VERDICT: {verdict}")
open(OUT+"/track3_results.txt","w").write("\n".join(L)+"\n")
json.dump(dict(verdict=verdict,rows=rows,ratios=ratios),open(OUT+"/track3_results.json","w"),indent=2,default=str)
print("TRACK3_EVAL_DONE")
