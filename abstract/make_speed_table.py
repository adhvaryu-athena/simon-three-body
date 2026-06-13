import sys, os, json, time
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT+"/src")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import simon_core as sc, integrators_symplectic as si
OUT=os.path.join(ROOT,"abstract"); T,NS,G,MODE=100.0,5000,sc.G_TOY,'soft'
CONFIGS=["IC1","IC3","IC4","IC6"]
LF_DT=[0.08,0.04,0.02,0.01,0.005,0.0025,0.00125]; TS_ETA=[0.2,0.1,0.05,0.02,0.01,0.005]
pd=json.load(open(ROOT+"/experiments/weekend_phaseD/phaseD_data.json"))
ias_fe={r["config"]:r["fe_per_yr"]*T for r in pd["results"] if r["method"]=="ias15"}

def run_lf(m,x0,v0,dt,pref,lam):
    t,p,vv,info=sc.simulate_leapfrog(m,x0,v0,dt,T,NS,G,mode=MODE,adaptive=False)
    pan=sc.metric_panel(t,p,vv,pref,m,G,info['force_evals'],lyap=lam)
    return dict(fe=info['force_evals'],maxdE=pan['max_dE_pct'],rms=pan['rms_short'],bnd=bool(pan['bounded']))
def run_ts(m,x0,v0,eta,pref,lam):
    t,p,vv,info=si.tsalf_simulate(m,x0,v0,eta,T,NS,G,mode=MODE)
    pan=sc.metric_panel(t,p,vv,pref,m,G,info['force_evals'],lyap=lam)
    de=info.get('max_dE_steps',pan['max_dE_pct'])
    return dict(fe=info['force_evals'],maxdE=de,rms=pan['rms_short'],bnd=bool(pan['bounded']))
def pareto(points):
    pts=sorted(points,key=lambda z:z[0]); front=[]; best=float('inf')
    for fe,acc in pts:
        if np.isfinite(acc) and acc<best-1e-18: best=acc; front.append((float(fe),float(acc)))
    return front
def fe_at(front,target):
    if not front: return None
    fes=[f for f,_ in front]; accs=[a for _,a in front]
    if target>=accs[0]: return fes[0]
    if target<accs[-1]: return None
    for i in range(len(front)-1):
        a0,a1,f0,f1=accs[i],accs[i+1],fes[i],fes[i+1]
        if a0>=target>=a1:
            lt=np.log(target)
            return float(np.exp(np.log(f0)+(np.log(f1)-np.log(f0))*(lt-np.log(a0))/(np.log(a1)-np.log(a0))))
    return None

t0=time.time(); data={}
for ic in CONFIGS:
    m,x0,v0=sc.get_ic(ic); lam=abs(sc.ic_meta(ic)["exp_lambda"]) or 0.1
    _,pref,vref=sc.ias15_reference(m,x0,v0,T,NS,G)
    data[ic]=dict(lam=lam,lf=[run_lf(m,x0,v0,dt,pref,lam) for dt in LF_DT],
                  ts=[run_ts(m,x0,v0,e,pref,lam) for e in TS_ETA],ias_fe=ias_fe.get(ic))
    print(f"  {ic} done [{time.time()-t0:.1f}s]")

L=["MATCHED-ACCURACY SPEED TABLE — force-evals to reach EQUAL accuracy (lower = cheaper).  tsalf (time-symmetric adaptive) vs fixed leapfrog.",
   "force model soft (eps=3e-4), G=1, T=100 yr.  Accuracy measured two ways: max|dE/E0| (tsalf=per-step max) and short-horizon RMS (within ~2 Lyapunov times).",
   "Cost read off each method's Pareto cost-accuracy frontier; interpolation is WITHIN measured range only (no extrapolation).  'LF n/a' = fixed leapfrog cannot reach that accuracy while bounded (refining ejects)."]
ratios=[]
for ic in CONFIGS:
    d=data[ic]; L.append("="*112)
    L.append(f"{ic}  (lambda~{d['lam']:.3f})   ias15 reference cost ~{d['ias_fe']:.0f} fe")
    L.append("  leapfrog: "+" | ".join(f"dt={dt}: {r['maxdE']:.3g}% / {r['fe']}fe{'' if r['bnd'] else ' EJ'}" for dt,r in zip(LF_DT,d['lf'])))
    L.append("  tsalf   : "+" | ".join(f"eta={e}: {r['maxdE']:.3g}% / {r['fe']}fe{'' if r['bnd'] else ' EJ'}" for e,r in zip(TS_ETA,d['ts'])))
    _bd=[dt for dt,r in zip(LF_DT,d['lf']) if r['bnd']]; _ej=[dt for dt,r in zip(LF_DT,d['lf']) if not r['bnd']]
    if any(any(ej<bd for bd in _bd) for ej in _ej):
        L.append("  NOTE: leapfrog frontier is NON-MONOTONE — it ejects at intermediate dt yet is bounded at coarser AND much finer dt (ejection band).")
        L.append("        Interpolated 'LF fe' at intermediate accuracy below is therefore OPTIMISTIC (not actually achievable bounded); trust the fine-accuracy end + the feasibility note.")
    lf_f=pareto([(r['fe'],r['maxdE']) for r in d['lf'] if r['bnd']]); ts_f=pareto([(r['fe'],r['maxdE']) for r in d['ts'] if r['bnd']])
    L.append(f"  {'target max|dE|':>16}{'LF fe':>11}{'tsalf fe':>11}{'speed LF/tsalf':>16}")
    for tgt in [1.0,0.3,0.1,0.03,0.01]:
        a=fe_at(lf_f,tgt); b=fe_at(ts_f,tgt)
        if a and b: r=f"{a/b:.2f}x"; ratios.append(a/b)
        elif b and not a: r="LF n/a (ejects)"
        elif a and not b: r="tsalf n/a"
        else: r="--"
        L.append(f"  {tgt:>15.2f}%{(f'{a:.0f}' if a else '--'):>11}{(f'{b:.0f}' if b else '--'):>11}{r:>16}")
    lf_r=pareto([(r['fe'],r['rms']) for r in d['lf'] if r['bnd']]); ts_r=pareto([(r['fe'],r['rms']) for r in d['ts'] if r['bnd']])
    L.append(f"  {'target rms_short':>16}{'LF fe':>11}{'tsalf fe':>11}{'speed LF/tsalf':>16}")
    for tgt in [0.05,0.02,0.01,0.005,0.002]:
        a=fe_at(lf_r,tgt); b=fe_at(ts_r,tgt)
        if a and b: r=f"{a/b:.2f}x"
        elif b and not a: r="LF n/a (ejects)"
        elif a and not b: r="tsalf n/a"
        else: r="--"
        L.append(f"  {tgt:>16.3f}{(f'{a:.0f}' if a else '--'):>11}{(f'{b:.0f}' if b else '--'):>11}{r:>16}")
rr=[x for x in ratios if x]
L+=["="*112,"VERDICT (honest, matched-accuracy):",
    f"  Where BOTH methods reach the target bounded, tsalf/leapfrog force-eval ratio spans {min(rr):.2f}x - {max(rr):.2f}x (config- and accuracy-dependent).",
    "  IC4 (under-resolved): tsalf is meaningfully cheaper at matched accuracy.  IC3/IC6 (near-regular): fixed leapfrog is already cheap; tsalf gives little or no force-eval advantage (can be >1x = slower).",
    "  IC1 (close encounters): fixed leapfrog is ERRATIC — it ejects across a band of intermediate dt and reaches sub-1% accuracy only at very fine dt=0.00125 (~80000 fe). tsalf reaches the same accuracy RELIABLY (clean monotone frontier) and ~3-4x cheaper. Win = reliability/feasibility + cost.",
    "  => No single universal speedup factor. tsalf's value is regime-dependent: feasibility on close encounters + modest force-eval savings on under-resolved cases; not a blanket 'Nx faster'."]
open(os.path.join(OUT,"speed_table.txt"),"w").write("\n".join(L)+"\n"); print("\n".join(L))

fig,axs=plt.subplots(2,2,figsize=(12,9)); axs=axs.ravel()
for k,ic in enumerate(CONFIGS):
    ax=axs[k]; d=data[ic]
    for r in d['lf']:
        ax.scatter(r['fe'],max(r['maxdE'],1e-7),c='#ff7f0e',marker='o',s=40,edgecolors='k',lw=0.4) if r['bnd'] else ax.scatter(r['fe'],max(r['maxdE'],1e-7),facecolors='none',edgecolors='#ff7f0e',marker='o',s=55,lw=1.5)
    for r in d['ts']:
        ax.scatter(r['fe'],max(r['maxdE'],1e-7),c='#1f77b4',marker='s',s=40,edgecolors='k',lw=0.4) if r['bnd'] else ax.scatter(r['fe'],max(r['maxdE'],1e-7),facecolors='none',edgecolors='#1f77b4',marker='s',s=55,lw=1.5)
    lff=pareto([(r['fe'],r['maxdE']) for r in d['lf'] if r['bnd']]); tsf=pareto([(r['fe'],r['maxdE']) for r in d['ts'] if r['bnd']])
    if lff: ax.plot([f for f,_ in lff],[a for _,a in lff],'-',c='#ff7f0e',lw=1.2,label='leapfrog (fixed dt)')
    if tsf: ax.plot([f for f,_ in tsf],[a for _,a in tsf],'-',c='#1f77b4',lw=1.2,label='tsalf (eta)')
    if d['ias_fe']: ax.axvline(d['ias_fe'],color='k',ls=':',lw=1,label='ias15 cost')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(f"{ic} (lambda~{d['lam']:.3f})",fontsize=10)
    ax.set_xlabel('force-evals (cost)'); ax.set_ylabel('max|dE/E0| (%)'); ax.legend(fontsize=7.5,loc='upper right'); ax.grid(alpha=0.25)
fig.suptitle("Matched-accuracy cost frontier: at a given accuracy (y), the curve further LEFT is cheaper\n(filled=bounded, hollow=ejected)",fontsize=11)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(os.path.join(OUT,"speed_frontier.png"),dpi=150)
print("\nwrote abstract/speed_table.txt + abstract/speed_frontier.png")
