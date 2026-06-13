import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "abstract")
d = json.load(open(os.path.join(ROOT, "experiments/weekend_phaseD/phaseD_data.json")))
res = d["results"]

METHODS = [("ias15","ias15 (ref)","#111111"),("tsalf","tsalf eta=0.05","#1f77b4"),
           ("leapfrog_0.04","leapfrog dt=0.04","#ff7f0e"),("leapfrog_0.005","leapfrog dt=0.005","#d62728"),
           ("heuristic","heuristic-adaptive","#2ca02c")]
ORDER=[m[0] for m in METHODS]; mname={m[0]:m[1] for m in METHODS}; mcol={m[0]:m[2] for m in METHODS}
CONFIGS=["IC1","IC3","IC4","IC6","BS_0.05","BS_0.10","BS_0.20","BS_0.40","SEM"]
def get(cfg,meth): return next((r for r in res if r["config"]==cfg and r["method"]==meth),None)

# ---------- TABLE ----------
L=[]
L.append("SIMON OPERATING-ENVELOPE FRONTIER  —  abstract table")
L.append("panel per (config,method): max|dE/E0| (softened PE) | force-evals/yr (cost) | bounded | pair-range[AU] (hierarchical only)")
L.append("methods: ias15 (15th-order Gauss-Radau reference) | tsalf (time-symmetric adaptive leapfrog, eta=0.05) | leapfrog fixed dt=0.04 | leapfrog fixed dt=0.005 | heuristic inverse-distance adaptive")
L.append("setup: toy ICs softened eps=3e-4 G=1 ; binary-single masses[1,0.5,0.1] G=4pi^2 perturber@3AU ; SEM=Newtonian point masses ; T=100 yr")
L.append("="*108)
L.append(f"{'config':8} {'method':20} {'max|dE/E0|':>13} {'fe/yr':>9} {'bnd':>6}   {'pair_range[AU]':>24}")
for cfg in CONFIGS:
    L.append("-"*108)
    for meth in ORDER:
        r=get(cfg,meth)
        if not r: L.append(f"{cfg:8} {mname[meth]:20} {'(missing)':>13}"); continue
        pr=r.get("pair_range"); prs=f"[{pr[0]:.5f}, {pr[1]:.5f}]" if (pr and r['cls'] in('binary','sem')) else "-"
        L.append(f"{cfg:8} {mname[meth]:20} {r['max_dE_pct']:>12.4g}% {r['fe_per_yr']:>9.1f} {('ok' if r['bounded'] else 'EJECT'):>6}   {prs:>24}")
# ---------- three-regime verdict ----------
def g(cfg,meth,k): r=get(cfg,meth); return r[k] if r else None
L.append("="*108); L.append("THREE-REGIME ENVELOPE (read off the table):")
L.append(f"  REGULAR / wide (IC6): cheap leapfrog dt=0.04 already bounded at {g('IC6','leapfrog_0.04','max_dE_pct'):.4g}% for {g('IC6','leapfrog_0.04','fe_per_yr'):.0f} fe/yr; ias15 needs {g('IC6','ias15','fe_per_yr'):.0f} fe/yr for no practical gain.")
L.append(f"  MODERATE scattering (IC1,IC4): tsalf bounded at {g('IC1','tsalf','max_dE_pct'):.3g}%/{g('IC1','tsalf','fe_per_yr'):.0f}fe and {g('IC4','tsalf','max_dE_pct'):.3g}%/{g('IC4','tsalf','fe_per_yr'):.0f}fe ;")
L.append(f"      fixed leapfrog dt=0.005 EJECTS IC1 ({g('IC1','leapfrog_0.005','max_dE_pct'):.4g}%, bnd={get('IC1','leapfrog_0.005')['bounded']}) at {g('IC1','leapfrog_0.005','fe_per_yr'):.0f}fe ; ias15 bounded but at {g('IC1','ias15','fe_per_yr'):.0f}fe.  -> tsalf is the cheap bounded option.")
L.append(f"  TIGHT BINARY (BS_0.05): only tsalf & ias15 survive; ias15 wall {g('BS_0.05','ias15','wall_s'):.3f}s vs tsalf {g('BS_0.05','tsalf','wall_s'):.3f}s ; heuristic & fixed leapfrog EJECT.  -> ias15 wins/ties tight binaries.")
open(os.path.join(OUT,"frontier_table.txt"),"w").write("\n".join(L)+"\n")
print("\n".join(L))

# ---------- FIGURE ----------
fig,(axA,axB)=plt.subplots(1,2,figsize=(13.5,5.4))
mk={"existing":"o","binary":"s","sem":"^"}
for r in res:
    if r["method"] not in ORDER: continue
    x=r["fe_per_yr"]; y=max(r["max_dE_pct"],1e-7); c=mcol[r["method"]]; m=mk[r["cls"]]
    if r["bounded"]: axA.scatter(x,y,c=c,marker=m,s=48,edgecolors='k',linewidths=0.4,zorder=3)
    else:            axA.scatter(x,y,facecolors='none',edgecolors=c,marker=m,s=60,linewidths=1.5,zorder=3)
axA.set_xscale("log"); axA.set_yscale("log")
axA.axhspan(100,axA.get_ylim()[1] if axA.get_ylim()[1]>100 else 1e6, color='red', alpha=0.05)
axA.axhline(100,color='grey',ls='--',lw=1); axA.text(axA.get_xlim()[0]*1.1,140,"ejection regime",fontsize=7,color='grey')
axA.set_xlabel("force-evaluations / simulated yr   (cost)"); axA.set_ylabel("max |dE/E0|   (%)")
axA.set_title("Cost vs energy error\n(filled = bounded, hollow = ejected)",fontsize=10)
mh=[Line2D([0],[0],marker='o',color='w',markerfacecolor=mcol[m],markeredgecolor='k',label=mname[m],markersize=8) for m in ORDER]
ch=[Line2D([0],[0],marker=v,color='grey',linestyle='',label=k2,markersize=8) for v,k2 in [("o","toy IC"),("s","binary-single"),("^","Sun-Earth-Moon")]]
l1=axA.legend(handles=mh,loc='lower right',fontsize=7.5,title="method"); axA.add_artist(l1)
axA.legend(handles=ch,loc='upper left',fontsize=7.5,title="config class")

grid=np.ones((len(ORDER),len(CONFIGS),3))
for i,meth in enumerate(ORDER):
    for j,cfg in enumerate(CONFIGS):
        r=get(cfg,meth)
        if r: grid[i,j]=[0.80,0.93,0.80] if r["bounded"] else [0.97,0.78,0.78]
axB.imshow(grid,aspect='auto')
for i,meth in enumerate(ORDER):
    for j,cfg in enumerate(CONFIGS):
        r=get(cfg,meth)
        if r: axB.text(j,i,f"{r['max_dE_pct']:.2g}",ha='center',va='center',fontsize=6.3)
axB.set_xticks(range(len(CONFIGS))); axB.set_xticklabels(CONFIGS,rotation=45,ha='right',fontsize=8)
axB.set_yticks(range(len(ORDER))); axB.set_yticklabels([mname[m] for m in ORDER],fontsize=8)
axB.set_title("Survival map: bounded (green) vs ejected (red)\ncell = max|dE/E0| %",fontsize=10)
for x in [3.5,7.5]: axB.axvline(x,color='white',lw=2)
fig.suptitle("SIMON operating-envelope frontier  —  toy ICs | binary-single | real Sun-Earth-Moon  (T=100 yr)",fontsize=11)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(OUT,"frontier.png"),dpi=150)
print("\nwrote abstract/frontier_table.txt and abstract/frontier.png")
