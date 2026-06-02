"""
weekend_phaseD PLOTTING — regenerates all outputs from phaseD_data.json.
 - operating_envelope_table.txt : full table, failures marked, nothing omitted.
 - operating_envelope.png       : cost-vs-drift log-log, one panel per config class.
 - cost_vs_rbin.png             : binary-single fe/yr vs r_bin, all methods.
 - sem_pair_range.png           : SEM Earth-Moon distance over time, all methods.
"""
import os, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = ROOT + "/experiments/weekend_phaseD"
D = json.load(open(OUT+"/phaseD_data.json")); R = D["results"]
METH = ["heuristic","leapfrog_0.04","leapfrog_0.01","leapfrog_0.005","tsalf","ias15"]
COL = {"heuristic":"#DC2626","leapfrog_0.04":"#F59E0B","leapfrog_0.01":"#A16207",
       "leapfrog_0.005":"#7C2D12","tsalf":"#16A34A","ias15":"#2563A6"}
MK  = {"heuristic":"s","leapfrog_0.04":"^","leapfrog_0.01":"v","leapfrog_0.005":"D","tsalf":"o","ias15":"*"}
def get(cfg,meth): return next((r for r in R if r["config"]==cfg and r["method"]==meth), None)

# ---------- TABLE ----------
L=[]; log=lambda s:L.append(s)
log("SIMON OPERATING ENVELOPE — full table (phaseD). NOTHING omitted; failures marked.")
log("force model: existing+binary = SIMON softened (eps=3e-4 negligible); SEM = Newtonian (point masses).")
log("fe/yr = force evals per simulated yr (ias15 = steps*8 estimate). max|dE| = softened PE. PARTIAL = tsalf step-capped.")
log("="*132)
hdr=f"{'config':<9}{'cls':<9}{'method':<15}{'fe/yr':>10}{'wall/yr(s)':>11}{'max|dE/E0|':>13}{'status':>11}{'completed':>10}{'pair_range(AU)':>26}"
log(hdr); log("-"*len(hdr))
order = [c["name"] for c in D["configs"]]
for cfg in order:
    for meth in METH:
        r = get(cfg,meth)
        if r is None: continue
        if "error" in r:
            log(f"{cfg:<9}{'?':<9}{meth:<15}{'ERROR: '+r['error'][:60]}"); continue
        dE = r.get("max_dE_pct")
        dEs = "FAIL" if dE is None else f"{dE:.4g}%"
        if r.get("max_dE_pct") is None: status="FAIL"
        elif not r["bounded"]: status="EJECTED"
        elif r["completed_yr"] < D["T"]-1e-6: status="PARTIAL"
        else: status="ok"
        comp = f"{r['completed_yr']:.1f}yr" if r["completed_yr"]<D["T"]-1e-6 else "100yr"
        pr = r.get("pair_range"); prs = f"[{pr[0]:.5f},{pr[1]:.5f}]" if pr else "-"
        log(f"{cfg:<9}{r['cls']:<9}{meth:<15}{r['fe_per_yr']:>10.1f}{r['wall_per_yr']:>11.4f}{dEs:>13}{status:>11}{comp:>10}{prs:>26}")
    log("-"*len(hdr))
# SEM EM-range callout
log("\nSEM Earth-Moon distance range (catches lunar unbinding global metrics miss):")
for meth in METH:
    r=get("SEM",meth)
    if r and r.get("pair_range"):
        pr=r["pair_range"]; flag="" if pr[1]<0.01 else "  <-- LUNAR ORBIT DISRUPTED"
        log(f"  {meth:<15} EM range [{pr[0]:.5f}, {pr[1]:.5f}] AU{flag}")
open(OUT+"/operating_envelope_table.txt","w").write("\n".join(L)+"\n")
print("wrote table")

# ---------- operating_envelope.png (3 panels by class) ----------
classes=[("existing","Existing ICs (IC1/3/4/6)"),("binary","Binary-single (r_bin 0.05-0.40)"),("sem","Sun-Earth-Moon")]
fig,axes=plt.subplots(1,3,figsize=(16,5.0))
for ax,(cl,title) in zip(axes,classes):
    rr=[r for r in R if r.get("cls")==cl and "error" not in r and r.get("max_dE_pct") is not None]
    for meth in METH:
        pts=[r for r in rr if r["method"]==meth]
        if not pts: continue
        xs=[p["fe_per_yr"] for p in pts]; ys=[max(p["max_dE_pct"],1e-7) for p in pts]
        bnd=[p["bounded"] and p["completed_yr"]>=D["T"]-1e-6 for p in pts]
        ax.scatter([x for x,b in zip(xs,bnd) if b],[y for y,b in zip(ys,bnd) if b],
                   c=COL[meth],marker=MK[meth],s=70,label=meth,edgecolor="k",linewidth=0.4,zorder=3)
        # failed (ejected/partial) = hollow red X
        ax.scatter([x for x,b in zip(xs,bnd) if not b],[y for y,b in zip(ys,bnd) if not b],
                   facecolors="none",edgecolors="red",marker="X",s=110,linewidth=1.6,zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(title)
    ax.set_xlabel("force evals / sim-yr  (cost, log)"); ax.set_ylabel("max |dE/E0|  (%, log)")
    ax.axhline(1.0,ls=":",color="gray",lw=0.8); ax.grid(True,which="both",alpha=0.25)
axes[0].legend(fontsize=7,loc="upper right",title="method (red X = ejected/partial)")
fig.suptitle("SIMON operating envelope: cost vs energy drift (lower-left = better; red X = ejected/unbounded)")
fig.tight_layout(); fig.savefig(OUT+"/operating_envelope.png",dpi=170); plt.close()
print("wrote operating_envelope.png")

# ---------- cost_vs_rbin.png ----------
bs=sorted({r["config"] for r in R if r.get("cls")=="binary"})
rbv=[float(c.split("_")[1]) for c in bs]
fig,ax=plt.subplots(figsize=(7.4,5.0))
for meth in METH:
    ys=[]; xs=[]
    for c,rb in zip(bs,rbv):
        r=get(c,meth)
        if r and "error" not in r:
            xs.append(rb); ys.append(r["fe_per_yr"])
    if xs: ax.loglog(xs,ys,marker=MK[meth],color=COL[meth],lw=1.6,label=meth)
ax.set_xlabel("binary separation r_bin (AU, log)"); ax.set_ylabel("force evals / sim-yr (log)")
ax.set_title("Binary-single cost vs separation"); ax.grid(True,which="both",alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT+"/cost_vs_rbin.png",dpi=170); plt.close()
print("wrote cost_vs_rbin.png")

# ---------- sem_pair_range.png ----------
S=D.get("sem_series",{})
if S:
    fig,ax=plt.subplots(figsize=(8.5,5.0))
    for meth in METH:
        if meth in S:
            ax.semilogy(S[meth]["t"], S[meth]["em"], color=COL[meth], lw=1.4, label=meth, alpha=0.85)
    ax.axhspan(0.00244,0.00271,color="green",alpha=0.10,label="physical lunar range")
    ax.set_xlabel("time (yr)"); ax.set_ylabel("Earth-Moon distance (AU, log)")
    ax.set_title("SEM: Earth-Moon distance over time (lunar unbinding is invisible to global energy)")
    ax.grid(True,which="both",alpha=0.3); ax.legend(fontsize=8,loc="upper left")
    fig.tight_layout(); fig.savefig(OUT+"/sem_pair_range.png",dpi=170); plt.close()
    print("wrote sem_pair_range.png")
print("PHASED_PLOT_DONE")
