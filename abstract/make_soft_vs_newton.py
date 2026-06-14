import sys, os, json, math, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT+"/src")
import numpy as np
import simon_core as sc, integrators_symplectic as si
T, NS, EPS = 100.0, 5000, sc.EPS
OUT = os.path.join(ROOT, "abstract")
t_start = time.time()

def binary_single_ic(r_bin, G):
    m = np.array([1.0,0.5,0.1]); mb = 1.5
    x = np.array([[-r_bin*0.5/mb,0,0],[r_bin*1.0/mb,0,0],[3.0,0,0]], float)
    vrel = math.sqrt(G*mb/r_bin); v2 = math.sqrt(G*mb/3.0)
    v = np.array([[0,-vrel*0.5/mb,0],[0,vrel*1.0/mb,0],[0,v2,0]], float)
    M = m.sum(); x -= (m[:,None]*x).sum(0)/M; v -= (m[:,None]*v).sum(0)/M
    return m, x, v

def lf(m,x0,v0,G,mode,dt,adaptive=False):
    _,p,v,info = sc.simulate_leapfrog(m,x0,v0,dt,T,NS,G,mode=mode,adaptive=adaptive,adapt_thresh=0.05,max_substeps=16)
    de = sc.max_dE(p,v,m,G, eps=(EPS if mode=="soft" else 0.0))
    return dict(de=de, bnd=sc.bounded(p,m), fe=int(info["force_evals"]))

def tsalf(m,x0,v0,G,mode,eta=0.05):
    _,p,v,info = si.tsalf_simulate(m,x0,v0,eta,T,NS,G,mode=mode,max_steps=2_500_000)
    ts,xs,vs = info["step_states"]
    de = sc.max_dE(xs,vs,m,G, eps=(EPS if mode=="soft" else 0.0))
    return dict(de=de, bnd=sc.bounded(p,m), fe=int(info["force_evals"]), done=bool(info["completed"]))

def safe(fn,*a,**k):
    try: return fn(*a,**k)
    except Exception as e: return dict(err=str(e)[:60])

TOY = [("IC1",sc.G_TOY),("IC3",sc.G_TOY),("IC4",sc.G_TOY),("IC6",sc.G_TOY)]
toy_ic = {ic: sc.get_ic(ic) for ic,_ in TOY}
BIN = [0.05,0.10,0.20,0.40]
bin_ic = {f"BS_{rb:.2f}": binary_single_ic(rb, sc.G_REAL) for rb in BIN}
pd = json.load(open(ROOT+"/experiments/weekend_phaseD/phaseD_data.json"))
pd_tsalf = {r["config"]: r for r in pd["results"] if r["method"]=="tsalf"}

LF_METHODS = [("leapfrog dt=0.04",0.04,False),("leapfrog dt=0.005",0.005,False),("heuristic-adaptive",0.04,True)]
rows = []
def emit(cfg,m,x0,v0,G,is_toy):
    for label,dt,adaptive in LF_METHODS:
        rs = safe(lf,m,x0,v0,G,"soft",dt,adaptive); rn = safe(lf,m,x0,v0,G,"newton",dt,adaptive)
        rows.append(dict(cfg=cfg,method=label,
            ds=rs.get("de"),dn=rn.get("de"),bs=rs.get("bnd"),bn=rn.get("bnd"),fes=rs.get("fe"),fen=rn.get("fe"),note=""))
        print(f"  {cfg} {label}: soft={rs.get('de')} newton={rn.get('de')} [{time.time()-t_start:.0f}s]",flush=True)
    if is_toy:
        rs = safe(tsalf,m,x0,v0,G,"soft"); rn = safe(tsalf,m,x0,v0,G,"newton")
        nt = ("" if rn.get("done",True) else " (newton PARTIAL: step-cap)")+("" if rs.get("done",True) else " (soft PARTIAL)")
        rows.append(dict(cfg=cfg,method="tsalf (eta=0.05)",
            ds=rs.get("de"),dn=rn.get("de"),bs=rs.get("bnd"),bn=rn.get("bnd"),fes=rs.get("fe"),fen=rn.get("fe"),note=nt))
        print(f"  {cfg} tsalf: soft={rs.get('de')} newton={rn.get('de')}{nt} [{time.time()-t_start:.0f}s]",flush=True)
    else:
        r = pd_tsalf[cfg]
        rows.append(dict(cfg=cfg,method="tsalf (eta=0.05)",ds=r["max_dE_pct"],dn="soft-only",bs=r["bounded"],bn="n/a",fes=r["total_fe"],fen=None,note=" (binary tsalf: soft from phaseD; eps negligible at r_bin)"))

for ic,G in TOY: m,x0,v0 = toy_ic[ic]; emit(ic,m,x0,v0,G,True)
for rb in BIN: c=f"BS_{rb:.2f}"; m,x0,v0 = bin_ic[c]; emit(c,m,x0,v0,sc.G_REAL,False)

# ---- Q3: matched-accuracy ratios, soft vs newton, toy configs ----
def pareto(points):
    pts=sorted(points,key=lambda z:z[0]); fr=[]; best=float('inf')
    for fe,acc in pts:
        if np.isfinite(acc) and acc<best-1e-18: best=acc; fr.append((float(fe),float(acc)))
    return fr
def fe_at(fr,t):
    if not fr: return None
    fes=[f for f,_ in fr]; accs=[a for _,a in fr]
    if t>=accs[0]: return fes[0]
    if t<accs[-1]: return None
    for i in range(len(fr)-1):
        a0,a1,f0,f1=accs[i],accs[i+1],fes[i],fes[i+1]
        if a0>=t>=a1: lt=math.log(t); return float(math.exp(math.log(f0)+(math.log(f1)-math.log(f0))*(lt-math.log(a0))/(math.log(a1)-math.log(a0))))
    return None
LF_DT=[0.04,0.02,0.01,0.005,0.0025]; TS_ETA=[0.2,0.1,0.05,0.02,0.01]
ratios={}
for ic,G in TOY:
    m,x0,v0=toy_ic[ic]
    for mode in ["soft","newton"]:
        lfpts=[]; tspts=[]
        for dt in LF_DT:
            r=safe(lf,m,x0,v0,G,mode,dt,False)
            if r.get("bnd") and np.isfinite(r.get("de",float("inf"))): lfpts.append((r["fe"],r["de"]))
        for e in TS_ETA:
            r=safe(tsalf,m,x0,v0,G,mode,e)
            if r.get("bnd") and np.isfinite(r.get("de",float("inf"))): tspts.append((r["fe"],r["de"]))
        lff=pareto(lfpts); tsf=pareto(tspts); rr={}
        for tgt in [0.3,0.1,0.03]:
            a=fe_at(lff,tgt); b=fe_at(tsf,tgt); rr[tgt]=(round(a/b,2) if (a and b) else None)
        ratios[(ic,mode)]=rr
        print(f"  ratios {ic}/{mode}: {rr} [{time.time()-t_start:.0f}s]",flush=True)

# ---- write table + verdict ----
def pct(x):
    if x is None or isinstance(x,str): return str(x)
    if not np.isfinite(x): return "inf(diverged)"
    return f"{x:.4g}%"
def bb(x): return "✓" if x is True else ("✗" if x is False else str(x))
L=[]
L.append("SOFT vs NEWTON force-model diff — leapfrog-family rows. T=100 yr. vs IAS15 reference.")
L.append("soft = softened close pairs (Plummer eps=3e-4, only for separations < NN_THRESH=0.15 AU); newton = pure unsoftened Newtonian (the advocated method).")
L.append("Energy diagnostic matches the dynamics: soft uses eps=3e-4; newton uses eps=0 (true Newtonian invariant). bounded = max distance from COM < 10 AU.")
L.append("tsalf is reported under its comparison force model = SOFT; toy ICs also re-run in newton to answer the verdict. Binary tsalf-soft reused from phaseD (eps negligible at r_bin).")
L.append("Note: soft==newton EXACTLY whenever no pair ever closes below 0.15 AU (softening never triggers).")
L.append("="*122)
L.append(f"{'config':8} {'method':20} {'|dE/E0|max soft':>16} {'|dE/E0|max newton':>18} {'Δ (newton−soft)':>16} {'bnd s→n':>9} {'force-evals s/n':>18}")
L.append("-"*122)
last=None
for r in rows:
    if last is not None and r['cfg']!=last: L.append("-"*122)
    last=r['cfg']
    ds,dn=r['ds'],r['dn']
    if isinstance(ds,(int,float)) and isinstance(dn,(int,float)) and np.isfinite(ds) and np.isfinite(dn):
        d=dn-ds; delta=(f"{d:+.4g} pp" if abs(d)>=1e-6 else "~0 (identical)")
    else: delta="—"
    fe=(f"{r['fes']}" if r['fen'] is None else (f"{r['fes']}" if r['fes']==r['fen'] else f"{r['fes']}/{r['fen']}"))
    L.append(f"{r['cfg']:8} {r['method']:20} {pct(ds):>16} {pct(dn):>18} {delta:>16} {bb(r['bs'])+'→'+bb(r['bn']):>9} {fe:>18}{r['note']}")
L.append("="*122)

def trow(cfg):
    for r in rows:
        if r['cfg']==cfg and r['method'].startswith('tsalf'): return r
    return None
L.append("VERDICT")
L.append("-"*122)
# Q1
L.append("Q1 — tsalf bounded across IC1/IC4/IC6 in NEWTON mode, and do drifts (soft 0.087/0.224/0.003%) change?")
for ic,anchor in [("IC1",0.087),("IC4",0.224),("IC6",0.003)]:
    r=trow(ic)
    L.append(f"     {ic}: soft={pct(r['ds'])} (anchor {anchor}%)  newton={pct(r['dn'])}  bounded soft→newton {bb(r['bs'])}→{bb(r['bn'])}{r['note']}")
# Q2
def getrow(cfg,meth):
    for r in rows:
        if r['cfg']==cfg and r['method']==meth: return r
ic1_005=getrow("IC1","leapfrog dt=0.005")
L.append("")
L.append("Q2 — Does fixed dt=0.005 on IC1 still eject in newton, and at what %? (soft was 128%)")
L.append(f"     IC1 leapfrog dt=0.005: soft={pct(ic1_005['ds'])} (bnd {bb(ic1_005['bs'])})  newton={pct(ic1_005['dn'])} (bnd {bb(ic1_005['bn'])})")
# Q3
L.append("")
L.append("Q3 — Do the matched-accuracy efficiency ratios (tsalf vs fixed leapfrog; soft was IC1 1.1–3.6×, IC4 1.5–2.9×, IC3/IC6 0.3–1.5×) move under newton?")
L.append(f"     {'config':6} {'target |dE|':>11} {'ratio SOFT':>11} {'ratio NEWTON':>13}")
for ic,_ in TOY:
    for tgt in [0.3,0.1,0.03]:
        s=ratios.get((ic,'soft'),{}).get(tgt); n=ratios.get((ic,'newton'),{}).get(tgt)
        L.append(f"     {ic:6} {str(tgt)+'%':>11} {(str(s)+'x' if s else '--'):>11} {(str(n)+'x' if n else '--'):>13}")
L.append("")
L.append("FLAG FOR ABSTRACT: any row above where Δ is large or bounded flips soft→newton, or where a ratio moves materially, needs the abstract number updating.")
L.append(f"[wall {time.time()-t_start:.0f}s]")
open(os.path.join(OUT,"soft_vs_newton_diff.txt"),"w").write("\n".join(L)+"\n")
print("DONE_SOFT_NEWTON",flush=True)
