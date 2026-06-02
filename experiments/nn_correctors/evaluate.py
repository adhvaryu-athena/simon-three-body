"""
Stage 3: decisive equal-compute comparison on HELD-OUT configs.
Methods: A1 (c_opt-corrected), A2 (residual-corrected), B (plain leapfrog dt=0.04),
C (plain leapfrog finer-dt frontier). Metrics: |dE/E0|max, short-horizon RMS (<=2/lambda),
time-to-divergence, pair fidelity. Equal-compute: charge NN at kappa FLOP-per-call;
A1/A2 beat physics only if BELOW the plain-leapfrog frontier at a defensible kappa.
"""
import sys, os, json, math
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT+"/src"); sys.path.insert(0, ROOT+"/experiments/nn_correctors")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import corr_lib as cl

OUT = ROOT+"/experiments/nn_correctors"
E = json.load(open(OUT+"/eval_set.json")); DT=E["DT"]; T=E["T"]; NS=E["NS_EVAL"]; W=E["W"]
DT_FRONTIER=[0.04,0.02,0.01,0.005,0.0025]
KAPPAS=[0,10,90]   # NN FLOP cost per call, in force-eval units (charitable / large-N / N=3-actual)
silu=lambda z: z/(1+np.exp(-z))
def fwd2(Wd,feat):
    h=(feat-Wd["mu"])/Wd["sd"]; h=silu(h@Wd["0.weight"].T+Wd["0.bias"]); h=silu(h@Wd["2.weight"].T+Wd["2.bias"])
    return h@Wd["4.weight"].T+Wd["4.bias"]
w1=dict(np.load(OUT+"/a1_weights.npz"))
have_a2=os.path.exists(OUT+"/a2_weights.npz")
if have_a2:
    w2=dict(np.load(OUT+"/a2_weights.npz"))
    def res_predict(f): return (fwd2(w2,f)*w2["ysd"]+w2["ymu"]).astype(np.float64)
def c_predict(f): return float(np.exp(fwd2(w1,f)[0]))

def run_methods(name,m,x0,v0,pref,lam):
    res={}
    # B + C frontier
    for dt in DT_FRONTIER:
        _,p,v,info=cl.leapfrog(m,x0,v0,dt,T,NS)
        res[("LF",dt)]=dict(fe=info["fe"],nn=0,
            maxdE=cl.maxdE(p,v,m), short=cl.short_horizon_rms(p,pref,np.linspace(0,T,NS),lam),
            ttd=cl.time_to_diverge(p,pref,np.linspace(0,T,NS)), pair=cl.pair_fidelity(p,pref,m),
            bounded=bool(np.all(np.isfinite(p))))
    # A1
    _,p,v,info=cl.leapfrog(m,x0,v0,DT,T,NS,c_predict=c_predict)
    res["A1"]=dict(fe=info["fe"],nn=info["nn_calls"],maxdE=cl.maxdE(p,v,m),
        short=cl.short_horizon_rms(p,pref,np.linspace(0,T,NS),lam),
        ttd=cl.time_to_diverge(p,pref,np.linspace(0,T,NS)),pair=cl.pair_fidelity(p,pref,m),
        bounded=bool(np.all(np.isfinite(p))))
    # A2
    if have_a2:
        _,p,v,info=cl.leapfrog(m,x0,v0,DT,T,NS,res_predict=res_predict,window=W,gate=cl.NN_THRESH)
        res["A2"]=dict(fe=info["fe"],nn=info["nn_calls"],maxdE=cl.maxdE(p,v,m),
            short=cl.short_horizon_rms(p,pref,np.linspace(0,T,NS),lam),
            ttd=cl.time_to_diverge(p,pref,np.linspace(0,T,NS)),pair=cl.pair_fidelity(p,pref,m),
            bounded=bool(np.all(np.isfinite(p))))
    return res

def frontier_at(fe_target, metric, res):
    pts=sorted([(res[("LF",dt)]["fe"],max(res[("LF",dt)][metric],1e-9)) for dt in DT_FRONTIER])
    env=[]; best=1e18
    for fe,mt in pts:
        if mt<best-1e-15: env.append((fe,mt)); best=mt
    if fe_target<=env[0][0]: return env[0][1]
    if fe_target>=env[-1][0]: return env[-1][1]
    for k in range(len(env)-1):
        f0,m0=env[k]; f1,m1=env[k+1]
        if f0<=fe_target<=f1:
            r=(math.log(fe_target)-math.log(f0))/(math.log(f1)-math.log(f0)+1e-30)
            return math.exp(math.log(m0)+r*(math.log(m1)-math.log(m0)))
    return env[-1][1]

# ---- run all held-out ----
configs=list(E["eval_set"].keys())
allres={}
for name in configs:
    d=E["eval_set"][name]; m=np.array(d["m"]); x0=np.array(d["x0"]); v0=np.array(d["v0"])
    pref=np.array(d["ias15_pos"]); lam=float(d["lam"])
    allres[name]=dict(res=run_methods(name,m,x0,v0,pref,lam), lam=lam, ias15_steps=d["ias15_steps"])
    print(f"[eval] {name} done (lam={lam:.3f})", flush=True)

# ---- table + verdict ----
L=[]; log=lambda s:(print(s),L.append(s))
log("NN CORRECTORS vs EQUAL-COMPUTE PHYSICS — held-out configs (IC1/IC4/IC6 = canonical OOD; TEST = unseen).")
log("Metrics: |dE/E0|max %, short-horizon RMS (<=2/lambda), time-to-divergence (yr), pair fidelity (rel).")
log("Verdict: corrector CLICKS only if BELOW plain-leapfrog frontier at equal compute (kappa = NN cost/force-eval).")
log("="*120)
hdr=f"{'config':<8}{'method':<10}{'fe':>8}{'nn':>6}{'|dE|max%':>12}{'short_rms':>11}{'ttd(yr)':>9}{'pairfid':>9}"
log(hdr); log("-"*len(hdr))
def fmt(r):
    dE="inf" if not np.isfinite(r["maxdE"]) else f"{r['maxdE']:.3g}"
    return f"{r['fe']:>8}{r['nn']:>6}{dE:>12}{r['short']:>11.4f}{r['ttd']:>9.2f}{r['pair']:>9.3f}"
for name in configs:
    R=allres[name]["res"]
    log(f"{name:<8}{'B(LF.04)':<10}"+fmt(R[('LF',0.04)]))
    log(f"{name:<8}{'C(LF.005)':<10}"+fmt(R[('LF',0.005)]))
    log(f"{name:<8}{'A1':<10}"+fmt(R['A1']))
    if have_a2: log(f"{name:<8}{'A2':<10}"+fmt(R['A2']))
    log("-"*len(hdr))

# frontier comparison: for each approach/metric/kappa, fraction of held-out where it beats the frontier
def verdict_for(app):
    out={}
    for metric in ["maxdE","short"]:
        for kap in KAPPAS:
            wins=0; tot=0
            for name in configs:
                R=allres[name]["res"]
                if app not in R: continue
                a=R[app];
                if not a["bounded"] or not np.isfinite(a[metric]): continue
                fe_eff=a["fe"]+kap*a["nn"]; fr=frontier_at(fe_eff,metric,R)
                tot+=1; wins+= (a[metric] < 0.9*fr)   # "beats" = >10% better than equal-compute frontier
            out[(metric,kap)]=(wins,tot)
    return out
v1=verdict_for("A1"); v2=verdict_for("A2") if have_a2 else {}
log("\nEQUAL-COMPUTE FRONTIER TEST (wins/total held-out where corrector beats finer-step physics by >10%):")
for app,vv in [("A1",v1)]+([("A2",v2)] if have_a2 else []):
    log(f"  {app}:")
    for metric in ["maxdE","short"]:
        for kap in KAPPAS:
            w,tt=vv[(metric,kap)]; log(f"     {metric:<7} kappa={kap:<3}: {w}/{tt} configs beat frontier")
def click(vv):
    # CLICK if beats frontier on majority of held-out at a defensible kappa (>=10) on either metric
    for metric in ["maxdE","short"]:
        w,tt=vv[(metric,10)]
        if tt>0 and w>0.6*tt: return True
    return False
A1_click=click(v1); A2_click=click(v2) if have_a2 else False
log(f"\nVERDICT A1 (c_opt): {'CLICK' if A1_click else 'NULL'}")
log(f"VERDICT A2 (residual): {'CLICK' if A2_click else ('NULL' if have_a2 else 'NO MODEL (insufficient data)')}")
# head-to-head
if have_a2:
    a1better=sum(allres[n]['res']['A1']['short']<allres[n]['res']['A2']['short'] for n in configs if 'A2' in allres[n]['res'])
    log(f"Head-to-head (short-horizon RMS): A1 better on {a1better}/{len(configs)} held-out")
open(OUT+"/results_table.txt","w").write("\n".join(L)+"\n")
def strk(d): return {(f"{k[0]}_{k[1]}" if isinstance(k,tuple) else str(k)):v for k,v in d.items()}
res_json={n:strk(allres[n]['res']) for n in configs}
v1j={f"{m}_k{k}":list(val) for (m,k),val in v1.items()}
v2j={f"{m}_k{k}":list(val) for (m,k),val in v2.items()} if have_a2 else {}
json.dump(dict(results=res_json,v1=v1j,v2=v2j,A1_click=A1_click,A2_click=A2_click),
          open(OUT+"/results.json","w"),default=str,indent=1)

# ---- plots: energy & short-horizon vs compute (frontier + A1/A2), per a representative held-out set ----
def plot_metric(metric,ylabel,fn):
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    for ax,name in zip(axes,["IC1","IC4","IC6"]):
        R=allres[name]["res"]
        fes=[R[("LF",dt)]["fe"] for dt in DT_FRONTIER]; mts=[max(R[("LF",dt)][metric],1e-9) for dt in DT_FRONTIER]
        ax.loglog(fes,mts,"o-",color="#2563A6",label="plain leapfrog (B/C frontier)")
        for app,col in [("A1","#16A34A"),("A2","#DC2626")]:
            if app in R:
                a=R[app]
                for kap,mk in [(0,"o"),(10,"s"),(90,"x")]:
                    ax.scatter(a["fe"]+kap*a["nn"],max(a[metric],1e-9),color=col,marker=mk,s=60,zorder=5)
                ax.scatter([],[],color=col,label=app)
        ax.set_title(f"{name} (lam={allres[name]['lam']:.2f})"); ax.set_xlabel("force evals (compute)"); ax.set_ylabel(ylabel)
        ax.grid(True,which="both",alpha=0.3)
    axes[0].legend(fontsize=7,title="A1/A2 markers: o=k0 s=k10 x=k90")
    fig.suptitle(f"{ylabel} vs compute — correctors must fall BELOW the leapfrog frontier to win")
    fig.tight_layout(); fig.savefig(OUT+"/"+fn,dpi=160); plt.close()
plot_metric("maxdE","|dE/E0|max %","energy_vs_compute.png")
plot_metric("short","short-horizon RMS (AU)","shorthorizon_vs_compute.png")
# time-to-divergence bar
fig,ax=plt.subplots(figsize=(9,4.6)); names=configs[:8]; xb=np.arange(len(names)); w=0.2
for i,(app,lab,col) in enumerate([(("LF",0.04),"B",("#2563A6")),(("LF",0.005),"C",("#7C2D12")),("A1","A1","#16A34A")]+([("A2","A2","#DC2626")] if have_a2 else [])):
    ax.bar(xb+(i-1.5)*w,[allres[n]["res"][app]["ttd"] for n in names],w,label=lab,color=col)
ax.set_xticks(xb); ax.set_xticklabels(names,rotation=45,fontsize=7); ax.set_ylabel("time-to-divergence (yr)")
ax.set_title("Time-to-divergence (higher=better)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT+"/time_to_divergence.png",dpi=160); plt.close()
print("EVAL_DONE")
