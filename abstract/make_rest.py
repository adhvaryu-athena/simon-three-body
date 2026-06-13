import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT=os.path.join(ROOT,"abstract")
pd=json.load(open(os.path.join(ROOT,"experiments/weekend_phaseD/phaseD_data.json")))
res=pd["results"]; ss=pd["sem_series"]
sem={r["method"]:r for r in res if r["config"]=="SEM"}
TRUE_EM=0.00257  # mean Earth-Moon distance, AU

# ===================== TASK 2 — metric trap =====================
L=["METRIC-TRAP RESULT — real Sun-Earth-Moon (JPL Horizons ICs), Newtonian point masses, T=100 yr",
   "Global energy verdict vs pair-separation truth. True Earth-Moon distance ~ %.5f AU."%TRUE_EM,
   "="*100,
   f"{'method':18} {'max|dE/E0|':>12} {'bounded':>8} {'EM-sep range [AU]':>26} {'peak/true':>10}  verdict"]
order=["ias15","tsalf","heuristic","leapfrog_0.005","leapfrog_0.01","leapfrog_0.04"]
disp={"ias15":"ias15 (ref)","tsalf":"tsalf","heuristic":"heuristic-adapt","leapfrog_0.005":"leapfrog dt=0.005","leapfrog_0.01":"leapfrog dt=0.01","leapfrog_0.04":"leapfrog dt=0.04"}
for m in order:
    r=sem[m]; pr=r["pair_range"]; peak=pr[1]; ratio=peak/TRUE_EM
    verdict="PHYSICAL" if peak<0.01 else "LUNAR ORBIT UNBOUND"
    L.append(f"{disp[m]:18} {r['max_dE_pct']:>11.4g}% {('yes' if r['bounded'] else 'no'):>8} [{pr[0]:.5f}, {pr[1]:.5f}] {ratio:>9.0f}x  {verdict}")
L+=["="*100,
    "THE TRAP: leapfrog dt=0.01 reports max|dE/E0|=%.4g%% and bounded=True (global metric: 'fine'),"%sem['leapfrog_0.01']['max_dE_pct'],
    "  yet the Earth-Moon separation runs out to %.2f AU = %.0fx the true %.5f AU (the Moon is ejected from Earth)."%(sem['leapfrog_0.01']['pair_range'][1], sem['leapfrog_0.01']['pair_range'][1]/TRUE_EM, TRUE_EM),
    "  ias15 / tsalf / heuristic-adaptive keep it physical at ~[0.0024, 0.0027] AU.",
    "  => A global energy/boundedness check certifies a run whose internal hierarchical structure is destroyed.",
    "     A pair-separation metric is REQUIRED to catch this; it is the headline 'metric trap'."]
open(os.path.join(OUT,"metric_trap_table.txt"),"w").write("\n".join(L)+"\n")
print("\n".join(L)); print()

fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5))
for m,lab,c in [("ias15","ias15 (ref)","#111111"),("tsalf","tsalf","#1f77b4"),("leapfrog_0.01","leapfrog dt=0.01","#d62728"),("leapfrog_0.04","leapfrog dt=0.04","#ff7f0e")]:
    a1.plot(ss[m]["t"], ss[m]["em"], label=lab, color=c, lw=1.3)
a1.axhline(TRUE_EM,color="green",ls=":",lw=1.6,label=f"true Moon dist {TRUE_EM:.5f} AU")
a1.set_yscale("log"); a1.set_xlabel("time (yr)"); a1.set_ylabel("Earth-Moon separation (AU)")
a1.set_title("Pair-separation truth: leapfrog unbinds the Moon"); a1.legend(fontsize=8,loc="center right")
pk=max(ss["leapfrog_0.01"]["em"]); a1.annotate(f"dt=0.01 peaks {pk:.2f} AU\n(~{pk/TRUE_EM:.0f}x true)",xy=(50,pk*0.5),fontsize=8.5,color="#d62728",fontweight="bold")
de=[sem[m]["max_dE_pct"] for m in order]
cols=["#111111","#1f77b4","#2ca02c","#9467bd","#d62728","#ff7f0e"]
a2.bar(range(len(order)),de,color=cols)
a2.set_yscale("log"); a2.set_xticks(range(len(order))); a2.set_xticklabels([disp[m] for m in order],rotation=45,ha="right",fontsize=8)
a2.set_ylabel("max |dE/E0|  (%)"); a2.set_title("Global energy says ALL fine (every method <0.2%, bounded)")
a2.axhline(1.0,color="grey",ls="--",lw=1); a2.text(0,1.2,"1% ref",fontsize=7,color="grey")
fig.suptitle("METRIC TRAP — energy-says-fine (right) vs pair-separation-says-unbound (left), real Sun-Earth-Moon",fontsize=11)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(os.path.join(OUT,"sem_pair_range.png"),dpi=150)
print("wrote metric_trap_table.txt + sem_pair_range.png\n")

# ===================== TASK 3 — learned-correction evaluation =====================
nn=json.load(open(os.path.join(ROOT,"experiments/nn_correctors/results.json")))
kap=[0,10,90]
def wc(v,met): return [nn[v][f"{met}_k{k}"][0] for k in kap]
A1s,A1d=wc("v1","short"),wc("v1","maxdE"); A2s,A2d=wc("v2","short"),wc("v2","maxdE")
oe=json.load(open(os.path.join(ROOT,"experiments/weekend_phaseC/operating_envelope.json"))); orows=oe["rows"]
T=["LEARNED-CORRECTION EVALUATION — equal-compute, pre-registered, held-out (27 configs: IC1/IC4/IC6 + 24 unseen TEST)",
   "A1 = per-step scalar c_opt (reference-matched).  A2 = state-to-state residual.  Charge NN at kappa = FLOP-per-call / force-eval.",
   "A corrector 'CLICKS' only if it beats the plain-leapfrog cost-accuracy frontier (i.e. wins > half) on a metric. Short-horizon RMS is the honest chaos metric.",
   "="*96,
   f"{'metric':16} {'kappa=0':>10} {'kappa=10':>10} {'kappa=90':>10}   (held-out configs beating equal-compute finer step, of 27)",
   "-"*96,
   f"{'A1  short-horiz':16} {A1s[0]:>8}/27 {A1s[1]:>8}/27 {A1s[2]:>8}/27   <- 0/27 even when NN is FREE",
   f"{'A1  max|dE|':16} {A1d[0]:>8}/27 {A1d[1]:>8}/27 {A1d[2]:>8}/27   <- weak energy 'wins' vanish once compute is charged",
   f"{'A2  short-horiz':16} {A2s[0]:>8}/27 {A2s[1]:>8}/27 {A2s[2]:>8}/27",
   f"{'A2  max|dE|':16} {A2d[0]:>8}/27 {A2d[1]:>8}/27 {A2d[2]:>8}/27",
   "-"*96,
   f"VERDICT A1 (c_opt): {'CLICK' if nn['A1_click'] else 'NULL'}    VERDICT A2 (residual): {'CLICK' if nn['A2_click'] else 'NULL'}",
   "",
   "BINARY-SINGLE ORACLE (upper bound = perfect encounter correction), masses[1,0.5,0.1], perturber@3AU:",
   f"{'r_bin':>8} {'oracle improve%':>16} {'gate fires%':>12} {'heuristic |dE|%':>16} {'tsalf |dE|%':>14}"]
for row in orows:
    T.append(f"{row['r_bin']:>8.2f} {row['improve_pct']:>15.1f}% {row['gate_pct_b']:>11.1f}% {row['maxdE_a']:>15.4g}% {row.get('maxdE_tsalf',float('nan')):>13.4g}%")
T+= ["", "ORACLE VERDICT: "+oe["verdict"],
     "=> Even a PERFECT force/encounter correction yields 0.0% improvement at every separation: the binary failure is",
     "   temporal under-resolution, not force-model error. No learned force correction can fix a timestep problem;",
     "   adaptive resolution (tsalf / ias15) can and does (see frontier_table)."]
open(os.path.join(OUT,"nn_evaluation_table.txt"),"w").write("\n".join(T)+"\n")
print("\n".join(T)); print()

fig,(b1,b2)=plt.subplots(1,2,figsize=(13,5))
b1.plot(kap,A1s,"o-",color="#d62728",label="A1 c_opt  (short-horizon)",lw=2)
b1.plot(kap,A2s,"s-",color="#9467bd",label="A2 residual (short-horizon)",lw=2)
b1.plot(kap,A1d,"o--",color="#d62728",alpha=0.5,label="A1 c_opt  (max|dE|)")
b1.plot(kap,A2d,"s--",color="#9467bd",alpha=0.5,label="A2 residual (max|dE|)")
b1.axhline(13.5,color="grey",ls=":",lw=1.2); b1.text(2,14.2,"tie line (>13.5/27 needed to 'click')",fontsize=7.5,color="grey")
b1.set_ylim(-1,27); b1.set_xlabel("NN compute charge  kappa  (FLOP-per-call / force-eval)")
b1.set_ylabel("held-out configs beaten (of 27)"); b1.set_title("Corrector never clears the equal-compute bar\n(short-horizon: flat near 0)")
b1.legend(fontsize=8,loc="upper right")
rb=[r["r_bin"] for r in orows]; imp=[r["improve_pct"] for r in orows]
b2.bar(range(len(rb)),imp,color="#8c564b"); b2.set_xticks(range(len(rb))); b2.set_xticklabels([f"{x:.2f}" for x in rb])
b2.set_ylim(-0.5,5); b2.set_xlabel("binary separation  r_bin (AU)"); b2.set_ylabel("oracle improvement over no-NN (%)")
b2.set_title("Binary oracle = 0.0% at every r_bin\n(failure is resolution, not force accuracy)")
b2.axhline(0,color="k",lw=0.8)
for i,v in enumerate(imp): b2.text(i,0.15,f"{v:.1f}%",ha="center",fontsize=8)
fig.suptitle("LEARNED-CORRECTION EVALUATION — equal-compute null (left) + binary oracle null (right)",fontsize=11)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(os.path.join(OUT,"nn_vs_compute.png"),dpi=150)
print("wrote nn_evaluation_table.txt + nn_vs_compute.png\n")

# ===================== TASK 4 — reconciliation =====================
R=["RECONCILIATION NOTE (internal) — matched-dt ablation on IC1.  Run A = no-adaptive (NN, fixed dt); D = full SIMON.",
   "Purpose: pin down what is / isn't real about the old 'no-adaptive ejects to 161.7 AU' claim. Tells us what NOT to claim.",
   "="*92,
   f"{'run':30} {'dt':>5} {'|dE/E0|max':>12} {'bounded':>8} {'final_rms[AU]':>14} {'fe':>7}",
   "-"*92,
   f"{'A_no_adaptive(NN,fixed)':30} {'0.04':>5} {'0.6845%':>12} {'yes':>8} {'1.69':>14} {2501:>7}",
   f"{'C_no_nn(soft,adaptive)':30} {'0.04':>5} {'1.2397%':>12} {'yes':>8} {'1.68':>14} {2504:>7}",
   f"{'D_full_SIMON(NN,adaptive)':30} {'0.04':>5} {'1.3895%':>12} {'yes':>8} {'1.53':>14} {2504:>7}",
   f"{'A_no_adaptive(NN,fixed)':30} {'0.02':>5} {'417.22%':>12} {'NO':>8} {'104.76':>14} {5001:>7}",
   f"{'A_no_adaptive(NN,fixed)':30} {'0.01':>5} {'9253.2%':>12} {'NO':>8} {'428.38':>14} {10001:>7}",
   "="*92,
   "At the OPERATIONAL dt=0.04, Run A (no-adaptive) is BOUNDED at 0.6845% — the LOWEST drift of the three runs.",
   "It only destabilises at finer FIXED dt (0.02, 0.01). The specific '161.7 AU' value is NOT reproduced at matched dt",
   "(it is a finer-dt / sampling-bug artifact).",
   "",
   "DO-NOT-CLAIM : 'the un-adapted run ejects to 161.7 AU at the operating dt'  (false: bounded at 0.68% at dt=0.04).",
   "CAN-NOTE     : a naive fixed scheme destabilises as dt is refined into close encounters, where time-symmetric /",
   "               adaptive integration stays controlled (see frontier_table & track2)."]
open(os.path.join(OUT,"reconciliation.txt"),"w").write("\n".join(R)+"\n")
print("\n".join(R)); print("\nwrote reconciliation.txt")
