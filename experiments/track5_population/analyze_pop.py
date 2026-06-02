"""
Track 5 analysis: operating envelope from the 400-config population sweep.
Methods per config: lf04, lf005, adapt, tsalf (panel: max_dE_pct, bounded, fe, minsep).
Regime by min pairwise separation. Winner = cheapest method that is bounded AND
|dE/E0|max < 1%. Quantify where tsalf wins vs loses.
"""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = ROOT + "/experiments/track5_population"
D = json.load(open(OUT + "/popsweep.json")); R = D["results"]
METHS = ["lf04", "lf005", "adapt", "tsalf"]
L=[]; log=lambda s:(print(s), L.append(s))

def regime(ms):
    if ms is None: return "unknown"
    if ms < 0.05: return "close (<0.05)"
    if ms < 0.3:  return "medium (0.05-0.3)"
    return "wide (>0.3)"

def ok(rec, meth):
    p = rec.get(meth, {}); dE = p.get("max_dE_pct")
    return bool(p.get("bounded")) and (dE is not None) and np.isfinite(dE) and dE < 1.0

log(f"Population: {D['N']} bound configs (T={D['T']}). Target = bounded AND |dE/E0|max<1%.")
# per-method aggregate
log("\nPer-method (over all configs):")
log(f"{'method':<8}{'%bounded':>10}{'%meet-target':>14}{'median fe(target)':>18}")
for m in METHS:
    nb = sum(1 for r in R if r.get(m,{}).get("bounded"))
    nt = sum(1 for r in R if ok(r,m))
    fes = [r[m]["force_evals"] for r in R if ok(r,m)]
    log(f"{m:<8}{100*nb/len(R):>9.1f}%{100*nt/len(R):>13.1f}%{int(np.median(fes)) if fes else 0:>18}")

# winner per config (cheapest meeting target)
regimes = ["close (<0.05)","medium (0.05-0.3)","wide (>0.3)"]
win = {m:{rg:0 for rg in regimes} for m in METHS}; nowin={rg:0 for rg in regimes}; reg_count={rg:0 for rg in regimes}
for r in R:
    rg = regime(r.get("minsep"))
    if rg not in reg_count: continue
    reg_count[rg]+=1
    cand = [(r[m]["force_evals"], m) for m in METHS if ok(r,m)]
    if not cand: nowin[rg]+=1; continue
    win[min(cand)[1]][rg]+=1
log("\nWinner (cheapest method meeting target), by regime:")
log(f"{'regime':<20}{'n':>5}" + "".join(f"{m:>9}" for m in METHS) + f"{'none':>7}")
for rg in regimes:
    log(f"{rg:<20}{reg_count[rg]:>5}" + "".join(f"{win[m][rg]:>9}" for m in METHS) + f"{nowin[rg]:>7}")

# tsalf vs lf04 head-to-head on robustness (bounded)
ts_b = sum(1 for r in R if r.get("tsalf",{}).get("bounded"))
lf_b = sum(1 for r in R if r.get("lf04",{}).get("bounded"))
both_close = [r for r in R if regime(r.get("minsep"))=="close (<0.05)"]
ts_bc = sum(1 for r in both_close if r.get("tsalf",{}).get("bounded"))
lf_bc = sum(1 for r in both_close if r.get("lf04",{}).get("bounded"))
log(f"\nRobustness: tsalf bounded {ts_b}/{len(R)} ({100*ts_b/len(R):.0f}%); lf04 bounded {lf_b}/{len(R)} ({100*lf_b/len(R):.0f}%).")
log(f"On close-encounter configs (n={len(both_close)}): tsalf bounded {ts_bc}/{len(both_close)}, lf04 bounded {lf_bc}/{len(both_close)}.")

# ---- plots ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
xb = np.arange(len(regimes)); w = 0.2
for i, m in enumerate(METHS):
    ax[0].bar(xb + (i-1.5)*w, [win[m][rg] for rg in regimes], w, label=m)
ax[0].bar(xb + 2.5*w*0, [0]*len(regimes), 0)  # spacer noop
ax[0].set_xticks(xb); ax[0].set_xticklabels([r.split()[0] for r in regimes])
ax[0].set_ylabel("# configs won (cheapest @ target)"); ax[0].set_title("Operating envelope: winner by regime")
ax[0].legend(fontsize=8)
# scatter minsep vs fe (tsalf green if bounded&target, lf04 blue), log-log
for r in R:
    ms = r.get("minsep");
    if ms is None: continue
    if ok(r,"tsalf"): ax[1].scatter(ms, r["tsalf"]["force_evals"], s=10, color="#16A34A", alpha=0.5)
    if ok(r,"lf04"):  ax[1].scatter(ms, r["lf04"]["force_evals"],  s=10, color="#2563A6", alpha=0.5)
ax[1].axvline(0.05, color='gray', ls=':'); ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("min pairwise separation (regime)"); ax[1].set_ylabel("force evals @ target")
ax[1].set_title("green=tsalf, blue=lf04 (only bounded&<1%)")
fig.tight_layout(); fig.savefig(OUT+"/operating_envelope.png", dpi=170); plt.close()
log("\nwrote operating_envelope.png")

# verdict
log("\nTRACK 5 VERDICT: operating envelope quantified (see win-by-regime table + plot).")
open(OUT+"/track5_results.txt","w").write("\n".join(L)+"\n")
print("TRACK5_ANALYSIS_DONE")
