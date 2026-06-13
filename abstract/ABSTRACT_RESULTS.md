# SIMON — abstract-ready result package

**For:** the 2-page research outline. **Status:** consolidated + verified from existing runs (no new sweeps, except the matched-accuracy speed table in §1.6, computed on request). **Guard:** PASS (IC1 1.3895% / IC3 0.6718% / IC4 13.7722% / IC6 0.0006% at dt=0.04, T=100 yr — exact reproduction).

> Scope note: framing, subject category (Physics vs CS), and venue are **not decided here** — flagged for Ninaad + Sushant. Numbers below are reproduced from the existing result files and confirmed against the prior anchors. One prior anchor — the "~1.05–2.1× speed" figure — was **checked directly and corrected**: the real matched-accuracy result is two-sided and regime-dependent (see §1.6 / ledger).

---

## 1. Locked headline numbers

Each claim is one line + its number + the source file under `abstract/` (or `experiments/`).

1. **Time-symmetry beats brute-force resolution.** A time-symmetric adaptive leapfrog (tsalf, η=0.05) stays energy-bounded across IC1 `0.087%`, IC4 `0.224%`, IC6 `0.003%` at *one* setting — while an 8× finer *fixed* step (dt=0.005) **ejects IC1 at 128.4%** (`bounded=False`). → `frontier_table.txt`, `experiments/track2_symplectic/track2_results.txt`
2. **The metric trap (headline).** Real Sun–Earth–Moon, leapfrog dt=0.01 reports `max|dE/E0| = 0.0079%`, `bounded=True` (global metric: "fine"), yet the Earth–Moon separation runs to **1.99 AU ≈ 774× the true 0.00257 AU** — the Moon is unbound. ias15 / tsalf / heuristic-adaptive hold it at `[0.0024, 0.0027] AU`. → `metric_trap_table.txt`, `sem_pair_range.png`
3. **Learned per-step correction is null (pre-registered, equal-compute).** On 27 held-out configs the scalar `c_opt` corrector (A1) beats an equal-compute finer symplectic step on the short-horizon (trajectory) metric in **0/27** cases — *even when the NN is charged nothing*. The state-residual corrector (A2) manages **2/27 → 1/27** once compute is charged. Both **NULL**. → `nn_evaluation_table.txt`, `nn_vs_compute.png`
4. **Oracle confirms it's a resolution problem, not a force problem.** A *perfect* (oracle upper-bound) encounter correction on binary-single gives **0.0% improvement at every r_bin ∈ [0.01, 0.40]** — the binary failure is temporal under-resolution, unfixable by any force/encounter correction; adaptive resolution (tsalf/ias15) fixes it. → `nn_evaluation_table.txt`, `experiments/weekend_phaseC/`
5. **Three-regime operating envelope.** Cheap fixed leapfrog suffices for regular/wide systems (IC6: `0.0006%` at `25 fe/yr`); tsalf is the cheapest *bounded* integrator for moderate scattering (IC1/IC4 at `75–88 fe/yr` vs ias15's `800`); ias15 dominates tight binaries (BS_0.05: bounded **and** ~65× faster wall-clock than tsalf). → `frontier_table.txt`, `frontier.png`
6. **Speed is regime-dependent and two-sided — NO universal multiplier (matched-accuracy, computed).** Force-evals to reach *equal accuracy*, tsalf vs fixed leapfrog: **IC4** (under-resolved) tsalf **1.5–2.9× cheaper** (energy); **IC1** (close encounters) leapfrog is *erratic* — it ejects across a band of intermediate dt and only reaches sub-1% at dt=0.00125 (~80k fe), where tsalf is **~3–4× cheaper and reliable**; **IC3 / IC6** (near-regular) fixed leapfrog is already cheap and tsalf is **0.3–0.9× — i.e. slower**. Full span **0.32×–3.57×**. Robust claim: *feasibility on close encounters + modest savings on under-resolved cases*, never a blanket "N× faster". → `speed_table.txt`, `speed_frontier.png`
7. **Reconciliation (internal, do-not-claim).** At the operational dt=0.04 the un-adapted run is bounded at `0.6845%` — the *lowest* drift of the three runs; the old "161.7 AU ejection" is **not reproduced at matched dt** (finer-dt / sampling artifact). → `reconciliation.txt`

---

## 2. Figures

- `abstract/frontier.png` — operating-envelope frontier: cost-vs-energy-error scatter + a green/red survival grid (5 methods × 9 configs).
- `abstract/sem_pair_range.png` — the metric trap: Earth–Moon separation vs time (leapfrog unbinds the Moon) beside the global-energy bars (all "fine").
- `abstract/nn_vs_compute.png` — equal-compute null (wins-of-27 vs NN compute charge κ) + binary-oracle null (0% at every r_bin).
- `abstract/speed_frontier.png` — matched-accuracy cost frontiers (force-evals vs max|dE|, 4 toy ICs): tsalf's curve sits *left* of leapfrog's (cheaper at equal accuracy) on IC1/IC4, *right* (slower) on IC3/IC6.

---

## 3. Draft abstract paragraph (~290 words)

> The gravitational three-body problem is the canonical example of deterministic chaos in celestial mechanics: lacking a general closed-form solution, its study is inherently numerical, and the reliability of any conclusion rests on the integrator. This work asks a physical question — *what structural properties must a numerical integrator preserve to follow a chaotic three-body system stably and cheaply, and how should "stability" even be measured when exponential divergence renders long-horizon trajectory error meaningless?* **(Background.)** Standard practice judges integrators by energy conservation and by global position error against a high-order reference — here REBOUND's `ias15`, a 15th-order Gauss–Radau scheme. **(Contribution.)** We carry out a controlled comparison of fixed-step, heuristically-adaptive, and time-symmetric adaptive leapfrog integrators — together with two learned per-step force corrections — across toy three-body initial conditions, binary–single scattering encounters, and the real Sun–Earth–Moon system initialised from JPL Horizons. Three results emerge. (1) A time-symmetric adaptive leapfrog (the reversible scheme of Hut, Makino & McMillan 1995, applied and characterised here) stays energy-bounded across regimes at a single control setting, where an eight-times-finer *fixed* step ejects the same system — symmetry, not brute-force resolution, controls secular drift. (2) Global energy conservation can actively mislead: a fixed-step Sun–Earth–Moon run conserves energy to 0.008% yet expands the Moon's orbit to ~770× its true radius; only a pair-separation metric detects the destroyed hierarchy. (3) A learned force-magnitude correction, under a pre-registered equal-compute protocol on held-out configurations, never outperforms spending the same compute on a finer symplectic step, and a perfect (oracle) correction gives zero improvement on under-resolved binaries — localising the failure to temporal resolution, not force accuracy. **(Significance.)** Together these map an operating envelope of which integrator class is necessary in each regime, and a measurement caution for chaotic N-body work: stability must be judged on dynamical structure, not energy alone.

*(Background vs original contribution is explicitly separated above. tsalf is stated as a known method (HMM 1995); the contribution is the controlled characterisation, the metric-trap demonstration, and the equal-compute null — not the integrator itself.)*

---

## 4. Honest claims ledger

**We CAN claim (directly supported):**
- tsalf stays bounded across IC1/IC4/IC6 at one η where fixed dt=0.005 ejects IC1 (128%).
- tsalf reaches *bounded* accuracy at ~10× fewer force-evals than ias15 on moderate-scattering ICs.
- **Matched-accuracy vs fixed leapfrog (computed):** tsalf is **1.5–3.6× cheaper** on close-encounter (IC1) and under-resolved (IC4) systems and reaches accuracies leapfrog cannot hit reliably (IC1 ejection band); on near-regular IC3/IC6 fixed leapfrog is cheaper (tsalf 0.3–0.9×). Span 0.32×–3.57×, regime-dependent. [`speed_table.txt`]
- The metric trap, quantified: energy 0.008% + bounded, yet Moon unbound to 774× true separation.
- Both learned corrections are NULL under equal-compute on held-out configs (A1 0/27 short-horizon; A2 2/27→1/27).
- The binary oracle gives 0% improvement at all r_bin ⇒ failure is temporal resolution, not force accuracy.
- A coherent three-regime operating envelope (cheap leapfrog / tsalf / ias15).

**We CANNOT claim:**
- **A single universal "N× faster" number for tsalf.** The matched-accuracy ratio is two-sided (0.32×–3.57×) and regime-dependent — tsalf is *slower* on near-regular IC3/IC6. The prior "~1.05–2.1×" anchor is **not supported** (too narrow and one-sided; reality has both >2× wins and <1× losses). Always state the regime breakdown, never one multiplier.
- **Novelty of tsalf** — it is the HMM-1995 reversible scheme; our contribution is the characterisation/operating-envelope, not the method.
- **That learned corrections can never help** — only that *these two*, under equal compute on *this* family, are null. (An encounter-NN prototype shows an apparent *global* "win", but on the chaos-saturated endpoint metric and against non-symplectic sub-stepping; it does not change the verdict and is kept out of the core.)
- **Anything from the "161.7 AU ejection"** — not reproduced at matched dt.
- **Subject category / framing / venue** — Ninaad + Sushant's call, not decided here.

---

## 5. For the paper later (fuller tables/figures + what each supports)

| Artifact | Supports |
|---|---|
| `frontier_table.txt` (5 methods × 9 configs, full metric panel) | the operating-envelope figure + the three-regime claim |
| `speed_table.txt` + `speed_frontier.png` (matched-accuracy, computed) | the regime-dependent speed result (claim §1.6); supersedes the unsupported single-multiplier anchor |
| `experiments/track2_symplectic/track2_results.txt` (tsalf/yoshida/leapfrog η & ds sweeps, IC1/IC4/IC6) | the time-symmetry claim |
| `experiments/nn_correctors/results_table.txt` (full per-config, 27 held-out, κ sensitivity) | the equal-compute null with sensitivity analysis |
| `experiments/weekend_phaseC/` (oracle + diagnostic: orbits/macrostep, sub-steps/orbit) | the "resolution not force" diagnosis |
| `reconciliation.txt` | what NOT to claim about the old ablation |

*Verification guard (`experiments/track1_reconciliation/reconcile.py`) must PASS before any of the above is trusted; it does. The matched-accuracy speed table uses Pareto cost-accuracy frontiers with interpolation strictly within measured range (no extrapolation); the IC1 leapfrog frontier is flagged non-monotone (ejection band) in `speed_table.txt`.*
