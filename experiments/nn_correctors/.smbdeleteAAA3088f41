# nn_correctors — corrections to the A1/A2 (Claim 3) pipeline

Applied June 2026 in response to the technical review *"Identified Issues with nn_correctors
Experiments for Claim 3."* All seven issues from that report are addressed. Each entry gives the
file, the change, and why it matters.

**Bottom line:** the bugs were real and worth fixing, but a corrected-inference diagnostic run on
the existing trained weights shows the **headline verdict is unchanged — A1 and A2 remain NULL**
(neither corrector beats an equal-compute finer leapfrog step). Per-config numbers move; the
conclusion of Finding 3 does not. Canonical artifacts should be regenerated on the rig with
`run_corrected.sh` before the numbers are quoted.

---

## 1. A2 residual sign mismatch — MAJOR — `gen_data.py`
**Was:** training target `resid = leapfrog_exit − IAS15_exit`, but inference does `x += resid`.
A perfect prediction therefore moved the state *away* from IAS15.
**Fix:** target is now `IAS15_exit − leapfrog_exit` (the correction to add). Inference still adds it,
so the saved target literally means "correction to apply." Requires regenerating `datasets.npz` and
retraining A2 (the saved `a2_weights.npz` were trained on the old sign).
*Diagnostic note:* negating the existing A2 prediction is mathematically equivalent to retraining on
the flipped target, which is how the diagnostic previews the corrected result without a retrain.

## 2. A2 residual window-timing mismatch — MEDIUM-MAJOR — `corr_lib.py`
**Was:** the target is generated for exactly `W` (=12) leapfrog steps from the feature state, but
inference captured the feature, set `in_window=1`, and applied at `in_window>=12` — i.e. after only
**11** future steps (off by one).
**Fix:** a `steps_remaining = window` counter is set at encounter entry, decremented after each
future step, and the correction is applied when it reaches 0 — exactly `W` steps after the feature
state, matching training.

## 3. A1 scalar `c` not applied to the starting half-kick — MEDIUM — `corr_lib.py`
**Was:** the predicted `c` only entered `accc(x, c)` after the drift, so it scaled the *end*
acceleration of the step; the *start* half-kick still used the previous step's cached acceleration.
**Fix:** after predicting `c`, the start acceleration is recomputed as `a = af + c*ac`, reusing the
split already computed for gating (no extra force evaluation, so the equal-compute accounting is
unchanged). The same `c` is now used for both half-kicks of the step (documented choice).

## 4. Evaluation `bounded` flag only checked finiteness — MINOR — `evaluate.py`
**Was:** `bounded = all(isfinite(p))` — only caught NaN/Inf, not physical ejection.
**Fix:** `bounded` now also requires every body to stay within a COM-distance threshold (30.0,
matching `corr_lib.trainable`); the old finiteness check is retained separately as `finite`.
*Effect:* a few configs where a corrector flings a body away (finite but unbound) are now correctly
excluded from "wins" — this is why the corrected totals drop from 27 to ~24 (A1) / ~22 (A2).

## 5. `pair_fidelity` docstring vs implementation — MINOR — `corr_lib.py`
**Was:** docstring said min/max separation; code compared only the maximum.
**Fix:** the metric now takes the worse normalized error over **both** the minimum (closest-approach)
and maximum (widest-excursion) pair separation. (Reported in tables only; not used in the verdict.)

## 6. `make_configs` could silently return fewer configs — MINOR — `gen_data.py`
**Fix:** a warning is logged if the trial cap is hit, and explicit assertions
(`len(train_cfgs)==N_TRAIN`, `len(test_cfgs)==N_TEST`) now fail loudly. Actual counts are written to
`gen_meta.json`.

## 7. A2 IAS15 window failures silently swallowed — MINOR — `gen_data.py`
**Fix:** `except Exception` now counts skipped windows (`skip_ias15`), logs the first few with the
exception type, and records the total in `gen_meta.json`.

---

## Diagnostic result (corrected inference, existing weights)

Short-horizon RMS is the decisive chaos-era metric. Wins = configs where the corrector beats the
equal-compute plain-leapfrog frontier by >10%. κ = NN cost per force-eval.

| Corrector | Metric | κ=0 | κ=10 | κ=90 | verdict |
|---|---|---|---|---|---|
| A1 | short-horizon RMS | 4/24 | 1/24 | 0/24 | **NULL** |
| A1 | energy max\|dE\| | 5/24 | 5/24 | 1/24 | NULL |
| A2 | short-horizon RMS | 4/22 | 4/22 | 2/22 | **NULL** |
| A2 | energy max\|dE\| | 6/22 | 5/22 | 3/22 | NULL |

(Pre-fix run for comparison: A1 short 0/27 at all κ; A2 short 2/27 → 1/27.) The verdict — both
correctors NULL, beaten by an equal-compute finer symplectic step — is robust to the fixes. Several
TEST configs show the correctors actively ejecting the system (A2 unbounded on TEST04/08/18/21/22),
now correctly flagged by the fixed `bounded` check.

## To regenerate canonical artifacts (on the rig)
```
cd experiments/nn_correctors && bash run_corrected.sh
```
This runs `gen_data.py` (rebuild datasets with corrected sign + meta), `train.py` (retrain A1/A2),
and `evaluate.py` (regenerate `results.json`, `results_table.txt`, and the three PNGs) in the pinned
`simon` env. Update the Finding-3 table in `guide/DATA_GUIDE.md` and the outline from the new
`results_table.txt` afterward.
