# SIMON — physically-constrained integration of the three-body problem

SIMON is a research codebase for hybrid and physically-constrained numerical integration of the
unrestricted gravitational three-body problem. The base integrator is a second-order leapfrog
(velocity-Verlet) with analytic Newtonian force directions, a learned scalar force-magnitude
correction, and adaptive sub-stepping for close encounters. The codebase additionally develops
and evaluates a **time-symmetric adaptive integrator** and a **fourth-order symplectic (Yoshida)
integrator**, and benchmarks every method against REBOUND's `ias15` (15th-order Gauss–Radau) as
the reference. Evaluation emphasises chaos-appropriate metrics — energy conservation
`|dE/E0|max`, short-horizon accuracy within a Lyapunov time, time-to-divergence, and
pair-separation fidelity — rather than long-horizon global position error, which in a chaotic
system is dominated by exponential trajectory divergence.

## Repository structure

- **`src/`** — core integrators and shared utilities.
  - `simon_core.py` — leapfrog backbone (general-N), force model, REBOUND `ias15` reference, and the metric panel.
  - `integrators_symplectic.py` — time-symmetric adaptive leapfrog and the Yoshida fourth-order integrator (with a Kepler / reversibility self-test).
- **`experiments/`** — the experiment suite; each subfolder is self-contained (scripts + result tables + plots):
  - `track1_reconciliation/` — matched-timestep ablation of the adaptive scheme.
  - `track2_symplectic/` — the time-symmetric / symplectic integrators against fixed-step and heuristic baselines.
  - `track3_nn_stepper/` — a learned timestep controller compared to an analytic dynamical-time criterion.
  - `track5_population/` — a population sweep mapping each integrator's operating regime.
  - `weekend_phaseC/` — binary-single (tight-pair) operating envelope.
  - `weekend_phaseD/` — combined cost-vs-accuracy operating envelope across configurations.
  - `nn_correctors/` — physically-independent learned correctors compared to equal-compute finer-step baselines.
- **`_orig/`** — the original SIMON implementation and its evaluation outputs (multi-initial-condition evaluation, before/after adaptive sub-stepping, real Sun–Earth–Moon validation, trained weights).
- **`sushant_experiments/`** — earlier exploratory experiments contributed by a collaborator (see its `NOTE.md`), kept separate from the main codebase.

## Setup

Requires **Python 3.11** with `numpy`, `scipy`, `torch`, `rebound`, `matplotlib`, `pandas`,
`pyarrow`, and `astroquery`/`astropy` (the last two only for fetching real Sun–Earth–Moon
initial conditions from JPL Horizons). `rebound` provides the `ias15` reference integrator;
`torch` is used for the small learned components and runs on CPU or GPU.

```bash
conda create -n simon python=3.11 -y
conda activate simon
pip install -r requirements.txt
# or: pip install numpy scipy torch rebound matplotlib pandas pyarrow astroquery astropy
```

## Reproduce

All commands run from the repository root with the environment active.

**Verification baseline** — energy drift `|dE/E0|max` at `dt = 0.04 yr`, `T = 100 yr`:
```bash
python experiments/track1_reconciliation/reconcile.py
```
Expected: `IC1 ≈ 1.39%`, `IC3 ≈ 0.67%`, `IC4 ≈ 13.77%`, `IC6 ≈ 0.0006%` (guard prints `PASS`).

**Symplectic integrator self-test** — Kepler `e = 0.9` energy conservation and reversibility round-trip:
```bash
python src/integrators_symplectic.py
```

**Time-symmetric integrator vs baselines** (IC1 / IC4 / IC6):
```bash
python experiments/track2_symplectic/run_track2.py     # -> track2_results.txt, frontier.png
```

**Operating-envelope figure** — cost vs energy drift across configurations (the Sun–Earth–Moon case fetches real Horizons initial conditions, so it needs network + `astroquery`):
```bash
python experiments/weekend_phaseD/phaseD_data.py       # -> phaseD_data.json
python experiments/weekend_phaseD/phaseD_plot.py       # -> operating_envelope.png, cost_vs_rbin.png, sem_pair_range.png, operating_envelope_table.txt
```

**Learned correctors vs equal-compute physics:**
```bash
python experiments/nn_correctors/gen_data.py           # population + training datasets
python experiments/nn_correctors/train.py              # train both correctors (GPU optional)
python experiments/nn_correctors/evaluate.py           # -> results_table.txt, energy_vs_compute.png
```

## Results crosswalk (finding → script → output)

| Finding | Script | Output |
|---|---|---|
| Baseline energy-drift values reproduce | `experiments/track1_reconciliation/reconcile.py` | `reconciliation.txt` |
| At matched timestep, the no-adaptive run is bounded at `dt = 0.04`; it ejects only at finer fixed `dt` | same | `reconciliation.txt` |
| A time-symmetric adaptive leapfrog stays energy-bounded across IC1/IC4/IC6 at a single setting and competitive cost | `experiments/track2_symplectic/run_track2.py` | `track2_results.txt`, `frontier.png` |
| A fixed-step fourth-order symplectic integrator is energy-bounded for near-circular orbits but not for high-eccentricity close encounters | `src/integrators_symplectic.py` | (stdout) |
| Cost-vs-accuracy operating envelope; a pair-separation metric reveals close-pair unbinding that global energy and centre-of-mass boundedness miss | `experiments/weekend_phaseD/` | `operating_envelope_table.txt`, `operating_envelope.png`, `sem_pair_range.png` |
| A learned timestep controller matches an analytic dynamical-time criterion at equal accuracy, with no net gain on held-out configurations | `experiments/track3_nn_stepper/eval_track3.py` | `track3_results.txt` |
| A learned scalar force-magnitude correction and a learned state-residual correction do not outperform an equal-compute finer symplectic step on held-out configurations | `experiments/nn_correctors/evaluate.py` | `results_table.txt`, `energy_vs_compute.png` |

## License & author

Released under the MIT License (see `LICENSE`).

Author: **Aarush Gupta** — research code for a study of physically-constrained integration of the three-body problem.
