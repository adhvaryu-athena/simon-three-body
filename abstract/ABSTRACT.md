# Research outline — abstract

**Title:** *Structure over resolution: an operating envelope for stable, low-cost integration of the chaotic three-body problem*

## Abstract

The Newtonian three-body problem is the oldest open problem in dynamics and the prototype of deterministic chaos: with no general closed-form solution, every quantitative statement about a three-body system rests on a numerical integrator — and on how that integrator's reliability is judged. This project asks two coupled questions: *what structural properties must an integrator preserve to follow a chaotic gravitational system stably and at low computational cost, and how should "stability" even be measured when exponential trajectory divergence makes long-horizon position error meaningless?*

I conduct a controlled comparison of four integrator classes — fixed-step leapfrog, heuristically adaptive leapfrog, a time-symmetric adaptive leapfrog (the reversible scheme of Hut, Makino & McMillan, 1995), and a fourth-order symplectic integrator — together with two machine-learned per-step force corrections, all benchmarked against a fifteenth-order Gauss–Radau reference (REBOUND's IAS15). The methods are tested across analytic three-body configurations, binary–single scattering encounters, and the real Sun–Earth–Moon system initialised from JPL Horizons ephemerides, using chaos-appropriate diagnostics: maximum relative energy drift, short-horizon trajectory error within a Lyapunov time, and — critically — the preserved separation of bound pairs.

Three findings result. First, *time-symmetry, not brute-force resolution, controls long-term energy drift*: the reversible adaptive scheme stays energy-bounded across every regime at a single control setting, whereas an eight-times-finer fixed step ejects the same close-encounter system. Second, *energy conservation can be actively misleading*: a fixed-step Sun–Earth–Moon integration conserves total energy to 0.008% yet expands the Moon's orbit to roughly 770 times its true radius — a destroyed hierarchy that only a pair-separation metric detects. Third, *a learned force correction does not beat physics at equal cost*: on held-out configurations a trained per-step force-magnitude correction never outperforms spending the same computation on a finer symplectic step, and an oracle (best-possible) correction yields zero improvement on under-resolved binaries — localising the failure to temporal resolution, not force accuracy.

Together these results map an *operating envelope* — which integrator class is necessary and sufficient in each dynamical regime — and establish a measurement principle for chaotic N-body work: stability must be judged on preserved dynamical structure, not energy alone.

---

*Numbers traceable to: `frontier_table.txt`, `metric_trap_table.txt`, `nn_evaluation_table.txt`, `speed_table.txt`. Verification guard reproduces IC1/IC3/IC4/IC6 energy drift exactly.*
