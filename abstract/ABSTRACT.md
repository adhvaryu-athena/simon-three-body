# Research outline: abstract

**Title:** *Structure over Resolution: an Operating Envelope for Stable Integration of the Chaotic Three-Body Problem*

## Abstract

The Newtonian three-body problem has no general closed-form solution and is a prototype of
deterministic chaos, so every quantitative statement about such a system is obtained numerically,
which makes both the integration scheme and the yardstick by which it is judged decisive. Two
difficulties are usually left implicit. Because neighbouring trajectories diverge exponentially, the
customary measure, global position error against a reference, saturates within a few Lyapunov
times and cannot rank methods over long time scales. And neural network models proposed to
accelerate integration are seldom tested at equal computational cost against the obvious alternative
of taking a finer step.

I compare five integrator methods and two neural network correctors across three test systems
(analytic three-body configurations of varying chaoticity, binary–single scattering encounters, and
the real Sun–Earth–Moon system initialised from JPL Horizons ephemerides), using chaos-appropriate
diagnostics: maximum relative energy drift, short-horizon position RMS within one Lyapunov time,
bounded/ejected status, and, critically for hierarchical systems, the preserved separation of
bound pairs.

Three findings result. First, *a time-symmetric, reversible adaptive integrator gives regime-independent stability from a single control setting*: the reversible adaptive leapfrog (η = 0.05) stays energy-bounded across every tested regime without per-regime tuning, and reaches the accuracy of a much finer fixed step at lower cost on close-encounter and under-resolved systems, with no advantage on near-regular orbits. The contribution is an operating map of where adaptivity pays, not a universal speedup.

Second, *energy conservation can actively mislead*: a fixed-step Sun–Earth–Moon integration
conserves total energy to max|ΔE/E₀| = 0.008% and reports the system as bound, yet the
Earth–Moon separation grows to 774× its true value, a destroyed hierarchy that only a
pair-separation metric detects.

Third, *neural network force and state corrections do not beat physics at equal cost*: under a
pre-registered equal-compute protocol on 27 held-out configurations, a per-step scalar corrector
(A1) beats an equal-cost finer leapfrog step on short-horizon position RMS in just 1 of 25 bounded
cases, even when charged no computational cost, and an encounter-triggered state corrector (A2)
in at most 2 of 24. An oracle (best-possible) correction yields 0% improvement on under-resolved
binaries at every separation tested, localising the failure to temporal resolution, not force
accuracy.

Together these results map an *operating envelope*, identifying which integrator is necessary and
sufficient in each dynamical regime, and establish a selection of stability diagnostics for chaotic
three-body work: stability must be judged on preserved dynamical structure, not energy alone.

---

*Numbers traceable to: `frontier_table.txt`, `metric_trap_table.txt`, `nn_evaluation_table.txt`,
`speed_table.txt`. Verification guard reproduces IC1/IC3/IC4/IC6 energy drift exactly.
Detailed claim-to-source mapping: `guide/DATA_GUIDE.md`.*
