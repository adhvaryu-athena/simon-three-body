# Structure over resolution: an operating envelope for stable, low-cost integration of the chaotic three-body problem

**Aarush Gupta**

---

## Abstract

The gravitational three-body problem has no general closed-form solution and is the prototype of deterministic chaos, so every quantitative statement about a three-body system rests on a numerical integrator — and on how that integrator's reliability is judged. I ask what structural property an integrator must preserve to follow a chaotic system stably and cheaply, and how "stability" should be measured when exponential divergence makes long-horizon position error meaningless. Comparing fixed, adaptive, time-symmetric, and fourth-order symplectic integrators — plus two machine-learned force corrections — against a fifteenth-order reference across analytic configurations, binary–single encounters, and the real Sun–Earth–Moon system, I find that (i) time-reversal symmetry, not finer resolution, controls long-term energy drift; (ii) energy conservation can certify a physically destroyed system that only a pair-separation metric detects; and (iii) a learned force correction does not beat equal-cost classical refinement, with the failure localised to temporal resolution rather than force accuracy. The results define an *operating envelope* of which integrator class is necessary in each regime, and a measurement principle for chaotic N-body work.

## 1. Background and motivation

Since Poincaré showed the three-body problem to be non-integrable [1], its study has been irreducibly numerical: trajectories are computed, not solved. Two integrator families dominate. *Symplectic* methods (leapfrog/velocity-Verlet, and higher-order compositions [2]) preserve a modified Hamiltonian and so bound energy error over long times; *high-order adaptive* methods such as IAS15 [3], available in the REBOUND framework [4], achieve near-machine-precision trajectories and serve as references. More recently, machine learning has been proposed to accelerate or replace classical N-body integration, including for the chaotic three-body problem [5].

Two problems motivate this work. First, in a chaotic system neighbouring trajectories separate exponentially, so global position error against a reference saturates within a few Lyapunov times and cannot rank integrators on long horizons — yet it is still widely reported. A principled choice of *stability metric* is therefore needed. Second, claims that learned corrections "improve" an integrator are rarely tested at **equal computational cost** against simply refining the classical method; without that control, an apparent gain may be nothing more than spending more compute.

## 2. Research questions

- **Q1 (structure).** Which structural property — symplecticity, time-reversal symmetry, or adaptivity — is necessary and sufficient for stable, low-cost integration, and does the answer depend on dynamical regime?
- **Q2 (measurement).** How should integrator stability be quantified for a chaotic system, where long-horizon trajectory error is uninformative?
- **Q3 (learning).** Can a learned per-step force correction outperform spending the same compute on a finer symplectic step?

## 3. Methods

**Integrators.** I implement and compare fixed-step leapfrog, a heuristic inverse-distance adaptive leapfrog, a *time-symmetric (reversible) adaptive leapfrog* following Hut, Makino & McMillan [6], and a fourth-order symplectic (Yoshida) integrator [2]. The reference is IAS15 (fifteenth-order Gauss–Radau) via REBOUND [3,4].

**Learned corrections.** Two neural corrections are trained on close-encounter-rich populations: a per-step scalar force-magnitude correction (reference-matched) and a state-to-state residual correction. Each is evaluated under a **pre-registered equal-compute protocol**: the network is charged a cost κ (floating-point operations per call, in units of one force evaluation), and a corrector is deemed to "click" only if it falls below the plain-leapfrog cost–accuracy frontier on **held-out** configurations.

**Test systems.** (a) Analytic three-body initial conditions spanning a range of Lyapunov exponents; (b) binary–single scattering with the inner pair's separation swept from 0.01 to 0.40 AU; (c) the real Sun–Earth–Moon system initialised from JPL Horizons ephemerides [7].

**Metrics.** Maximum relative energy drift |ΔE/E₀|; short-horizon RMS trajectory error within ~1–2 Lyapunov times; time-to-divergence; bounded-vs-ejected status; and — for hierarchical systems — the range of each bound pair's separation. Speed is compared only at **matched accuracy**, by interpolating force-evaluation counts on each method's Pareto cost–accuracy frontier (no extrapolation).

## 4. Results

**Finding 1 — time-symmetry, not resolution, controls drift.** The reversible adaptive leapfrog stays energy-bounded across every regime at a *single* control setting (e.g. 0.09%, 0.22%, and 0.003% on three representative configurations), whereas an eight-times-finer *fixed* step *ejects* the close-encounter system at 128% energy error. A matched-accuracy comparison shows the advantage is genuinely regime-dependent: the time-symmetric scheme reaches accuracies fixed leapfrog cannot attain reliably on close encounters and is ~1.5–3× cheaper on under-resolved systems, but is *slower* on near-regular orbits where a uniform step already suffices — so there is no single "N× faster" claim, only an operating map (**Figure 1** = `frontier.png`; matched-accuracy detail in `speed_frontier.png`).

**Finding 2 — the energy-metric trap.** Integrating the real Sun–Earth–Moon system with a fixed step conserves total energy to 0.008% and reports the system as bound, yet the Earth–Moon separation expands to roughly **770× its true value** — the Moon is effectively unbound — while time-symmetric, heuristic-adaptive, and reference integrators all hold it within 6% of truth. A global energy or boundedness check thus certifies a run whose internal hierarchy has been destroyed; only a pair-separation metric detects the failure (**Figure 2**; `sem_pair_range.png`).

**Finding 3 — learned correction does not beat physics at equal cost.** Under the equal-compute protocol, the per-step force-magnitude correction beats an equal-cost finer symplectic step on the short-horizon trajectory metric in **0 of 27** held-out configurations — even when the network is charged nothing — and the residual correction in at most 2 of 27. Decisively, an **oracle** (best-possible) correction yields **0% improvement** on under-resolved binaries at every separation, proving the failure mode to be temporal under-resolution, not force-model error: no force correction can repair a timestep problem, whereas adaptive resolution can (**Figure 3**; `nn_vs_compute.png`).

**Synthesis — the operating envelope.** Across regimes a clear map emerges: cheap fixed leapfrog suffices for regular, well-separated systems; the time-symmetric adaptive scheme is the economical choice for moderate scattering; and the high-order reference is necessary for tight binaries, where it is both more accurate and faster in wall-clock than the alternatives (**Figure 1**).

## 5. Significance and outlook

This work contributes (1) a practical operating envelope specifying which integrator class is necessary and sufficient in each dynamical regime; (2) a measurement principle for chaotic N-body simulation — stability must be judged on preserved dynamical structure (pair separations), not energy alone, with a concrete real-system demonstration of how energy alone misleads; and (3) a reproducible, honestly-accounted negative result showing that a learned force correction does not outperform equal-cost classical refinement, with the failure mechanism identified. The time-symmetric scheme is an existing method [6]; the contribution is its controlled characterisation, the metric-trap demonstration, and the equal-compute null. Natural extensions are higher-N hierarchical systems, a learned *step-size controller* (as opposed to a force correction), and a fuller matched-accuracy cost characterisation. All results are reproducible from a fixed verification guard that exactly recovers the reported energy-drift values.

## References

[1] H. Poincaré (1890), *Sur le problème des trois corps et les équations de la dynamique*, Acta Mathematica 13, 1–270.
[2] H. Yoshida (1990), *Construction of higher order symplectic integrators*, Physics Letters A 150, 262–268.
[3] H. Rein & D. S. Spiegel (2015), *IAS15: a fast, adaptive, high-order integrator for gravitational dynamics*, MNRAS 446, 1424–1437.
[4] H. Rein & S.-F. Liu (2012), *REBOUND: an open-source multi-purpose N-body code for collisional dynamics*, A&A 537, A128.
[5] P. G. Breen, C. N. Foley, T. Boekholt & S. Portegies Zwart (2020), *Newton versus the machine: solving the chaotic three-body problem using deep neural networks*, MNRAS 494, 2465–2470.
[6] P. Hut, J. Makino & S. McMillan (1995), *Building a better leapfrog*, ApJ Letters 443, L93–L96.
[7] J. D. Giorgini et al. (1996), *JPL's on-line Solar System data service (HORIZONS)*, BAAS 28, 1158.

*(Bibliographic details should be confirmed against the originals before final submission.)*
