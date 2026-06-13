# Structure over Resolution: an Operating Envelope for Stable Integration of the Chaotic Three-Body Problem

**Aarush Gupta**

---

## Abstract

The Newtonian three-body problem has no general closed-form solution and is a prototype of
deterministic chaos, so every quantitative statement about such a system is obtained numerically —
which makes both the integration scheme and the yardstick by which it is judged decisive. Two
difficulties are usually left implicit. Because neighbouring trajectories diverge exponentially, the
customary measure — global position error against a reference — saturates within a few Lyapunov
times and cannot rank methods over long time scales. And neural network models proposed to
accelerate integration are seldom tested at equal computational cost against the obvious alternative
of taking a finer step, so an apparent gain may reflect only extra computation.

I compare five integrator methods and two neural network correctors across three test systems —
analytic three-body configurations of varying chaoticity, binary–single scattering encounters, and
the real Sun–Earth–Moon system initialised from JPL Horizons — and find three results.
(i) Time-reversal symmetry, not brute-force resolution, controls long-term energy drift: the
reversible adaptive leapfrog (tuning parameter η = 0.05) stays energy-bounded across all dynamical
regimes, whereas a fixed step using eight times as many force evaluations ejects the same
close-encounter system at 128% energy error.
(ii) Energy conservation can certify a physically destroyed system: a fixed-step integration
conserves total energy to max|ΔE/E₀| = 0.008% and reports the system as bound, yet the
Earth–Moon separation grows to 774× its true value — a failure visible only in a pair-separation
diagnostic.
(iii) A neural network force or state correction does not beat equal-cost classical refinement:
on 27 held-out configurations the scalar corrector wins 0 of 27 on short-horizon position RMS
even when charged no computational cost, and a perfect oracle correction yields 0% improvement
on under-resolved binaries — localising the failure to temporal resolution, not force-model error.
Together, these results define an *operating envelope* mapping which integrator is necessary and
sufficient in each dynamical regime, and establish which stability diagnostics are required for
chaotic three-body work.

---

## 1. Background and motivation

Since Poincaré showed the three-body problem to be non-integrable [1], its study has been
irreducibly numerical: trajectories are computed, not solved. Two integrator families dominate.
*Symplectic* methods (leapfrog/velocity-Verlet, and higher-order compositions such as the
Yoshida fourth-order integrator [2]) preserve a modified Hamiltonian and so bound energy error
over long times. *High-order adaptive* methods, such as IAS15 — a fifteenth-order Gauss–Radau
scheme [3] available in the REBOUND framework [4] — achieve near-machine-precision
trajectories and serve as references. More recently, neural network models have been proposed to
accelerate or correct classical integration for the chaotic three-body problem [5].

IAS15 is the gold standard but carries a high force-evaluation cost (~800–30,000 evaluations
per year depending on the system). For applications requiring long integration times, large
populations of initial conditions, or real-time computation — such as planet formation surveys,
gravitational scattering statistics, or spacecraft trajectory planning — a cheaper integrator with
known reliability guarantees is needed. Two gaps motivate this work. First, in a chaotic system
neighbouring trajectories separate exponentially, so position error against a reference saturates
within a few Lyapunov times and cannot rank integrators on long time scales — yet it is still
widely reported as the primary diagnostic. A principled selection of *stability diagnostics* is
therefore needed. Second, claims that neural network corrections "improve" an integrator are rarely
tested at **equal computational cost** against simply refining the classical step; without that
control, an apparent gain may be nothing more than extra computation.

---

## 2. Research questions

- **Q1 (structure).** Which structural property — symplecticity, time-reversal symmetry, or
  adaptive step-sizing — governs stable integration, and does the answer depend on the dynamical
  *regime* (the chaoticity and hierarchical structure of the initial conditions: regular, moderate
  scattering, or tight binary)?
- **Q2 (measurement).** Which diagnostics are sufficient to classify integrator *stability* —
  that is, to determine whether a numerical solution preserves the physical structure of the
  three-body system — when long-time position error is uninformative due to chaos?
- **Q3 (learning).** Can a neural network force or state correction outperform spending the same
  compute on a finer classical step?

---

## 3. Methods

**Integrator methods.** Five methods are implemented and compared against the IAS15 reference:

1. *Fixed-step leapfrog* (dt ∈ {0.04, 0.005} yr; 25 and 200 force evaluations/yr).
2. *Heuristic inverse-distance adaptive leapfrog*: step size scales as the cube of the minimum
   inter-body separation; this is the SIMON integrator without its neural network component.
3. *Time-symmetric (reversible) adaptive leapfrog* [Hut, Makino & McMillan, 1995 [6]]: step size
   is found by fixed-point iteration to satisfy a time-reversibility condition, ensuring no secular
   drift in the modified Hamiltonian. Tuning parameter η = 0.05 is held fixed across all regimes.
4. *Fourth-order symplectic integrator* [Yoshida, 1990 [2]]: a composition of leapfrog steps that
   achieves fourth-order accuracy while preserving the symplectic structure.
5. *IAS15* [3,4]: fifteenth-order Gauss–Radau adaptive scheme; used as the high-accuracy reference
   throughout (not compared as a "choice" in the operating envelope but as the reliability standard).

**Neural network correctors.** Two correctors are trained on close-encounter data and applied to
the leapfrog baseline:

- *A1 — per-step scalar force-magnitude correction*: a small network that outputs a scalar
  multiplier on the gravitational force at every leapfrog step, trained to match IAS15 forces.
- *A2 — encounter-triggered state-to-state residual correction*: a network activated only at
  close approach that predicts the residual correction to the integrated state after one step;
  this is not a per-step corrector and fires on average once per 100-yr rollout on the test ICs.

Each corrector is evaluated under a **pre-registered equal-compute protocol**: the network is
charged a cost κ (in units of one force evaluation) and is deemed to "click" only if it falls
below the plain-leapfrog cost–accuracy frontier on **held-out** configurations not seen during
training.

**Test systems** (all integrated over T = 100 yr):

- *Analytic three-body ICs* spanning a range of Lyapunov exponents: near-regular (IC6,
  λ ≈ 0.019 yr⁻¹), moderate scattering (IC1, λ ≈ 0.17 yr⁻¹; IC4, λ ≈ 0.10 yr⁻¹).
- *Binary–single scattering*: inner binary separation r_bin ∈ {0.05, 0.10, 0.20, 0.40} AU,
  perturber at 3 AU.
- *Real Sun–Earth–Moon system* initialised from JPL Horizons ephemerides [7].

**Stability diagnostics** (chaos-appropriate; all reported together):

- Maximum relative energy drift, max|ΔE/E₀|, over T = 100 yr.
- Short-horizon position RMS within ~1 Lyapunov time (the informative chaos-era metric).
- Time-to-divergence (time at which position error exceeds a fixed threshold).
- Bounded vs ejected status (whether any body reaches escape velocity).
- Range of bound-pair separation [AU] — required for hierarchical systems such as binary–single
  and Sun–Earth–Moon; this is the diagnostic that detects sub-system unbinding.

Method *efficiency* (force evaluations to reach equal accuracy) is compared only at matched
accuracy, by interpolating each method's Pareto cost–accuracy frontier within the measured range.
Wall-clock speed is noted only as a secondary observation.

---

## 4. Results

**Finding 1 — time-reversal symmetry governs long-term energy drift (answers Q1).**
The reversible adaptive leapfrog (η = 0.05) stays energy-bounded across all test systems at a
single setting: max|ΔE/E₀| = 0.087%, 0.224%, and 0.003% on IC1, IC4, and IC6 respectively.
By contrast, a fixed step with dt = 0.005 (eight times more force evaluations per year) ejects
the IC1 close-encounter system at 128% energy error. The heuristic adaptive leapfrog also fails
on close encounters (IC1: 1.24%; BS_0.05: 4905%, ejected); only time-symmetric adaptation and
IAS15 remain bounded across all regimes.

The efficiency advantage is regime-dependent. On the energy metric, the time-symmetric scheme
reaches accuracies that fixed leapfrog cannot attain reliably (IC1 ejection band at intermediate
dt makes its frontier non-monotone) and is 1.5–3.6× cheaper in force evaluations on
close-encounter (IC1) and under-resolved (IC4) configurations. On near-regular IC3/IC6 the
fixed step is already cheap and the time-symmetric scheme uses 2–3× more evaluations for no
accuracy gain. The contribution is therefore an operating map — not a universal speedup
(**Figure 1**: cost vs energy drift across all 9 configurations).

**Finding 2 — energy conservation can actively mislead (answers Q2, first part).**
The real Sun–Earth–Moon system integrated with leapfrog at dt = 0.01 yr reports max|ΔE/E₀| =
0.008% and bounded = True. Yet over 100 yr the Earth–Moon separation grows from its true value
of 0.00257 AU to a peak of 1.99 AU — 774× the true distance — while IAS15, the time-symmetric
scheme, and the heuristic-adaptive integrator all hold it within [0.0024, 0.0027] AU. A global
energy or ejection criterion thus certifies a run whose internal hierarchical structure is
destroyed. This motivates the selection of stability diagnostics in the conclusions (**Figure 2**).

**Finding 3 — neural network corrections do not beat equal-cost classical refinement (answers Q3).**
Under the pre-registered equal-compute protocol on 27 held-out configurations:

- *A1* (per-step scalar): 0 of 27 configurations beat an equal-cost finer leapfrog step on
  short-horizon position RMS — even when the network is charged zero cost (κ = 0).
- *A2* (encounter-triggered state residual): 2 of 27 at κ = 0, dropping to 1 of 27 once compute
  is charged. Both correctors are **null**.

Decisively, a perfect *oracle* correction applied at every close encounter yields **0% improvement**
on the binary–single system at every separation r_bin ∈ {0.05–0.40 AU}, confirming that the
failure is temporal under-resolution — an inadequate time step — rather than force-model error.
No force or state correction can repair a step-size problem; adaptive resolution (time-symmetric
leapfrog or IAS15) can and does (**Figure 3**). The prototype A2 demonstrates that encounter
dynamics contain learnable structure, but the learned model does not translate into a net gain at
equal computational cost.

**Operating envelope.**
Three dynamical regimes and their sufficient integrator emerge from the nine-configuration
comparison:

| Regime | Sufficient integrator | Example | Cost (fe/yr) |
|---|---|---|---|
| Regular / well-separated | Fixed leapfrog dt = 0.04 | IC6: 0.0006% energy | 25 |
| Moderate scattering | Time-symmetric adaptive (η = 0.05) | IC1: 0.087%, IC4: 0.224% | 75–88 |
| Tight binary (r_bin ≤ 0.05 AU) | IAS15 | BS_0.05: bounded; tsalf 65× slower | ~31,600 |

---

## 5. Conclusions and contributions

**C1 — Operating envelope (answers Q1).** Fixed-step leapfrog is necessary and sufficient for
regular, well-separated systems. The time-symmetric adaptive leapfrog is necessary and sufficient
for moderate scattering: it is the only method that stays bounded at a single control setting
across this regime and reaches accuracies that fixed leapfrog cannot attain. IAS15 is necessary
for tight binaries, where all fixed-step and heuristic methods eject the binary and the
time-symmetric scheme, while bounded, is 65× slower in wall-clock time.

**C2 — Stability diagnostics (answers Q2).** For chaotic three-body work, stability cannot be
judged from energy drift or ejection status alone. The full diagnostic set required is:
(a) max|ΔE/E₀| over the full integration time; (b) short-horizon position RMS within one
Lyapunov time; (c) bounded/ejected status; (d) bound-pair separation range for hierarchical
systems. Finding 2 provides a concrete real-system demonstration: energy and boundedness both
pass while the Moon's orbit is destroyed; only (d) detects the failure.

**C3 — Neural network force/state corrections do not beat equal-cost refinement (answers Q3).**
Under a pre-registered protocol, neither the per-step scalar correction (A1: 0/27) nor the
encounter-triggered state correction (A2: ≤2/27) outperforms an equal-cost finer classical step
on the short-horizon metric. The oracle analysis localises the failure to temporal resolution.
The A2 prototype shows that encounter dynamics have learnable structure, but learning alone
is not sufficient to overcome the timestep problem; adaptive resolution is.

**Novelty.** The time-symmetric adaptive leapfrog is the method of Hut, Makino & McMillan
(1995) [6]. The original contributions of this work are: the controlled characterisation of this
method across dynamical regimes on a common suite of test systems; the demonstration that an
energy criterion misclassifies a physically destroyed real system and the identification of
the pair-separation diagnostic as necessary; and the pre-registered equal-compute null result on
learned corrections with the failure mechanism identified.

---

## Keywords

three-body problem; symplectic and time-symmetric integration; deterministic chaos;
computational celestial mechanics; neural network force corrections; operating envelope

---

## References

[1] H. Poincaré (1890). Sur le problème des trois corps et les équations de la dynamique.
    *Acta Mathematica*, 13, 1–270.

[2] H. Yoshida (1990). Construction of higher order symplectic integrators.
    *Physics Letters A*, 150, 262–268.

[3] H. Rein & D. S. Spiegel (2015). IAS15: a fast, adaptive, high-order integrator for
    gravitational dynamics. *MNRAS*, 446, 1424–1437.

[4] H. Rein & S.-F. Liu (2012). REBOUND: an open-source multi-purpose N-body code for
    collisional dynamics. *A&A*, 537, A128.

[5] P. G. Breen, C. N. Foley, T. Boekholt & S. Portegies Zwart (2020). Newton versus the
    machine: solving the chaotic three-body problem using deep neural networks.
    *MNRAS*, 494, 2465–2470.

[6] P. Hut, J. Makino & S. McMillan (1995). Building a better leapfrog.
    *ApJ Letters*, 443, L93–L96.

[7] J. D. Giorgini et al. (1996). JPL's on-line Solar System data service (HORIZONS).
    *BAAS*, 28, 1158.

*(Bibliographic details should be confirmed against the originals before final submission.)*
