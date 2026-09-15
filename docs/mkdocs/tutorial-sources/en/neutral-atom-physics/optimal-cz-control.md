---
title: "Time-optimal CZ control for Rydberg atoms"
description: "Reproduce the time-optimal global CZ pulse of Jandura and Pupillo with GRAPE: the phase profile, the Rydberg population, and the threshold duration that no shorter pulse can reach."
figure_alts:
  - "Two panels: the laser phase of the time-optimal CZ pulse as a GRAPE staircase and as a smooth curve, next to the populations of |0r> and |W> during the pulse, which peak and dip at different times because the double-excitation block is coupled sqrt(2) times more strongly."
  - "Gate error of the best GRAPE pulse versus dimensionless duration on a logarithmic scale, with the fitted parabola 1-F = A (T*-T)^2 marking the threshold T* below which no pulse reaches unit fidelity."
---

# Time-optimal CZ control for Rydberg atoms

Two neutral atoms in the Rydberg blockade regime implement a controlled-Z
gate when a laser drives (|1\rangle \leftrightarrow |r\rangle) on both atoms
at once. The gate is fast and simple, and one question decides its quality:
how short can the pulse be? This tutorial answers that question for the
two-atom gate by running the optimization of

> S. Jandura and G. Pupillo, *Time-Optimal Two- and Three-Qubit Gates for
> Rydberg Atoms*, [Quantum **6**, 712 (2022)](https://doi.org/10.22331/q-2022-06-20-712),
> [arXiv:2202.00903](https://arxiv.org/abs/2202.00903),

and reproduces its central numbers.

You need basic Python and the idea of a Rabi oscillation. No prior optimal
control is required. The optimizer works in the dimensionless units of the
paper: time in (1/\Omega_{\max}) and frequency in \(\Omega_{\max}\).

The tutorial uses the optimization helpers that ship with FatQat's
three-level atom family. That family is still private, so the imports below
reach into `fatqat.emulator._atom_3level.optimal_cz`; treat it as experimental
and expect the path to change.

## Why only two levels matter

In the blockade limit \(B\to\infty\) the state \(|rr\rangle\) is never
populated, and a *global* pulse makes the two atoms equivalent. The whole gate
then follows from two independent two-level systems (paper Eq. (16)):

| block | states | coupling |
| --- | --- | --- |
| one excitation | \(|01\rangle, |0r\rangle\) | \(\Omega/2\) |
| two excitations | \(|11\rangle, |W\rangle=(|1r\rangle+|r1\rangle)/\sqrt2\) | \(\sqrt2\,\Omega/2\) |

The pulse keeps its amplitude at the maximum \(\Omega_{\max}\) — that is what
time-optimality demands — and modulates only the laser phase \(\phi(t)\).
The figure of merit is the averaged gate error \(1-F\) of paper Eq. (7),
which depends on the two diagonal amplitudes \(\langle 01|U(T)|01\rangle\)
and \(\langle 11|U(T)|11\rangle\) and on the free single-qubit phase
\(\theta\) of the implemented phase gate.

```python
import matplotlib.pyplot as plt
import numpy as np

from fatqat.emulator._atom_3level.optimal_cz import (
    CZPulse,
    averaged_gate_fidelity,
    excited_state_populations,
    fit_time_optimal_duration,
    optimize_cz_pulse,
    scan_durations,
)

np.set_printoptions(precision=6, suppress=True)

# An infinitely short pulse is the identity, and the paper quotes its error as
# 0.4 for theta = pi/2.  This is the baseline every optimized pulse beats.
print("identity error:", 1.0 - averaged_gate_fidelity(1.0 + 0j, 1.0 + 0j, np.pi / 2))
```

## Optimize one duration

`optimize_cz_pulse` implements GRAPE with the ansatz of paper Sec. 3.1: the
phase is piecewise constant on `pieces` intervals, the amplitude stays at the
limit, and the gate error is minimized over every phase and over \(\theta\)
with analytic gradients. The paper reports the time-optimal duration
\(T_*\Omega_{\max}=7.612\), so we start there.

```python
pieces = 99
pulse = optimize_cz_pulse(7.612, pieces=pieces, seed=1, max_iterations=4000)

print(f"gate error at T*Omega_max = {pulse.duration:.3f}: {pulse.gate_error():.2e}")
print(f"average Rydberg time TR*Omega_max:            {pulse.rydberg_time():.4f}")
print(f"single-qubit phase theta:                    {pulse.theta:.4f}")
```

Both numbers match the paper: it quotes \(T_R\Omega_{\max}=2.957\) for this
pulse. The error is at machine precision, which means GRAPE found an exact
gate rather than a good approximation.

## The pulse in the time domain

The left panel shows the GRAPE staircase together with the same pulse sampled
on a uniform grid, which is the smooth control an experiment would program.
The right panel shows the population of the excited state in each block. The
\(|W\rangle\) population oscillates faster because the double-excitation
block is coupled by \(\sqrt2\) — the inset of Fig. 1(d) of the paper.

```python
smooth = pulse.resampled(401)
times = np.linspace(0.0, pulse.duration, 400)
one_excitation, two_excitations = excited_state_populations(pulse, times[1:])

figure, (phase_axis, population_axis) = plt.subplots(1, 2, figsize=(9.5, 3.6))
phase_axis.step(
    pulse.times[:-1], pulse.phases, where="post", label="GRAPE (99 pieces)"
)
phase_axis.plot(smooth.times, smooth.phases, label="smooth control")
phase_axis.set_xlabel(r"$t\,\Omega_{\max}$")
phase_axis.set_ylabel(r"laser phase $\phi$ (rad)")
phase_axis.set_title("Time-optimal phase")
phase_axis.legend(loc="lower right", fontsize="small")

population_axis.plot(times[1:], one_excitation, label=r"$|0r\rangle$ from $|01\rangle$")
population_axis.plot(times[1:], two_excitations, label=r"$|W\rangle$ from $|11\rangle$")
population_axis.set_xlabel(r"$t\,\Omega_{\max}$")
population_axis.set_ylabel("population")
population_axis.set_title("Rydberg population")
population_axis.legend(loc="upper right", fontsize="small")

figure.tight_layout()
plt.show()
```

## Where the optimum ends

Below some duration no pulse can reach unit fidelity. To locate it we optimize
a ladder of durations, each warm-started from the previous, slightly longer
pulse — the continuation strategy behind Fig. 1(c) of the paper.

```python
scan = scan_durations(longest=7.7, shortest=7.50, samples=21, pieces=pieces, seed=1)
fit = fit_time_optimal_duration(scan)

print(f"fitted threshold T*Omega_max: {fit.optimal_duration:.4f}   (paper: 7.612)")
print(f"fitted curvature A:           {fit.coefficient:.4f}   (paper: 0.0544)")
```

The gate error vanishes quadratically at the threshold, so a pulse just below
\(T_*\) is already unusable: the fit \(1-F=A(T_*-T)^2\) is the practical
statement of time-optimality.

```python
figure, axis = plt.subplots(figsize=(6.2, 4.0))
durations = np.asarray(scan.durations)
errors = np.asarray(scan.gate_errors)
reached = errors > 1e-9
axis.semilogy(durations[reached], errors[reached], "o", label="best GRAPE pulse")
grid = np.linspace(fit.optimal_duration - 0.2, fit.optimal_duration, 200)
axis.semilogy(
    grid,
    fit.coefficient * (fit.optimal_duration - grid) ** 2,
    label=r"$1-F = A\,(T_*-T)^2$",
)
axis.axvline(fit.optimal_duration, color="0.6", linestyle=":", label=r"$T_*$")
axis.set_xlabel(r"$T\,\Omega_{\max}$")
axis.set_ylabel("gate error  $1-F$")
axis.set_title("Threshold of the time-optimal CZ gate")
axis.legend()
figure.tight_layout()
plt.show()
```

## What to take away

* A single global laser pulse with a modulated phase realizes a CZ gate in
  \(T_*\Omega_{\max}=7.612\), about 10% faster than the pulse of
  Levine *et al.* (2019), which corresponds to \(8.585\).
* The result needs no single-site addressing, and the phase profile is smooth
  enough to program directly.
* Time-optimality is a threshold, not a slope: the error grows quadratically
  once the duration drops below \(T_*\).

The same module also carries the error analysis of the paper. The finite
blockade strength and a finite Rydberg lifetime enter as

\[
1-F=\frac{\Gamma(T_R\Omega_{\max})}{\Omega_{\max}}
+\frac{\Omega_{\max}^2\,\alpha}{B^2(T\Omega_{\max})^2},
\]

with \(\alpha=35.9\) for this pulse. Those helpers
(`quadratic_blockade_coefficient`, `rydberg_decay_error`,
`optimal_rabi_frequency`, and the emulator-backed
`emulated_cz_metrics` and `emulated_decay_fidelity`) are covered by the
package tests rather than plotted here.
