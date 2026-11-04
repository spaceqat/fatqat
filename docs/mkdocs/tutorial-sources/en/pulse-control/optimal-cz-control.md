---
title: "Defining a near-time-optimal Rydberg CZ pulse"
description: "Recreate a compact approximation to the constant-amplitude, phase-shaped Rydberg CZ pulse of Jandura and Pupillo, simulate its two blockade-limit control blocks, and locate the conditional-phase crossing."
figure_alts:
  - "The authored Rydberg pulse over its dimensionless duration: constant unit-normalized amplitude and a smooth phase made from a linear ramp and two sine harmonics."
  - "Conditional phase versus dimensionless pulse duration between 7.5 and 7.7, showing a nearly linear crossing of pi near 7.614 while the computational-state return probability remains close to one."
---

# Defining a near-time-optimal Rydberg CZ pulse

Jandura and Pupillo identify a time-optimal global pulse for a two-atom
Rydberg CZ gate in the infinite-blockade limit. Their result has a particularly
simple control structure: the laser stays at its maximum amplitude while its
phase changes smoothly in time.

This tutorial recreates that control idea with FatQat. It does **not** run the
GRAPE or Pontryagin-maximum-principle optimizations from the paper. Instead, it
uses a compact analytic approximation to the published phase profile, shows
how to turn it into a complex waveform, and verifies the resulting gate with
the public two-level atom emulator.

The reference is:

> S. Jandura and G. Pupillo, *Time-Optimal Two- and Three-Qubit Gates for
> Rydberg Atoms*, [Quantum **6**, 712
> (2022)](https://doi.org/10.22331/q-2022-05-13-712),
> [arXiv:2202.00903](https://arxiv.org/abs/2202.00903).

The authors provide the phase samples used for their Figure 1(d) in a
[CC BY 4.0 data set](https://doi.org/10.6084/m9.figshare.19658427). The short
formula below approximates that profile; it is not a replacement for the
paper's optimization procedure or a new claim about the exact optimum.

## The three-level gate becomes two two-level problems

Each atom has computational states \(|0\rangle\) and \(|1\rangle\), plus a
Rydberg state \(|r\rangle\). A global laser couples \(|1\rangle\) to
\(|r\rangle\) with the same complex Rabi frequency \(\Omega(t)\) on both
atoms. The computational state \(|00\rangle\) is dark.

At infinite blockade, \(|rr\rangle\) cannot be populated. Symmetry then reduces
the other computational inputs to the two independent blocks of Eq. (16) in
the paper:

\[
\begin{aligned}
\{|01\rangle,|0r\rangle\}:&\qquad
H_{01}(t)=\frac{1}{2}
\begin{pmatrix}0&\Omega^*(t)\\\Omega(t)&0\end{pmatrix},\\[4pt]
\{|11\rangle,|W\rangle\}:&\qquad
H_{11}(t)=\frac{\sqrt2}{2}
\begin{pmatrix}0&\Omega^*(t)\\\Omega(t)&0\end{pmatrix},
\end{aligned}
\]

where \(|W\rangle=(|1r\rangle+|r1\rangle)/\sqrt2\). The \(|10\rangle\) block
is identical to the \(|01\rangle\) block. We can therefore use a one-site
`Atom2LevelEmulator` as an exact simulator for each reduced block: use the
authored drive once at its original strength and once multiplied by \(\sqrt2\).
The second factor is a collective coupling, not a stronger physical laser.

If the pulse returns both blocks to their computational states, write

\[
\langle01|U(T)|01\rangle=e^{i\xi_{01}},\qquad
\langle11|U(T)|11\rangle=e^{i\xi_{11}}.
\]

The global pulse is locally equivalent to CZ when

\[
\Phi_{\mathrm{CZ}}=\xi_{11}-2\xi_{01}=\pi\pmod{2\pi}.
\]

The return amplitudes must also have magnitude one. A phase crossing by itself
is not sufficient if population remains in \(|0r\rangle\) or \(|W\rangle\).

## Write down the pulse

In dimensionless time \(s=t/T\), the paper's time-optimal ansatz is

\[
\Omega(t)=\Omega_{\max}e^{i\varphi(s)},\qquad |\Omega(t)|=\Omega_{\max}.
\]

Keeping the amplitude below \(\Omega_{\max}\) cannot be time-optimal in the
ideal blockade-limit model: increasing the whole Hamiltonian and shortening
the duration traces the same trajectory faster. The phase remains the useful
control. It rotates the drive axis in the equatorial plane of each effective
two-level system; equivalently, in a frame following the phase,
\(\dot\varphi(t)\) acts as a detuning.

The two blocks rotate at different rates because their couplings differ by
\(\sqrt2\). A shaped phase can steer both trajectories back to their starting
states while leaving the non-additive phase \(\Phi_{\mathrm{CZ}}=\pi\).

For a self-contained tutorial, we use the compact three-term approximation

\[
\varphi(s)=c_0s+c_1\sin(2\pi s)+c_2\sin(4\pi s),
\]

with the coefficients below. It reaches unit fidelity in the reduced model at
\(T\Omega_{\max}\simeq7.614\), within \(0.03\%\) of the paper's optimized
value \(7.612\).

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

import fatqat as fq
import fatqat.operations as ops

OMEGA_MAX = 2 * np.pi * 5.0  # rad/us; the dimensionless result is scale-free
PULSE_DURATION = 7.6141739593  # T * Omega_max
PHASE_COEFFICIENTS = (-0.7746649293, -0.7667645594, 0.2183724760)

model = fq.emulator.Atom2LevelModel.from_document(
    fq.emulator.load_model_document("atom2level.reference")
)
block_backend = fq.emulator.Atom2LevelEmulator(
    model,
    arrangement=fq.emulator.AtomArrangement.chain(num_sites=1, spacing=1.0),
    method="unitary",
)


def laser_phase(scaled_time):
    """Compact approximation to the paper's smooth phase profile."""
    linear, first_harmonic, second_harmonic = PHASE_COEFFICIENTS
    return (
        linear * scaled_time
        + first_harmonic * np.sin(2 * np.pi * scaled_time)
        + second_harmonic * np.sin(4 * np.pi * scaled_time)
    )


physical_duration = PULSE_DURATION / OMEGA_MAX
print(f"T * Omega_max = {PULSE_DURATION:.6f}")
print(f"T = {1e3 * physical_duration:.3f} ns at Omega_max / 2pi = 5 MHz")
```

## Plot amplitude and phase

The amplitude plot is intentionally a horizontal line. The ideal pulse turns
on at \(\Omega_{\max}\), stays there for the whole gate, and turns off at the
end. Only the phase is shaped.

```python
scaled_times = np.linspace(0.0, 1.0, 401)
dimensionless_times = PULSE_DURATION * scaled_times
phases = laser_phase(scaled_times)
normalized_amplitude = np.ones_like(scaled_times)

figure, amplitude_axis = plt.subplots(figsize=(6.8, 3.8))
amplitude_line = amplitude_axis.plot(
    dimensionless_times,
    normalized_amplitude,
    color="tab:blue",
    label=r"amplitude $|\Omega|/\Omega_{\max}$",
)
amplitude_axis.set_xlabel(r"dimensionless time $t\Omega_{\max}$")
amplitude_axis.set_ylabel(r"$|\Omega|/\Omega_{\max}$")
amplitude_axis.set_ylim(0.0, 1.2)

phase_axis = amplitude_axis.twinx()
phase_line = phase_axis.plot(
    dimensionless_times,
    phases,
    color="tab:red",
    linestyle="--",
    label=r"phase $\varphi$",
)
phase_axis.set_ylabel(r"laser phase $\varphi$ (rad)")

lines = amplitude_line + phase_line
amplitude_axis.legend(lines, [line.get_label() for line in lines], loc="lower left")
figure.tight_layout()
plt.show()
```

This curve is one member of the symmetry pair discussed in the paper. Complex
conjugation reverses the phase profile and changes the removable single-qubit
phase, but implements the same CZ gate in the infinite-blockade limit. The
matrix convention above selects the member used by FatQat's drive channel.

## Simulate both control blocks

`block_return_amplitude` passes the plotted complex waveform directly to the
drive channel, runs it for either effective coupling, and returns the
computational-state amplitude at the end.

```python
def block_return_amplitude(coupling, duration, samples=201):
    """Return <q|U(T)|q> for one blockade-limit control block."""
    physical_time = duration / OMEGA_MAX
    times = np.linspace(0.0, physical_time, samples)
    scaled_time = times / physical_time
    phase = laser_phase(scaled_time)
    drive = coupling * OMEGA_MAX * np.exp(1j * phase)
    waveform = fq.emulator.SampledWaveform(tuple(times), tuple(drive))

    program = fq.Program(1)
    program.add(
        ops.PulseOperation(
            physical_time,
            (fq.emulator.PulseControl(model.control.drive(), waveform),),
        )
    )
    return complex(block_backend.run(program).result().get_unitary()[0, 0])


def conditional_phase(single_amplitude, double_amplitude):
    """Return xi_11 - 2 xi_01 wrapped to (-pi, pi]."""
    phase = np.angle(double_amplitude) - 2 * np.angle(single_amplitude)
    return float(np.angle(np.exp(1j * phase)))


def averaged_cz_fidelity(single_amplitude, double_amplitude, theta):
    """Paper Eq. (7), including return error outside the computational states."""
    corrected_single = np.exp(-1j * theta) * single_amplitude
    corrected_double = -np.exp(-2j * theta) * double_amplitude
    return float(
        (
            abs(1 + 2 * corrected_single + corrected_double) ** 2
            + 1
            + 2 * abs(corrected_single) ** 2
            + abs(corrected_double) ** 2
        )
        / 20
    )


def gate_metrics(duration, samples=201):
    """Return conditional phase, return probability, fidelity, and local phase."""
    single = block_return_amplitude(1.0, duration, samples)
    double = block_return_amplitude(np.sqrt(2.0), duration, samples)
    result = minimize_scalar(
        lambda theta: -averaged_cz_fidelity(single, double, theta),
        bounds=(-np.pi, np.pi),
        method="bounded",
    )
    return (
        conditional_phase(single, double),
        float(min(abs(single) ** 2, abs(double) ** 2)),
        float(-result.fun),
        float(result.x),
    )


phase, return_probability, fidelity, local_phase = gate_metrics(
    PULSE_DURATION,
    samples=401,
)
print(f"conditional phase magnitude = {abs(phase):.6f} rad")
print(f"worst computational-state return = {min(return_probability, 1.0):.10f}")
print(f"average CZ fidelity = {min(fidelity, 1.0):.10f}")
print(f"removable single-qubit phase theta = {local_phase:.6f} rad")
```

The fidelity is not inferred from the phase alone. Equation (7) also penalizes
failure to return \(|01\rangle\) and \(|11\rangle\) from their Rydberg-coupled
blocks, which is why the return probability is reported separately.

## Resolve one conditional-phase crossing

A wide duration scan mixes several wrapped copies of the phase and makes the
result look irregular. The paper resolves the optimum in the narrow interval
\(7.5\leq T\Omega_{\max}\leq7.7\); we use the same interval. Stretching the
fixed normalized phase profile across this window produces one nearly linear
crossing.

```python
durations = np.linspace(7.5, 7.7, 31)
wrapped_phases = []
return_probabilities = []

for duration in durations:
    phase, return_probability, _, _ = gate_metrics(float(duration))
    wrapped_phases.append(phase)
    return_probabilities.append(return_probability)

unwrapped_phases = np.unwrap(wrapped_phases)
middle = len(unwrapped_phases) // 2
unwrapped_phases += 2 * np.pi * round(
    (np.pi - unwrapped_phases[middle]) / (2 * np.pi)
)

if unwrapped_phases[0] < unwrapped_phases[-1]:
    crossing = float(np.interp(np.pi, unwrapped_phases, durations))
else:
    crossing = float(np.interp(np.pi, unwrapped_phases[::-1], durations[::-1]))

crossing_metrics = gate_metrics(crossing, samples=401)
print(f"pi crossing at T * Omega_max = {crossing:.6f}")
print(f"return probability there = {min(crossing_metrics[1], 1.0):.10f}")
print(f"average CZ fidelity there = {min(crossing_metrics[2], 1.0):.10f}")
```

```python
figure, phase_axis = plt.subplots(figsize=(6.8, 4.0))
phase_line = phase_axis.plot(
    durations,
    unwrapped_phases,
    color="tab:blue",
    marker="o",
    markersize=3,
    label=r"conditional phase $\Phi_{\mathrm{CZ}}$",
)
target_line = phase_axis.axhline(
    np.pi,
    color="0.35",
    linestyle=":",
    label=r"$\pi$ (CZ condition)",
)
crossing_point = phase_axis.plot(
    [crossing],
    [np.pi],
    color="tab:red",
    marker="*",
    markersize=12,
    label="crossing",
)
phase_axis.set_xlabel(r"dimensionless duration $T\Omega_{\max}$")
phase_axis.set_ylabel(r"unwrapped conditional phase (rad)")

return_axis = phase_axis.twinx()
return_line = return_axis.plot(
    durations,
    return_probabilities,
    color="tab:green",
    linestyle="--",
    label="worst return probability",
)
return_axis.set_ylabel("computational-state return probability")
return_axis.set_ylim(0.998, 1.0002)

lines = phase_line + [target_line] + crossing_point + return_line
phase_axis.legend(lines, [line.get_label() for line in lines], loc="lower left")
phase_axis.set_title("One CZ crossing near the time-optimal duration")
figure.tight_layout()
plt.show()
```

## What this tutorial does and does not establish

The example reproduces the essential physics and scale of the paper's
blockade-limit CZ pulse:

* the physical laser amplitude is constant at \(\Omega_{\max}\);
* a smooth phase profile coordinates blocks whose Rabi frequencies differ by
  \(\sqrt2\);
* both blocks return to the computational subspace; and
* their phases differ by the nonlocal angle \(\pi\).

The compact harmonic phase is an approximation to the published pulse, so the
tutorial calls it *near*-time-optimal. Proving time optimality requires the
duration sweep with re-optimization used in the paper, not merely stretching
one fixed profile. Finite blockade, Rydberg decay, and experimental control
limits are also outside this reduced-model tutorial; the paper treats those
effects separately.
