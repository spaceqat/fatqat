---
title: "Optical gates"
---

# Optical gates


These gates act on optical modes, where level `n` is the photon number. They
run on [`ModeRegister`][fatqat.ModeRegister] slots under
[`LOSimulator`](../simulators/linear-optics.md), and on fixed-dimension
[`QuantumRegister`][fatqat.QuantumRegister] slots, where level `d - 1` is the
largest photon number. Both gates keep the total photon number.

| Constructor | Targets | Action |
| --- | --- | --- |
| [`PhaseShifter`][fatqat.operations.PhaseShifter] `(phi)` | One mode or view | `\|n> -> exp(i*n*phi)\|n>`. On a dimension-2 target it is `Phase(phi)`. |
| [`BeamSplitter`][fatqat.operations.BeamSplitter] `(theta, phi=0.0)` | Two modes `(mode_0, mode_1)`, or two views | Mixes the modes with reflectivity `sin(theta)**2`; `theta=pi/4` is a 50:50 splitter. |

Angles are numbers in radians or [`Parameter`][fatqat.Parameter] values to
bind before execution.

## Beam splitter definition


With annihilation operators $a_0$ and $a_1$ of the two targets,

$$
\mathrm{BS}(\theta, \phi)
= \exp\!\left(\theta\left(e^{i\phi} a_1^\dagger a_0
  - e^{-i\phi} a_0^\dagger a_1\right)\right).
$$

On one photon, in the ordered basis $\lvert 1, 0\rangle, \lvert 0, 1\rangle$, it
acts as

$$
\begin{pmatrix}
\cos\theta & -e^{-i\phi}\sin\theta \\
e^{i\phi}\sin\theta & \cos\theta
\end{pmatrix}.
$$

For a dual-rail qubit with $\lvert 0\rangle = \lvert 1, 0\rangle$ and
$\lvert 1\rangle = \lvert 0, 1\rangle$, `BeamSplitter(a / 2, -pi / 2)` acts as
`RX(a)` and `BeamSplitter(a / 2)` acts as `RY(a)`.

## Truncation


A finite cutoff keeps both gates unitary. `PhaseShifter` is always exact.
`BeamSplitter` is exact while its two modes hold at most `cutoff` photons in
total (`d - 1` on dimension-`d` targets); beyond that, its result is an
approximation.

## API


::: fatqat.operations.PhaseShifter
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.operations.BeamSplitter
    options:
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
