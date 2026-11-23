---
title: "Optical gates"
---

# Optical gates

These gates conserve total photon number. Use them on
[`ModeRegister`][fatqat.ModeRegister] with
[`LOSimulator`](../simulators/linear-optics.md), or on fixed-dimension
[`QuantumRegister`][fatqat.QuantumRegister], where level `n` represents
`n` photons.

| Constructor | Use it to |
| --- | --- |
| [`PhaseShifter`][fatqat.operations.PhaseShifter] `(phi)` | Apply a photon-number-dependent phase to one mode. |
| [`BeamSplitter`][fatqat.operations.BeamSplitter] `(theta, phi=0.0)` | Mix two modes with reflectivity `sin(theta)**2`. Use `theta=pi/4` for a balanced splitter. See [basis order and phase convention](#beam-splitter-definition). |

<a id="api"></a>

## API reference


::: fatqat.operations.PhaseShifter
    options:
      heading_level: 3
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

<a id="truncation"></a>

<a id="fatqat.operations.BeamSplitter.validate_targets"></a>

::: fatqat.operations.BeamSplitter
    options:
      heading_level: 3
      inherited_members: false
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
        - "!^(?:validate_targets)$"

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
