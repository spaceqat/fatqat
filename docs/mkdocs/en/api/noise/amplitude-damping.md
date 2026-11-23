---
title: "AmplitudeDamping"
---

# AmplitudeDamping

[`AmplitudeDamping`][fatqat.noise.AmplitudeDamping] models the decay of one qubit
from $|1\rangle$ to $|0\rangle$.

<a id="choose-probabilities-or-rates"></a>

Supply exactly one parameter. Use `p` for a single channel application in a
simulator, or `rate` for continuous noise in a pulse emulator.

To specify your own transitions or work with more than two levels, use
[`TransitionRelaxation`](transition-relaxation.md).

## Simulators

For probability $p$, the simulator uses

$$
K_0 =
\begin{pmatrix}
1 & 0 \\
0 & \sqrt{1-p}
\end{pmatrix},
\qquad
K_1 = \sqrt{p}|0\rangle\!\langle1|.
$$

The channel acts on one selected qubit; use `target_positions` to choose an
operand of a multi-operand gate. See [Simulators](backend-support.md#noise-simulator-support) for
built-in availability.

## Pulse emulators

For rate $r$, a compatible pulse backend uses the local Lindblad operator

$$
L = \sqrt{r}|0\rangle\!\langle1|.
$$

Use the inverse of the model's time unit for `rate`. The built-in
[`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator] accepts this form as
background noise.

## Converting between forms

The conversion methods apply

$$
p(t)=1-e^{-rt}
$$

This is the exact isolated qubit amplitude-damping relation. Applying the
resulting channel after a driven operation is not generally equivalent to
simultaneous Hamiltonian and Lindblad evolution.

Duration must be finite and nonnegative. A probability of 1 has no finite
rate, and a nonzero probability cannot be converted to a finite rate at zero
duration.

## API reference { #api }

::: fatqat.noise.AmplitudeDamping
    options:
      members:
        - "as_probability"
        - "as_rate"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
