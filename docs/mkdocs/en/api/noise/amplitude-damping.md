---
title: "AmplitudeDamping"
---

# AmplitudeDamping


[`AmplitudeDamping`][fatqat.noise.AmplitudeDamping] describes conventional
qubit decay from $|1\rangle$ to $|0\rangle$. Use `p` for a simulator channel
or `rate` for a local Lindblad operator on a compatible emulator.

## Choose probabilities or rates


Supply exactly one scalar `p` or `rate`. A probability must lie in
`[0, 1]`; a rate must be finite and nonnegative. The descriptor acts on one
qubit and does not accept an iterable. Use
[`TransitionRelaxation`](transition-relaxation.md) for an explicitly authored
finite-dimensional jump operator.

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

Rates are finite, nonnegative, and measured in the inverse of the backend's
time unit. See [Pulse emulators](backend-support.md#noise-emulator-support) for each built-in emulator's local
dimension, accepted scope, and implementation-map requirements.

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

## API


::: fatqat.noise.AmplitudeDamping
    options:
      members:
        - "as_probability"
        - "as_rate"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
