---
title: "Depolarizing"
---

# Depolarizing

[`Depolarizing`][fatqat.noise.Depolarizing] mixes the quantum state with the
maximally mixed state.

<a id="choose-a-form"></a>

Supply exactly one parameter. Use `p` for a single channel application in a
simulator, or `rate` for continuous noise in a pulse emulator.

## Simulators

`p` is the weight of complete depolarization and lies in `[0, 1]`. For selected
operands with combined dimension $d$, the channel is

$$
\mathcal{E}_p(\rho) = (1-p)\rho + p\frac{I_d}{d}.
$$

The channel acts jointly on all selected operands. Applying
`Depolarizing(p=...)` to two qubits therefore uses $d=4$; it does not
create two independent single-qubit channels. For independent local noise,
use `target_positions` or separate registrations.

Here `p` is the weight of complete depolarization, not the probability of
choosing a nonidentity error. For one qubit, the total probability assigned to
the nonidentity branches is $3p/4$.

See [Simulators](backend-support.md#noise-simulator-support) for method and
backend support.

## Pulse emulators

`rate` is finite and nonnegative, in the inverse of the model's time unit. It acts
on one subsystem. For local dimension $d$, the Lindblad generator is

$$
\mathcal{L}_r(\rho)
= r\left(\operatorname{Tr}(\rho)\frac{I_d}{d}-\rho\right).
$$

See [Pulse emulators](backend-support.md#noise-emulator-support) for the supported
scopes and local dimensions.

## Converting between forms

For depolarizing noise acting on its own for a duration $t$, the corresponding
probability parameter is

$$
p(t)=1-e^{-rt}.
$$

[`Depolarizing.as_probability`][fatqat.noise.Depolarizing.as_probability] and [`Depolarizing.as_rate`][fatqat.noise.Depolarizing.as_rate] perform
this parameter conversion when you know the duration. A duration must be
finite and nonnegative. Probability 1 has no finite rate, and a nonzero
probability cannot be converted at zero duration.

## API reference { #api }

::: fatqat.noise.Depolarizing
    options:
      members:
        - "as_probability"
        - "as_rate"
        - "num_subsystems"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
