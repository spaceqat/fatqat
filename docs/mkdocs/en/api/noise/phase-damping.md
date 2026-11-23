---
title: "PhaseDamping"
---

# PhaseDamping

[`PhaseDamping`][fatqat.noise.PhaseDamping] reduces coherence between energy
levels without changing the probability of finding the system in each level.

<a id="choose-a-form"></a>

Supply exactly one parameter. Use `p` in a simulator. In a pulse emulator, specify
the rate with `rate` or the pure-dephasing time with `t_phi`. In a multilevel
system, the simulator and emulator describe different coherence decay.

## Simulators

`p` is the weight of complete dephasing in one channel application. It must be
finite and in `[0, 1]`. For a target of dimension $d$, the channel is

$$
\mathcal{E}_p(\rho)
= (1-p)\rho + p\,\operatorname{diag}(\operatorname{diag}(\rho)).
$$

All populations are unchanged, and every off-diagonal entry is multiplied by
$1-p$, regardless of the separation between its two levels. The channel
acts on one selected operand.

## Pulse emulators

Pass a finite nonnegative `rate` in inverse time or a finite positive `t_phi` in
the model's time unit. If you supply a time, the object stores `rate = 1 / t_phi`.
The local Lindblad operator for rate $r$ is

$$
L=\sqrt{2r}\,\operatorname{diag}(0,1,\ldots,d-1).
$$

Its effect on coherence between levels j and k is

$$
\rho_{jk}(t)=e^{-r(j-k)^2t}\rho_{jk}(0).
$$

See [Pulse emulators](backend-support.md#noise-emulator-support) for the supported
scopes.

## Converting between forms

[`PhaseDamping.as_probability`][fatqat.noise.PhaseDamping.as_probability] uses $p=1-e^{-rt}$, and
[`PhaseDamping.as_rate`][fatqat.noise.PhaseDamping.as_rate] uses $r=-\log(1-p)/t$. These relations match
a qubit channel and adjacent-level coherence. For larger systems, the two
forms differ: the simulator channel damps every coherence uniformly, while
Lindblad evolution scales the decay with $(j-k)^2$.

Duration must be finite and nonnegative. Probability 1 has no finite rate, and
a nonzero probability cannot be converted at zero duration.

## API reference { #api }

::: fatqat.noise.PhaseDamping
    options:
      members:
        - "as_probability"
        - "as_rate"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
