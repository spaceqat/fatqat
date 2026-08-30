---
title: "ThermalRelaxation"
---

# ThermalRelaxation


[`ThermalRelaxation`][fatqat.noise.ThermalRelaxation] models zero-temperature T1 and T2 relaxation on an
emulator. It combines downward population decay with the additional pure
dephasing needed to reproduce T2. It is a continuous-time emulator
declaration, not a finite simulator channel. Matrix simulators instead use
operation-bound probability-form [`AmplitudeDamping`][fatqat.noise.AmplitudeDamping] and
[`PhaseDamping`][fatqat.noise.PhaseDamping] descriptors.

## Times and rates


`t1` and `t2` are finite positive values in the same unit; they have no
intrinsic unit of their own. When registering the noise with an emulator, use
the emulator model's time unit. Physical consistency requires

$$
T_2 \leq 2T_1.
$$

The derived rates are

$$
\gamma_1 = \frac{1}{T_1}, \qquad
\gamma_\phi = \frac{1}{T_2}-\frac{1}{2T_1}.
$$

[`ThermalRelaxation.amplitude_rate`][fatqat.noise.ThermalRelaxation.amplitude_rate] returns $\gamma_1$, and
[`ThermalRelaxation.pure_dephasing_rate`][fatqat.noise.ThermalRelaxation.pure_dephasing_rate] returns $\gamma_\phi$.
The latter is nonnegative under the T2 bound and becomes zero at
$T_2=2T_1$.

## Pulse emulators


For a $d$-level pulse model, the emulator uses the local Lindblad
operators

$$
\begin{aligned}
L_1 &= \sum_{k=1}^{d-1}\sqrt{\frac{k}{T_1}}
       |k-1\rangle\!\langle k|,\\
L_\phi &= \sqrt{2\gamma_\phi}\,
          \operatorname{diag}(0,1,\ldots,d-1).
\end{aligned}
$$

The second operator is omitted when $\gamma_\phi=0$. This is a local,
zero-temperature model: it includes downward relaxation but no thermal
excitation or equilibrium-population parameter.

See [Pulse emulators](backend-support.md#noise-emulator-support) for the accepted scopes and implementation-map
requirements of each built-in emulator.

## Matrix simulators


For a known qubit operation duration $t$, the corresponding finite
probabilities are below. The duration must use the same time unit as `t1`
and `t2`.

$$
\begin{aligned}
p_1(t) &= 1-e^{-t/T_1},\\
p_\phi(t) &= 1-e^{-\gamma_\phi t}.
\end{aligned}
$$

Construct the two probability-form descriptors explicitly. The rate-form
descriptors provide the public conversion methods:

```python
import fatqat as fq

relaxation = fq.noise.ThermalRelaxation(t1=60e-6, t2=80e-6)
duration = 2e-6

amplitude_source = fq.noise.AmplitudeDamping(
    rate=relaxation.amplitude_rate
)
phase_source = fq.noise.PhaseDamping(
    rate=relaxation.pure_dephasing_rate
)
damping = fq.noise.AmplitudeDamping(
    p=amplitude_source.as_probability(duration)
)
dephasing = fq.noise.PhaseDamping(
    p=phase_source.as_probability(duration)
)
```

Register `damping` and `dephasing` on the relevant operation. For a qubit,
their composition gives population decay $e^{-t/T_1}$ and coherence
decay $e^{-t/T_2}$.

This construction is for qubits. A higher-dimensional amplitude-damping
channel needs $d-1$ values, and the elementwise probability conversion
does not reproduce full multilevel Lindblad evolution. See
[Simulators](backend-support.md#noise-simulator-support) for simulator support.

## API


::: fatqat.noise.ThermalRelaxation
    options:
      members:
        - "amplitude_rate"
        - "pure_dephasing_rate"
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
