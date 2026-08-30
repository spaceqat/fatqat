---
title: "Noise"
---

# Noise


FatQat keeps noise separate from [`Program`][fatqat.Program]. Add the noise
sources you need to a [`NoiseModel`][fatqat.NoiseModel], then pass that model to a
compatible simulator or emulator with `noise=...`. You can reuse the same
program for ideal and noisy runs.

See [Compare the same Program ideally and noisily](../guide/ideal-and-noisy.md) for a controlled ideal-versus-noisy
comparison. This section is the reference for selectors, support, units, and
validation.

Noise types live in `fatqat.noise`. `NoiseModel` is also available as
`fatqat.NoiseModel`.

## Choose a noise type


Probabilities describe one simulator channel application after a matched
operation. Rates and relaxation times describe local Lindblad operators that
act over emulator time. Backends do not convert between these forms.

**Built-in noise types**

| Noise type | Accepted parameters | Applies to | User-visible effect |
| --- | --- | --- | --- |
| [`Depolarizing`][fatqat.noise.Depolarizing] | Exactly one of `p` or `rate` | `p`: the selected operands; `rate`: one subsystem | Uniform mixing toward the maximally mixed state |
| [`PauliChannel`][fatqat.noise.PauliChannel] | Pauli-string probability mapping or pair sequence | One qubit per string character | A stochastic mixture of `I`, `X`, `Y`, and `Z` strings |
| [`AmplitudeDamping`][fatqat.noise.AmplitudeDamping] | Exactly one of `p` or `rate`, with one value per adjacent transition | One subsystem | Ladder decay from level `k` to `k - 1` |
| [`PhaseDamping`][fatqat.noise.PhaseDamping] | Exactly one of `p`, `rate`, or `t_phi` | One subsystem | Coherence decay without population transfer |
| [`ThermalRelaxation`][fatqat.noise.ThermalRelaxation] | `t1` and `t2` | One subsystem | Combined energy relaxation and residual pure dephasing |
| [`Loss`][fatqat.noise.Loss] | Per-carrier probability `p` | Every selected carrier in a matched operation | Persistent carrier removal on an occupancy-aware backend |
| [`ReadoutConfusion`][fatqat.noise.ReadoutConfusion] | Column-stochastic report matrix | Each measured subsystem independently, or one selected subsystem | Classical resampling of the reported digit after physical collapse |

See [Noise model](noise/model.md) for operation and background noise, target selection,
composition, conflicts, and validation timing. Each noise-type page covers
its parameters, units, and mathematical definition.

## Backend support


Support depends on the noise form, where it applies, and the backend. The
[Backend support](noise/backend-support.md#noise-backend-support) tables show the built-in behavior and unsupported
forms for each backend family. Pulse-emulator continuous-noise realizations are
family-owned rather than user-replaceable.

## Quick start


This model adds a joint channel after every `CX` and then applies binary
readout confusion to every measurement:

```python
import numpy as np
import fatqat as fq
import fatqat.operations as ops

noise = fq.NoiseModel()
noise.add(fq.noise.Depolarizing(p=0.05), operation=ops.CX)
noise.add(
    fq.noise.ReadoutConfusion(
        np.array([[0.98, 0.04], [0.02, 0.96]])
    )
)

backend = fq.simulator.Simulator(method="density_matrix", noise=noise)
```

For a pulse backend, express rates and relaxation times in the model's time
unit. For example, the reference transmon model uses device labels such as
`"q0"`. This relaxation noise stays active there throughout elapsed pulse
time:

```python
pulse_noise = fq.NoiseModel()
pulse_noise.add(
    fq.noise.ThermalRelaxation(t1=60_000.0, t2=80_000.0),
    targets="q0",
)
```

The reference transmon model uses nanoseconds, while the neutral-atom model
uses microseconds. Check the chosen model's `time_unit` instead of inferring a
unit from the size of a value: see
[`time_unit`][fatqat.emulator.TransmonModel.time_unit] and
[`time_unit`][fatqat.emulator.Atom2LevelModel.time_unit].

## API pages


- [Noise model](noise/model.md)
- [Backend support](noise/backend-support.md)
- [Depolarizing](noise/depolarizing.md)
- [PauliChannel](noise/pauli-channel.md)
- [AmplitudeDamping](noise/amplitude-damping.md)
- [PhaseDamping](noise/phase-damping.md)
- [ThermalRelaxation](noise/thermal-relaxation.md)
- [Loss](noise/loss.md)
- [ReadoutConfusion](noise/readout-confusion.md)
- [Custom noise implementations](noise/custom-implementations.md)
