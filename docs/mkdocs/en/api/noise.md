---
title: "Noise"
---

# Noise

Use a `NoiseModel` to vary the noise in a run while keeping the same
[`Program`][fatqat.Program]. Simulators can apply a channel after an operation.
Pulse emulators can evolve the state with continuous noise, including during idle
time. Readout confusion changes the reported digits in either case.

Noise types are available in `fatqat.noise`. Pass a
[`NoiseModel`][fatqat.NoiseModel] to the backend with `noise=...`.

## Choose a noise type

Choose a type by the effect you want to model. For a type with both forms, use a
probability for a single channel application or a rate or relaxation time for
continuous evolution. Follow the definition for your backend on the type's page.

**Built-in noise types**

| Noise type | Accepted parameters | Applies to | User-visible effect |
| --- | --- | --- | --- |
| [`Depolarizing`][fatqat.noise.Depolarizing] | Exactly one of `p` or `rate` | `p`: the selected operands; `rate`: one subsystem | Uniform mixing toward the maximally mixed state |
| [`PauliChannel`][fatqat.noise.PauliChannel] | Pauli-string probability mapping or pair sequence | One qubit per string character | A stochastic mixture of `I`, `X`, `Y`, and `Z` strings |
| [`AmplitudeDamping`][fatqat.noise.AmplitudeDamping] | Exactly one scalar `p` or `rate` | One qubit | Conventional finite channel or continuous qubit energy relaxation |
| [`TransitionRelaxation`][fatqat.noise.TransitionRelaxation] | Exactly one scalar `p` or `rate`, plus explicit transition coefficients | One subsystem | One explicitly authored finite or continuous jump operator |
| [`PhaseDamping`][fatqat.noise.PhaseDamping] | Exactly one of `p`, `rate`, or `t_phi` | One subsystem | Coherence decay without population transfer |
| [`ThermalRelaxation`][fatqat.noise.ThermalRelaxation] | `t1` and `t2` | One qubit | Combined energy relaxation and residual pure dephasing |
| [`Loss`][fatqat.noise.Loss] | Per-carrier probability `p` | Every selected carrier in a matched operation | Persistent carrier removal on an occupancy-aware backend |
| [`ReadoutConfusion`][fatqat.noise.ReadoutConfusion] | Column-stochastic report matrix | Each measured subsystem independently, or one selected subsystem | Classical resampling of the reported digit after physical collapse |

See [Noise model](noise/model.md) for operation and background noise, target selection,
composition, conflicts, and validation timing. Each noise-type page covers
its parameters, units, and mathematical definition.

## Backend support

Check [Backend support](noise/backend-support.md#noise-backend-support) for the
forms, scopes, and target dimensions your backend accepts.

## Quick start

### After an operation

Pass `operation=ops.CX` to apply a depolarizing channel jointly to the two
operands after each `CX`. For a full comparison, see [Ideal and noisy
runs](../guide/ideal-and-noisy.md).

```python
import fatqat as fq
import fatqat.operations as ops

noise = fq.NoiseModel()
noise.add(fq.noise.Depolarizing(p=0.05), operation=ops.CX)
simulator = fq.simulator.Simulator(method="density_matrix", noise=noise)
```

### During pulse evolution

Omit `operation` to keep relaxation active on `"q0"` throughout the elapsed time,
including idle periods. The reference transmon model uses nanoseconds, so `rate`
is in inverse nanoseconds.

```python
import fatqat as fq
from fatqat.emulator import TransmonEmulator, TransmonModel, load_model_document

model = TransmonModel.from_document(load_model_document("transmon.reference"))
noise = fq.NoiseModel()
noise.add(
    fq.noise.TransitionRelaxation(
        rate=1 / 60_000,
        coefficients={(1, 0): 1, (2, 1): 2**0.5},
    ),
    targets="q0",
)
emulator = TransmonEmulator(model, method="density_matrix", noise=noise)
```

## API pages

- [Noise model](noise/model.md)
- [Backend support](noise/backend-support.md)
- [Depolarizing](noise/depolarizing.md)
- [PauliChannel](noise/pauli-channel.md)
- [AmplitudeDamping](noise/amplitude-damping.md)
- [TransitionRelaxation](noise/transition-relaxation.md)
- [PhaseDamping](noise/phase-damping.md)
- [ThermalRelaxation](noise/thermal-relaxation.md)
- [Loss](noise/loss.md)
- [ReadoutConfusion](noise/readout-confusion.md)
- [Custom noise implementations](noise/custom-implementations.md)
