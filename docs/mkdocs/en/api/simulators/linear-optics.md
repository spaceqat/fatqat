---
title: "LOSimulator"
---

# LOSimulator


[`LOSimulator`][fatqat.simulator.LOSimulator] is a
[`Simulator`](../simulator.md) for Programs with optical modes. It runs,
returns results, and handles noise and sweeps like `Simulator`, except as
listed below. For a worked example, see
[Simulate linear optics](../../guide/linear-optics.md).

**Differences from `Simulator`**

| Topic | `LOSimulator` |
| --- | --- |
| Registers | Also accepts [`ModeRegister`][fatqat.ModeRegister] and [`CountRegister`][fatqat.CountRegister]; `Simulator` rejects them |
| Methods | Adds `"fixed_photon_state"`, the default |
| `execution_config` | Adds `"cutoff"` |
| [`Estimator`][fatqat.Estimator] | On a mode, accepts [`NumberOperator`][fatqat.observable.NumberOperator] and [`Projector`][fatqat.observable.Projector] up to level `cutoff`; Pauli operators raise at any cutoff |
| Noise selectors | Select modes by device label; a `ModeRegister` reference is not supported |

## API


::: fatqat.simulator.LOSimulator
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: true
      filters:
        - "!^_"

::: fatqat.simulator.fixed_photon_index

::: fatqat.simulator.fixed_photon_occupation
