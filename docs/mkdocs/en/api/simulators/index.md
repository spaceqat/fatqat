---
title: "Simulators"
---

# Simulators


FatQat provides a general circuit simulator and two hardware profiles. They
use the same run and result API. Choose [`Simulator`][fatqat.simulator.Simulator] for unrestricted
gate-level work, or a profile when the program must obey a native gate set,
layout, or connectivity rule.

Start with [Choose how much physics to model](../../guide/execution-models.md) to choose an execution level, or
follow [Test a Program against a hardware profile](../../guide/hardware-profile-simulation.md) for the profile workflow.

[`SCQubitSimulator`][fatqat.simulator.SCQubitSimulator] uses a configurable
superconducting coupling graph and accepts `X`, `SX`, virtual `RZ`, and
coupled `CZ`. It offers an optional reference noise model.

[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] has no fixed
connectivity. Every program site starts empty: `Put` loads atoms, `Loss`
removes them, and `Pair` and `Unpair` change which atoms can interact.

The profiles validate the program as written: they do not transpile or route
it, and they do not reproduce a named processor. Use the
[pulse emulators](../emulators/index.md) when timing or Hamiltonian
evolution matters.

**Choose a simulator**

| Class | Use it for | Main constraint |
| --- | --- | --- |
| [`Simulator`][fatqat.simulator.Simulator] | General circuit simulation and custom matrix implementations | No device topology |
| [`SCQubitSimulator`][fatqat.simulator.SCQubitSimulator] | Constrained superconducting native-gate experiments | `X`, `SX`, `RZ`; `CZ` on configured couplings |
| [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] | Neutral-atom occupancy, loss, and dynamic connectivity | `RX`, `RY`, `RZ`, and paired `CZ` |

- [Simulator](../simulator.md)
- [SCQubitSimulator](sc-qubit.md)
- [AtomArraySimulator](atom-array.md)
