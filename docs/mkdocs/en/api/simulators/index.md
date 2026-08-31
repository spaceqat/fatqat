---
title: "Simulators"
---

# Simulators


FatQat provides a general circuit simulator and three hardware profiles. They
use the same run and result API. Choose [`Simulator`][fatqat.simulator.Simulator] for unrestricted
gate-level work, or a profile when the program must obey a native gate set,
layout, or connectivity rule.

Start with [Choose how much physics to model](../../guide/execution-models.md) to choose an execution level, or
follow [Test a Program against a hardware profile](../../guide/hardware-profile-simulation.md) for the profile workflow.

The superconducting profiles use a fixed rectangular grid.
[`SCQubitIBMSimulator`][fatqat.simulator.SCQubitIBMSimulator] accepts IBM-style native gates and
nearest-neighbour `CZ`. [`SCQubitGoogleSimulator`][fatqat.simulator.SCQubitGoogleSimulator] accepts native
rotations and nearest-neighbour `iSwap` and `CZ`. Both offer an optional
reference noise model.

[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] has no fixed connectivity. `Pair` and `Unpair`
change which atoms can interact, while `Put` and `Loss` control occupancy.

The profiles validate the program as written: they do not transpile or route
it, and they do not reproduce a named processor. Use the
[pulse emulators](../emulators/index.md) when timing or Hamiltonian
evolution matters.

**Choose a simulator**

| Class | Use it for | Main constraint |
| --- | --- | --- |
| [`Simulator`][fatqat.simulator.Simulator] | General circuit simulation and custom matrix implementations | No device topology |
| [`SCQubitIBMSimulator`][fatqat.simulator.SCQubitIBMSimulator] | IBM-style native-gate and grid experiments | `X`, `SX`, `RZ`; nearest-neighbour `CZ` |
| [`SCQubitGoogleSimulator`][fatqat.simulator.SCQubitGoogleSimulator] | Google-style native-gate and grid experiments | `RX`, `RY`, `RZ`; nearest-neighbour `iSwap` and `CZ` |
| [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] | Neutral-atom occupancy, loss, and dynamic connectivity | `RX`, `RY`, `RZ`, and paired `CZ` |

- [Simulator](../simulator.md)
- [SCQubitIBMSimulator](sc-qubit-ibm.md)
- [SCQubitGoogleSimulator](sc-qubit-google.md)
- [AtomArraySimulator](atom-array.md)
