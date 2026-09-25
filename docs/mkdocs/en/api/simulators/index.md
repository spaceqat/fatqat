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

[`SCQubitSimulator`][fatqat.simulator.SCQubitSimulator] uses a configurable
superconducting coupling graph and accepts `X`, `SX`, virtual `RZ`, and
coupled `CZ`. It offers an optional reference noise model.

[`SCQubitQEC17Simulator`][fatqat.simulator.SCQubitQEC17Simulator] uses the fixed
QZ01 17-qubit graph and native gates. Its optional calibration model applies
individual qubit and coupler noise, including scheduled idle relaxation.

[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] has no fixed
connectivity. Every program site starts empty: `Put` loads atoms, `Loss`
removes them, and `Pair` and `Unpair` change which atoms can interact.

The profiles validate the program as written: they do not transpile or route
it. QEC17 uses fixed gate durations for idle accounting; the other profiles
have no timing model. Use the [pulse emulators](../emulators/index.md) for
pulse-resolved timing and Hamiltonian evolution.

**Choose a simulator**

| Class | Use it for | Main constraint |
| --- | --- | --- |
| [`Simulator`][fatqat.simulator.Simulator] | General circuit simulation and custom matrix implementations | No device topology |
| [`SCQubitSimulator`][fatqat.simulator.SCQubitSimulator] | Constrained superconducting native-gate experiments | `X`, `SX`, `RZ`; `CZ` on configured couplings |
| [`SCQubitQEC17Simulator`][fatqat.simulator.SCQubitQEC17Simulator] | Individual QZ01 calibration errors and idle decoherence | Fixed 17-qubit topology and QZ01 native gates |
| [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] | Neutral-atom occupancy, loss, and dynamic connectivity | `RX`, `RY`, `RZ`, and paired `CZ` |

- [Simulator](../simulator.md)
- [SCQubitSimulator](sc-qubit.md)
- [QEC17 simulator](qec17.md)
- [AtomArraySimulator](atom-array.md)
