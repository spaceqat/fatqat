---
title: "Simulators"
---

# Simulators

FatQat's simulators run general quantum circuits and can account for the rules
of a device. On a superconducting layout, for example, a `CZ` can act only on
two connected sites. FatQat also simulates linear-optical circuits, following
how photons are distributed across modes after passing through optical
components.

All four simulators execute programs with `run()` and use the same interface
to read results. The table shows what each simulator is suited to.

| Simulator | Use it to |
| --- | --- |
| [`Simulator`](../simulator.md) | Run gate-level programs with qubits, qudits, or a mix of subsystem dimensions. |
| [`SCQubitSimulator`](sc-qubit.md) | Check superconducting native gates and qubit connectivity. |
| [`AtomArraySimulator`](atom-array.md) | Model neutral-atom loading, pairing, and loss during a run. |
| [`LOSimulator`](linear-optics.md) | Compute photon counts and quantum states for optical modes. |

A program can also include [custom operations](custom-operations.md) that read
or update the state of an individual shot. If the question is how pulses drive
a system over time, the [emulators](../emulators/index.md) describe that
evolution with a Hamiltonian.
