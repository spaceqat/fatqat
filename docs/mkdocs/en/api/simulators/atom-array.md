---
title: "AtomArraySimulator"
---

# AtomArraySimulator


[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] adds neutral-atom occupancy, loss, and dynamic
pairing to the [`Simulator`][fatqat.simulator.Simulator] execution model. Use it to check a program
against those constraints. It is not a Hamiltonian or transport model; use
the [neutral-atom emulator](../atom-emulators.md) when pulse timing and physical interactions matter.

**Hardware profile**

| Property | Value |
| --- | --- |
| Site count | Taken from the quantum resources declared by the `Program` |
| Native gates | [`fatqat.operations.RX`][fatqat.operations.RX], [`fatqat.operations.RY`][fatqat.operations.RY], [`fatqat.operations.RZ`][fatqat.operations.RZ], and [`fatqat.operations.CZ`][fatqat.operations.CZ] |
| Connectivity | No fixed topology; `CZ` is legal only while its two atoms are paired |
| Dimensions | Qubits only |
| Methods | `statevector` and `density_matrix` |
| Runtime | `numba` by default; `numpy` is also supported |
| Noise | Ideal by default; no built-in reference model |

## Program size, mapping, and native operations


The program declares the array size through its quantum resources; the backend
has no separate site-count or capacity argument. Registers map to integer
device labels in declaration order; a [`GridRegister`][fatqat.GridRegister] is
flattened and its coordinates have no physical meaning on this backend.

The native operations are `RX`, `RY`, `RZ`, and `CZ`. The simulator
does not decompose other gates, so `CX` is rejected even when the atoms are
paired. [`AtomArraySimulator.implementation_map`][fatqat.simulator.AtomArraySimulator.implementation_map] lists the native gate
set; the program's current `Pair` state determines whether a particular
`CZ` is allowed.

Native-operation and pairing checks happen independently of occupancy. An
unsupported gate or unpaired `CZ` is therefore rejected even if one of its
sites is never loaded. Only an otherwise-valid supported gate can become an
empty-site no-op.

Pairing changes as the program runs. An unconditional
[`fatqat.operations.Pair`][fatqat.operations.Pair] connects two sites and
[`fatqat.operations.Unpair`][fatqat.operations.Unpair] disconnects them. A `CZ` on an unpaired
pair is rejected. Pairing operations do not change the quantum state, though
noise attached to them still applies. Conditional `Pair` and `Unpair` are
not supported.

## Occupancy and loss


Occupancy is tracked separately for every shot. Every declared site starts
empty, independently of whether the program contains `Put` or a matching loss
source:

**Occupancy rules**

| Program action | Occupancy behavior |
| --- | --- |
| [`fatqat.operations.Put`][fatqat.operations.Put] runs for a site | Loads a fresh `\|0>` atom if the site is empty. |
| No `Put` runs for a site | The site remains empty: gates and reset do nothing, and measurement reports erasure digit `2`. |

`Put` on an occupied site has no effect. A gate or reset on an empty or
previously lost site likewise has no effect for that shot. Measurement still
reports an erasure, pairing still changes connectivity, and a later `Put`
can refill a lost site.

[`fatqat.noise.Loss`][fatqat.noise.Loss] can eject gate targets, make `Put` fail, or model
loss during `Pair` and `Unpair`. It affects a run only when its selector
matches an operation; a matching source with `p=0` removes no atoms and does
not change the explicit-loading rule. This is the only gate-level simulator
that accepts `Loss`. A missing atom makes an otherwise valid paired `CZ` do
nothing for that shot; an unpaired `CZ` is rejected before execution.

Measurement of an empty site reports the erasure digit `2`. Erasure bypasses
readout-confusion noise because there is no occupied qubit to read. Atom loss
makes the final state stochastic, so `final_state=True` requires
`shots == 1`. A measured lossy run returns counts by default but not an
arbitrary trajectory's final state.

## Example


```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
program.add(ops.Put, (0, 1))
program.add(ops.Pair, (0, 1))
program.add(ops.RY(0.4), 0)
program.add(ops.CZ, (0, 1))
program.measure_all()

backend = fq.simulator.AtomArraySimulator()
counts = backend.run(program, shots=1000).result().get_counts()
```

The atom lifecycle cannot be represented by `unitary` or `superop` because
occupancy is state outside the quantum matrix. `AtomArraySimulator` therefore
accepts only `statevector` and `density_matrix`; selecting `unitary` or
`superop` is rejected when the backend is constructed, even if a program would
contain no `Put` or loss.

## API


[`Simulator.method`][fatqat.simulator.Simulator.method], [`AtomArraySimulator.implementation_map`][fatqat.simulator.AtomArraySimulator.implementation_map],
[`Simulator.run`][fatqat.simulator.Simulator.run], [`Simulator.run_sweep`][fatqat.simulator.Simulator.run_sweep], and
[`Simulator.validate_noise_model`][fatqat.simulator.Simulator.validate_noise_model] follow the general Simulator API and are
included below for a complete class reference.

::: fatqat.simulator.AtomArraySimulator
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: true
      filters:
        - "!^_"
