---
title: "AtomArraySimulator"
---

# AtomArraySimulator


[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] layers neutral-atom
occupancy, loss, and dynamic pairing onto the
[`Simulator`][fatqat.simulator.Simulator] execution model. It is useful for
checking whether a gate-level program respects those constraints. For pulse
timing, transport, or physical interactions, use the
[neutral-atom emulator](../atom-emulators.md) instead.

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


The quantum resources in each program define the array size, so the backend
needs no separate site-count or capacity setting. Registers map to integer
device labels in declaration order. A [`GridRegister`][fatqat.GridRegister] is
flattened because its coordinates have no physical meaning on this backend.

Only `RX`, `RY`, `RZ`, and `CZ` are native. The simulator does not decompose
other gates, so it rejects `CX` even when the atoms are paired.
[`AtomArraySimulator.implementation_map`][fatqat.simulator.AtomArraySimulator.implementation_map]
exposes this gate set, while the current `Pair` graph determines where `CZ`
is legal.

Native-gate support and `CZ` pairing are checked before occupancy is
considered. An unsupported gate or unpaired `CZ` is therefore rejected even
when one of its sites is empty; only a supported, correctly-paired gate can
become an empty-site no-op.

Connectivity evolves as the program runs. An unconditional
[`fatqat.operations.Pair`][fatqat.operations.Pair] connects two sites and
[`fatqat.operations.Unpair`][fatqat.operations.Unpair] disconnects them.
Neither operation changes the quantum state, although attached noise still
applies. Both must be unconditional.

## Occupancy and loss


Occupancy is tracked independently for each shot, and every declared site
starts empty whether or not the program contains `Put` or has a matching loss
source.
[`fatqat.operations.Put`][fatqat.operations.Put] loads a fresh `\|0>` atom into
an empty target and leaves an occupied target unchanged. Until a site is
loaded—or after it is lost—supported gates and reset have no effect there,
while measurement reports erasure digit `2`. A later `Put` can refill the
site, and pairing instructions continue to update connectivity throughout.

Occupancy is tracked outside the quantum-state representation. Final
statevectors, density matrices, and [`Estimator`][fatqat.Estimator]
calculations still include one qubit subsystem per declared site, whether or
not an atom is present. With the default initial state, a never-loaded site is
represented by `|0>`: measuring it returns `2`, while evaluating `Z` returns
`+1`. These quantum-state exports cannot reveal whether a site is occupied;
use measurements and counts for that distinction.

[`fatqat.noise.Loss`][fatqat.noise.Loss] can eject gate targets, make `Put` fail, or model
loss during `Pair` and `Unpair`. Its selector runs only after a matching
operation; `p=0` removes nothing and leaves the loading rule unchanged. No
other gate-level simulator accepts `Loss`. If an otherwise-valid paired `CZ`
finds a missing atom, it does nothing for that shot; an unpaired `CZ` still
fails before execution.

Because an empty site produces no physical readout digit, its erasure value
`2` bypasses readout-confusion noise. Atom loss also makes the final state
stochastic, so `final_state=True` requires `shots == 1`. A measured lossy run
returns counts by default rather than one arbitrary trajectory's final state.

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

Because occupancy sits outside the quantum matrix, `unitary` and `superop`
cannot represent this lifecycle. `AtomArraySimulator` accepts only
`statevector` and `density_matrix`, rejecting either operator method when the
backend is constructed—even for a program with no `Put` or loss.

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
