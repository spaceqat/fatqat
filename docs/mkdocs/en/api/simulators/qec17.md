---
title: "QEC17 simulator"
---

# QEC17 simulator

[`SCQubitQEC17Simulator`][fatqat.simulator.SCQubitQEC17Simulator] executes
native gate circuits on the fixed 17-qubit QZ01 coupling graph. Use it to
compare placements and study gate, readout, and idle errors using individual
qubit and coupler calibrations. It follows the ordinary
[`Simulator`](../simulator.md) run, result, method, and runtime rules.
The default method is `statevector`, and the default runtime is `numba`;
`numpy` also supports the noise model.

The backend is ideal by default. Enable its calibration noise explicitly:

```python
import fatqat as fq
import fatqat.operations as ops

Sim = fq.simulator.SCQubitQEC17Simulator
backend = Sim(noise=Sim.default_noise_model())

program = fq.Program(2, 2)
program.add(ops.H, 0)
program.add(ops.H, 1)
program.add(ops.CZ, (0, 1))
program.measure_all()
qubits = program.quantum_registers[0]
layout = fq.ResourceLayout({qubits[0]: 0, qubits[1]: 9})
counts = backend.run(
    program, resource_layout=layout, shots=1000,
    simulation_config={"seed": 17},
).result().get_counts()
```

## Native operations

| Family | Operations | Modeled duration |
| --- | --- | --- |
| Identity | `I` | 20 ns |
| Driven one-qubit gates | `H`, `HY`, `X`, `MX`, `Y`, `MY`, `XHalf`, `MXHalf`, `YHalf`, `MYHalf`, `XYHalf`, `MXYHalf`, `MXMYHalf`, `XMYHalf`, `SU2(matrix)` | 20 ns |
| Virtual Z | `Z`, `MZ`, `S`, `Sdg`, `T`, `RZ(theta)` | 0 ns |
| Entangling | `CZ` on a device coupling | 40 ns |
| Measurement and Reset | Ordinary Simulator operations | 0 ns; full-program synchronization |

See [Qubit gates](../operations/qubit-gates.md) for the matrix definitions.
Each native `SU2` is modeled as one driven instruction, including matrices
that happen to be diagonal. `RX`, `RY`, `SX`, `Tdg`, and `CX` are not native
on this profile. They are rejected rather than decomposed automatically.
The existing SC compiler targets the separate `SCQubitSimulator` profile;
it does not compile to QEC17.

## Physical labels and couplings

Physical labels `0` through `8` identify data qubits and `9` through `16`
identify syndrome qubits. The 24 undirected CZ couplings are:

| Data qubit | Connected syndrome qubits |
| --- | --- |
| 0 | 9, 16 |
| 1 | 9, 10, 13 |
| 2 | 10, 13 |
| 3 | 9, 11, 16 |
| 4 | 9, 10, 11, 12 |
| 5 | 10, 12, 14 |
| 6 | 11, 15 |
| 7 | 11, 12, 15 |
| 8 | 12, 14 |

Both operand orders of each edge are accepted. Inspect `device_sites` and
`implementation_map.device_operands_for(ops.CZ)` for machine-readable values.
The backend validates the program as written and performs no routing.

A program may declare at most 17 dimension-2 qubits. With no explicit
[`ResourceLayout`][fatqat.ResourceLayout], registers flatten in declaration
order onto physical labels `0, 1, ...`. `GridRegister` follows the same rule:
its frontend geometry does not define hardware connectivity. An explicit
layout must map every declared qubit to a distinct physical label in `0..16`.

Declare only the logical qubits needed by the computation, then map them to
the desired physical sites. Unselected device qubits are outside the simulated
state and idle schedule. To model spectators, declare and map them explicitly.

## Individual calibration noise

The built-in model uses the fixed LQCloud QZ01 snapshot exported on
**2026-09-12**. It contains 17 separate qubit records and 24 separate coupler
records. It uses no chip-average noise parameters. Changing a physical layout
changes which calibration values apply.

| Operation | Noise applied |
| --- | --- |
| Driven one-qubit gate | Depolarizing noise from the selected physical qubit's XEB fidelity |
| CZ | Joint two-qubit depolarizing noise from the selected physical edge's fidelity |
| Explicit or scheduled I | Amplitude and phase damping from that physical qubit's T1 and T2E over 20 ns |
| Virtual Z | None in the built-in model |
| Measurement | That physical qubit's F00/F11 readout confusion |

For example, physical qubits 0 and 4 use XEB fidelities `0.99948` and
`0.99900`; edges `(0, 9)` and `(1, 10)` use CZ fidelities `0.99601` and
`0.99176`. Both orders of a CZ edge use the same calibration.

Reported XEB and CZ fidelities are **interpreted as average gate fidelities**
for an effective depolarizing channel, with
`p = d * (1 - F) / (d - 1)` for dimension `d = 2` or `4`.
The model does not add separate active-gate relaxation on top of these
fidelity-derived errors. Each driven gate on a given qubit shares that
qubit's reported XEB value; pulse-specific errors are not available in the
snapshot.

Idle coherence uses T2E, not T2*. The residual pure-dephasing rate is
`1/T2E - 1/(2*T1)`. Readout matrices have reported values as rows and true
values as columns: `[[F00, 1-F11], [1-F00, F11]]`.
There is no correlated readout, leakage, or crosstalk model.

Every `default_noise_model()` call returns an independent `NoiseModel`,
and the backend copies the supplied model at construction. Normal noise
selector and overlap rules apply when authoring a custom model.

## Idle scheduling and limits

When the supplied model contains noise attached to `I`, QEC17 builds an
ASAP schedule in 20 ns ticks. Operations sharing a qubit retain source order;
independent operations may overlap. A gate starts when all its operands are
available. Waiting qubits receive enough internal `I` ticks to account for
the gap, and any matching `I` noise uses the usual physical or logical
selectors. An explicit `I` already occupies one tick and is not counted twice.
The input Program is never modified.

`Barrier` synchronizes only its targets. Measurement and Reset synchronize
every declared qubit, and the end of the program synchronizes all declared
qubits again. These boundaries add no measurement, reset, or feedback latency.
This is a fixed-duration gate model, not a pulse-resolved timing model.

An executable classical condition raises `BackendValidationError` when idle
scheduling is enabled; unknown operation durations also raise this error.
Idle noise is never silently dropped to execute such a program. Conditions
retain ordinary Simulator behavior on ideal instances or custom models
without `I`-scoped noise. A Barrier's recorded condition is always ignored.

## API

::: fatqat.simulator.SCQubitQEC17Simulator
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: true
      filters:
        - "!^_"
