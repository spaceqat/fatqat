---
title: "LQCloud QEC17"
---

# LQCloud QEC17

Use [`LQCloudQEC17Backend`][fatqat.lqcloud.LQCloudQEC17Backend] to submit one
static, native-gate [`Program`][fatqat.Program] to LQCloud's
`QZ01-surface_code` processor. The optional adapter returns the LQCloud SDK's
asynchronous job unchanged. See [Run a Program on LQCloud hardware](../../guide/lqcloud.md)
for the complete workflow.

Install the supported optional SDK separately:

```sh
python -m pip install lqcloud==0.4.2
```

Importing `fatqat.lqcloud` does not require the SDK. Constructing the backend
or translating a program requires it and raises `ImportError` when it is missing.

```python
import os
import fatqat as fq
import fatqat.operations as ops
from fatqat.lqcloud import LQCloudQEC17Backend

program = fq.Program(1)
program.add(ops.H, 0)
backend = LQCloudQEC17Backend(api_key=os.environ["LQCLOUD_API_KEY"])
job = backend.run(program, shots=1024, initial_layout=[6])
```

The last line submits a hardware job. `api_key` must be a string containing
at least one non-whitespace character.
`url=None` uses the SDK's service URL; pass `url` to override it. The adapter
uses non-interactive authentication and selects its fixed backend name without
a discovery request. Authentication and service errors follow the SDK.

## Supported programs

A program must contain 1 through 17 qubits, all of dimension 2. The accepted
single-qubit operations are:

```text
I, H, HY,
X, MX, Y, MY, Z, MZ,
XHalf, MXHalf, YHalf, MYHalf,
XYHalf, MXYHalf, MXMYHalf, XMYHalf,
S, Sdg, T, RZ(theta), SU2(matrix)
```

`CZ` is the only supported two-qubit gate. Input `Barrier` operations are
preserved. See [Qubit gates](../operations/qubit-gates.md) for gate matrices.
`RZ` requires a bound, finite real angle; `SU2` carries a numeric 2 × 2 unitary.

The adapter translates each native operation to its SDK counterpart, then
appends a full-width barrier and `measure_all()` on the program's logical
qubits. Only an explicit FatQat `I` produces an SDK `id` instruction; the
adapter does not generate idle gates.

Unsupported operations raise `ValueError`, including `RX`, `RY`, `SX`, `CX`,
`Tdg`, user-authored measurements, explicit reset, and classically conditioned
operations. Initial state preparation belongs to LQCloud. There is no gate
decomposition, optimization, routing, or dynamic-circuit execution.

## Placement

Quantum registers are flattened in declaration order.
`initial_layout[k]` maps flattened logical qubit `k` to a physical QEC17 index.
It accepts a list or tuple with one distinct integer per program qubit, each
in `0..16`; Boolean values are not indices. `None` validates against the
identity layout and leaves the SDK's layout option unset.

Every mapped `CZ` must use one of these undirected edges:

```text
(6, 15), (16, 3), (3, 11), (11, 7),
(0, 9), (9, 4), (4, 12), (12, 8),
(1, 10), (10, 5), (5, 14), (13, 2),
(6, 11), (15, 7), (16, 0), (3, 9),
(11, 4), (7, 12), (9, 1), (4, 10),
(12, 5), (8, 14), (1, 13), (10, 2)
```

Invalid layout lengths, indices, duplicates, or mapped edges raise `ValueError`.
Passing a layout other than a list, tuple, or `None` raises `TypeError`.

## Submission and results

`shots` defaults to `1024` and must be an integer from 1 through 50,000.
Boolean and other non-`int` values raise `TypeError`; out-of-range values raise
`ValueError`. Additional `run()` keyword arguments are passed unchanged to
the SDK, which defines their accepted values and validation. For example,
`readout_correction` and `result_format` are SDK options. Omitting
`result_format` preserves the SDK's default selection.

The returned job's `status()`, `cancel()`, and `result()` methods keep their
LQCloud behavior. Results, counts, and bitstring ordering also follow LQCloud;
they are not converted into FatQat [`Job`][fatqat.Job] or
[`Result`][fatqat.Result] objects.

## Translate without submitting

[`program_to_qec17_circuit`][fatqat.lqcloud.program_to_qec17_circuit] returns an
`lqcloud.QuantumCircuit` with the same native-gate translation and terminal
measurement used by the backend. It needs the SDK but no API key or provider
connection. It validates the program's shape and language; physical layout
and `CZ` adjacency are checked separately by `run()`.

```python
from fatqat.lqcloud import program_to_qec17_circuit

circuit = program_to_qec17_circuit(program)
print(circuit.draw())
```

## Reference

::: fatqat.lqcloud.LQCloudQEC17Backend
    options:
      inherited_members: false
      show_bases: false
      merge_init_into_class: true
      filters:
        - "!^_"

::: fatqat.lqcloud.program_to_qec17_circuit
    options:
      show_bases: false
      filters:
        - "!^_"
