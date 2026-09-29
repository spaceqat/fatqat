---
title: "LogicalProgram"
---

# LogicalProgram

Use [`LogicalProgram`][fatqat.LogicalProgram] to describe a circuit that FatQat
will compile for a hardware profile. It follows the familiar
[`Program`](program.md) interface, so the same circuit can also be simulated
directly.

```python
import fatqat as fq

circuit = fq.LogicalProgram(2, 2, metadata={"name": "bell"})
circuit.add(fq.operations.H, 0)
circuit.add(fq.operations.CX, (0, 1))
circuit.measure_all()
```

`LogicalProgram(2, 2)` creates two logical qubits and two classical bits.
`add()` appends gates, while `measure()` and `measure_all()` record measurement
results. These methods update the circuit in place.

## Compile or simulate the circuit

Compile the circuit by choosing a target family:

```python
sc_backend = fq.simulator.SCQubitSimulator()
sc_compiled = fq.compiler.compile_to_sc(circuit, sc_backend)

from fatqat.compiler.algorithms import load_architecture

architecture = load_architecture("default")
na_compiled = fq.compiler.compile_to_na(circuit, architecture)
```

For algorithm development, the same circuit can run directly on the general
simulator:

```python
result = fq.simulator.Simulator().run(circuit, shots=100).result()
```

See [Compile and run a logical circuit](../guide/compiler.md) for complete SC,
neutral-atom, and OpenQASM workflows.

## Reuse parameterized circuits

`copy()` returns a `LogicalProgram` with independent instructions and top-level
metadata. Values nested inside metadata remain shared. `assign_parameters()`
returns a new `LogicalProgram` with the selected values, leaving the original
template unchanged.

```python
theta = fq.Parameter("theta")

template = fq.LogicalProgram(1)
template.add(fq.operations.RY(theta), 0)

first = template.assign_parameters({theta: 0.25})
second = template.assign_parameters({theta: 0.75})
```

Bind all parameters to numeric values before compiling. Partial binding remains
useful while assembling a larger template.

## Choose between LogicalProgram and Program

`LogicalProgram` is for device-independent gates that can be lowered by the
compiler. Use [`Program`](program.md) when you are constructing physical
operations directly, such as atom placement, pairing, or pulse controls, or
when you are adding custom operation types.

Classical conditions are available when simulating a `LogicalProgram`
directly. The current compiler handles static circuits with terminal
measurements; the [compiler guide](../guide/compiler.md#current-compiler-support)
summarizes the supported inputs.

## Reference

::: fatqat.LogicalProgram
    options:
      inherited_members: false
      show_bases: true
