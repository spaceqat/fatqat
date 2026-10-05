---
title: "SCTarget"
---

# SCTarget

Use [`SCTarget`][fatqat.compiler.SCTarget] to describe the superconducting
device that you compile a program for: how many sites it has and which pairs of
sites support `CZ`. Pass it to [`compile_to_sc()`][fatqat.compiler.compile_to_sc]
or [`compile_qasm_to_sc()`][fatqat.compiler.compile_qasm_to_sc] in place of a
simulator.

## Why compile against a target

A target contains only the information the compiler needs. It has no gate set,
runtime, simulation method, or noise model, and it cannot run a program. You
can therefore compile without constructing a simulator, and choose how to run
the result afterwards.

The compiled result is not tied to one simulator. An
[`SCQubitSimulator`][fatqat.simulator.SCQubitSimulator] can run it if it
provides the sites and couplings that the result uses, including a simulator
with more sites or additional couplings. The simulator's method must also
support the program. For example, `method="unitary"` rejects measurements.

You can still pass an `SCQubitSimulator` as the `backend` argument. In that
case, the compiler also applies the simulator's implementation map, both
when choosing couplings for routing and when checking each emitted gate. To
reuse the sites and couplings of an existing simulator, call
[`get_compiler_target()`][fatqat.simulator.SCQubitSimulator.get_compiler_target]:

```python
import fatqat as fq

simulator = fq.simulator.SCQubitSimulator(num_qubits=3, couplings=((0, 1), (1, 2)))
target = simulator.get_compiler_target()

assert target.num_qubits == 3
assert target.couplings == ((0, 1), (1, 2))
```

`get_compiler_target()` returns the site count and couplings the simulator was
constructed with. For a simulator created with its public constructor,
compiling against the simulator or against this target produces the same native
operations and layouts. The results can differ for a subclass that changes the implementation
map, for example by removing a native gate or allowing `CZ` in one direction
only.

## Construct a target

Both arguments are keyword-only and required. There is no default capacity or
topology.

- `num_qubits` is the number of device sites, labeled `0` to `num_qubits - 1`.
  It must be a Python `int` greater than zero. The constructor rejects `bool`,
  `float`, and NumPy integers and does not convert them.
- `couplings` is a finite iterable of undirected connections. Each element
  must be a two-element `tuple` of `int` site labels.

The constructor reads the couplings once. You can therefore pass a generator,
and later changes to the original container do not affect the target. The
target stores each edge as `(low, high)` and merges reversed and repeated edges
into their first occurrence:

```python
target = fq.compiler.SCTarget(num_qubits=3, couplings=[(1, 0), (1, 2), (0, 1)])

assert target.couplings == ((0, 1), (1, 2))
```

An empty `couplings` describes a device without `CZ` connections. It is a valid
target. Circuits without two-qubit gates compile. A circuit with a two-qubit
gate fails during compilation.

`num_qubits` and `couplings` are read-only. You can reuse a target for any
number of compilations. Equality, hashing, and serialization are not part of
the `SCTarget` interface.

## When errors are raised

| Stage | Condition | Error |
| --- | --- | --- |
| `SCTarget(...)` | A missing or positional argument, a non-`int` site count or site label, or a coupling that is not a two-element `tuple` | `TypeError` |
| `SCTarget(...)` | A site count below one, a site label outside `range(num_qubits)`, or a coupling between a site and itself | `ValueError` |
| Compilation | More program qubits than target sites, a two-qubit gate whose qubits the couplings cannot connect, or a `backend` that is neither an `SCTarget` nor an `SCQubitSimulator` | [`PassError`][fatqat.compiler.PassError] from the `lower-sc-to-native` stage. Its `__cause__` is the original `ValueError` or `TypeError` |
| `SCQubitSimulator.run()` | The result uses a site that the simulator does not have | [`BackendValidationError`][fatqat.errors.BackendValidationError] |
| `SCQubitSimulator.run()` | The result applies `CZ` to a pair that the simulator does not couple | [`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError] |

`run()` raises these errors itself, before it returns a job. The compiled
result does not keep a reference to its target. The simulator checks the
program against its own sites and couplings.

## Compiled output

Compiling for a target always produces gates from the canonical SC native
basis: `X`, `SX`, `RZ`, and `CZ`. The compiler keeps the measurements and
`Reset` operations of the source. Every operation uses a site within
`range(num_qubits)`, and every `CZ` acts on a target coupling. A target cannot
request a different native gate set. `SCQubitSimulator` executes this basis.

The convenience functions return an
[`ExecutableCompilationResult`][fatqat.compiler.ExecutableCompilationResult]
at the final boundary. A compiler created with
[`create_sc_pipeline()`][fatqat.compiler.create_sc_pipeline] accepts the same
target through [`CompileContext`][fatqat.compiler.CompileContext], but returns
a [`CompilationResult`][fatqat.compiler.CompilationResult] whose `output` is an
`SCNativeProgram`. Convert that output with
[`to_sc_simulator_program()`][fatqat.compiler.to_sc_simulator_program] before
running it.

## Compile and run

```python
circuit = fq.LogicalProgram(2, 2)
circuit.add(fq.operations.H, 0)
circuit.add(fq.operations.CX, (0, 1))
circuit.measure_all()

target = fq.compiler.SCTarget(num_qubits=3, couplings=((0, 1), (1, 2)))
compiled = fq.compiler.compile_to_sc(circuit, target, seed=7)

backend = fq.simulator.SCQubitSimulator(
    num_qubits=3,
    couplings=target.couplings,
    runtime="numpy",
)
counts = backend.run(compiled, shots=100).result().get_counts()
```

The same target works with a directly constructed compiler:

```python
from fatqat.compiler import (
    LOGICAL_SC_PIPELINE,
    CompileContext,
    create_sc_pipeline,
    to_sc_simulator_program,
)

result = create_sc_pipeline().compile(
    circuit,
    pipeline=LOGICAL_SC_PIPELINE,
    context=CompileContext(target=target, options={"seed": 7}),
)
program, layout = to_sc_simulator_program(result.output)
counts = backend.run(program, shots=100, resource_layout=layout).result().get_counts()
```

## Reference

::: fatqat.compiler.SCTarget
