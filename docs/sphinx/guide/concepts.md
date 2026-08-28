# Core concepts

A typical FATQAT workflow builds a program, submits it to a backend, and reads
the result. Registers, operations, and an optional resource layout describe
the workload and where it should run:

| Concept | What you use it for |
| --- | --- |
| {py:class}`~fatqat.Program` | Record gates, measurements, and their order. |
| registers and references | Name the quantum and classical slots that operations use. |
| {py:class}`~fatqat.ResourceLayout` | Optionally map program quantum references to backend device labels for one run. |
| operations | Describe a gate or another instruction before it is added to a program. |
| backend | Validate and execute a program. |
| {py:class}`~fatqat.Job` | Represent one submitted run; call ``result()`` to obtain its output. |
| {py:class}`~fatqat.Result` | Read the data requested from a run. |

## Program

{py:class}`~fatqat.Program` records registers and instructions in execution
order. It describes the workload without choosing or running a backend.

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))
program.measure((0, 1), (0, 1))
```

Instructions run in the order above. A measurement is also an instruction,
so it can appear before a later conditional operation.

## Registers and references

For the common case, `Program(quantum_count, classical_count)` creates one
quantum register and one classical register. A bare integer such as `0` is
then a convenient reference to one quantum or classical slot.

Use an explicit {py:class}`~fatqat.RegisterRef` when a program has more than one register.
Indexing a register produces that reference:

```python
import fatqat as fq
import fatqat.operations as ops

left = fq.QuantumRegister(2, name="left")
right = fq.QuantumRegister(2, name="right")
program = fq.Program([left, right])
program.add(ops.H, right[0])
```

Slots default to dimension 2 (qubits). Registers with `dim > 2` hold
qudits; see [Advanced user topics](advanced.md) when you need them.

## Operations and measurements

An operation says what should happen; {py:meth}`~fatqat.Program.add` binds it to target
slots. Fixed gates are values such as `ops.X`; parametric gates are
created with their parameter, such as `ops.RX(0.2)`.

Use {py:meth}`~fatqat.Program.measure` to write quantum outcomes into classical slots.
Use [Measurement and conditions](measurement-and-conditions.md) for
grouped measurement, reset, and feedforward.

## Backends and results

A {py:class}`~fatqat.simulator.Simulator` is the object that executes a program:

```python
backend = fq.simulator.Simulator()
job = backend.run(program, shots=1000)
result = job.result()
counts = result.get_counts()
```

The backend validates and executes the program. ``run()`` returns a completed
{py:class}`~fatqat.Job`; ``job.result()`` returns its
{py:class}`~fatqat.Result` or re-raises a stored execution error.

Backends provide a default {py:class}`~fatqat.ResourceLayout`. Pass one to
``run(resource_layout=...)`` when you need a different supported placement,
for example to map two program qubits to selected transmons in a larger model.
A supplied layout must cover every declared quantum reference with labels the
backend recognizes.

Execution options and supported program features are backend-specific. Use
the {doc}`../api/index` capability comparison and the selected backend's API
page for its complete ``run()`` contract.

## Optional grid registers

{py:class}`~fatqat.GridRegister` is useful when your program’s qubits have a rectangular
layout. Its {py:meth}`~fatqat.GridRegister.row`, {py:meth}`~fatqat.GridRegister.column`, {py:meth}`~fatqat.GridRegister.block`, and {py:meth}`~fatqat.GridRegister.all` helpers return
targets accepted by selected rotation and two-qubit gates:

```python
qubits = fq.GridRegister(2, 3, name="qubits")
program = fq.Program([qubits])
program.add(ops.RX(0.2), qubits.row(1))
```

The grid describes logical grouping, not physical placement. The backend maps
it to device resources and checks connectivity when the program runs. Some
backends choose placement themselves and do not accept a custom layout.
