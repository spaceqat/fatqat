# Core concepts

A fatqat program has seven user-facing pieces. Each has one job:

| Concept | What you use it for |
| --- | --- |
| {py:class}`~fatqat.Program` | Record gates, measurements, and their order. |
| registers and references | Name the quantum and classical slots that operations use. |
| {py:class}`~fatqat.ResourceLayout` | Optionally map program quantum references to backend device operands for one run. |
| operations | Describe a gate or reset before it is added to a program. |
| backend | Validate and execute a program. |
| {py:class}`~fatqat.Job` | Represent one submitted run; call ``result()`` to obtain its output. |
| {py:class}`~fatqat.Result` | Read the data requested from a run. |

## Program

{py:class}`~fatqat.Program` is the frontend object you build. It owns the program’s
registers and the ordered instructions you add. It does not execute those
instructions itself.

```python
import fatqat as fq
import fatqat.operations as op

program = fq.Program(2, 2)
program.add(op.H, 0)
program.add(op.CX, (0, 1))
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
import fatqat.operations as op

left = fq.QuantumRegister(2, name="left")
right = fq.QuantumRegister(2, name="right")
program = fq.Program([left, right])
program.add(op.H, program.quantum_registers[1][0])  # first slot in "right"
```

Slots default to dimension 2 (qubits). Registers with `dim > 2` hold
qudits; see [Advanced user topics](advanced.md) when you need them.

## Operations and measurements

An operation says what should happen; {py:meth}`~fatqat.Program.add` binds it to target
slots. Fixed gates are values such as `op.X`; parametric gates are
created with their parameter, such as `op.RX(0.2)`.

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

The backend handles validation and execution. ``run()`` returns a {py:class}`~fatqat.Job`, and
``job.result()`` gives its {py:class}`~fatqat.Result`. The normal user-level Job contract is
intentionally small: obtain the result, then read it. Applications do not
configure the simulator engine or its private execution state.

Backends provide a default {py:class}`~fatqat.ResourceLayout`. Pass an explicit
one to ``run(resource_layout=...)`` only when you need a different legal
program-to-device binding, for example mapping two program qubits to selected
transmons in a larger model. The layout contains public device identities; it
must cover every declared quantum reference and does not expose or choose the
numerical tensor axes owned by the backend.

## Optional grid registers

{py:class}`~fatqat.GridRegister` is useful when your program’s qubits have a rectangular
layout. Its {py:meth}`~fatqat.GridRegister.row`, {py:meth}`~fatqat.GridRegister.column`, {py:meth}`~fatqat.GridRegister.block`, and {py:meth}`~fatqat.GridRegister.all` helpers return
targets accepted by selected rotation and two-qubit gates:

```python
qubits = fq.GridRegister(2, 3, name="qubits")
program = fq.Program([qubits])
program.add(op.RX(0.2), qubits.row(1))
```

The grid is an abstract description. A backend applies its device-specific
default mapping and connectivity checks when it runs the program. A backend
may deliberately constrain or reject a supplied layout when its workflow
owns placement.
