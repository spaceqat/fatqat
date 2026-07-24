# Advanced user topics

These features are optional. The normal {py:class}`~fatqat.Program` → backend → {py:class}`~fatqat.Result`
workflow does not change, and you still do not interact with a simulator
engine directly.

## Qudits

A register with `dim > 2` holds qudits. Qudit gates use the same
{py:meth}`~fatqat.Program.add` interface as qubit gates. This example combines a qubit and
a qutrit in one program:

```python
import fatqat as fq

qubit = fq.QuantumRegister(1, name="qubit")
qutrit = fq.QuantumRegister(1, dim=3, name="qutrit")
cbit = fq.ClassicalRegister(1, name="c_qubit")
ctrit = fq.ClassicalRegister(1, dim=3, name="c_qutrit")

program = fq.Program([qubit, qutrit], [cbit, ctrit])
program.add(fq.ops.X, program.qreg[0][0])
program.add(fq.ops.Shift(1), program.qreg[1][0])
program.add_measurement(program.qreg[0][0], program.clreg[0][0])
program.add_measurement(program.qreg[1][0], program.clreg[1][0])

result = fq.backends.SimulatorBackend().run(program, shots=100).result()
print(result.get_counts_as_tuples())  # {(1, 1): 100}
```

The qubit and qutrit keep their own dimensions. There is not yet a mixed
qubit–qutrit gate, so drive each register with an operation that supports
its dimension.

## Grid registers

Use {py:class}`~fatqat.GridRegister` when a logical rectangular layout makes a program easier
to write. The [Gates](gates.md) guide shows how row, column, block, and
all-grid selections expand for supported operations. A grid is a logical
program feature; the backend validates hardware-like connectivity when that
matters.

## Parallel shot execution

Programs with conditions, reset, or reuse after measurement may require
per-shot work. {py:class}`~fatqat.backends.SimulatorBackend` chooses an execution strategy automatically.
Most applications should keep that default.

For a deliberately serial, easier-to-debug run, pass an option when you
construct the backend:

```python
backend = fq.backends.SimulatorBackend(
    options={"parallel_mode": "serial"},
)
```

The documented `max_workers` and `parallel_mode` options affect performance,
not the simulated result. Pair them with a fixed `seed` when you need
reproducible sampled counts.

## Extending fatqat

Writing a backend, changing lowering rules, or replacing simulator machinery
is library-development work rather than the normal application workflow.
Those implementation details are intentionally not presented as a public
beginner API. Build programs against the supported {py:class}`~fatqat.Program`,
``fq.ops``, backend, {py:class}`~fatqat.NoiseModel`, and
{py:class}`~fatqat.Result` surfaces described in this guide.
