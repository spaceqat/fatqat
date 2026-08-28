# Measurement and conditions

Measurements sample quantum targets in the computational basis and write the
outcomes to classical slots. They are ordered instructions, so a program can
use an earlier outcome to control a later operation.

## Measure a qubit

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(1, 1)
program.add(ops.X, 0)
program.measure(0, 0)

result = fq.simulator.Simulator().run(program, shots=10).result()
print(result.get_counts())  # {"1": 10}
```

{py:meth}`~fatqat.Program.measure` (``quantum_target, classical_output``) accepts one target or
matching tuples of targets. The tuple order determines which quantum outcome
is written to which classical slot:

```python
# Assumes a Program with two qubits and two classical bits.
program.measure((0, 1), (0, 1))
```

Use {py:meth}`~fatqat.Program.measure_all` when the program has the same number of quantum
and classical slots and you want every quantum slot measured in declaration
order.

## Classical conditions

Pass `condition=(classical_bit, value)` to `add()` to apply an operation only
when the slot currently contains that value. Classical slots start at zero;
measurements replace their current values. A sequence of
`(classical_bit, value)` pairs means all conditions must hold.

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 1)
program.add(ops.H, 0)
program.measure(0, 0)
program.add(ops.X, 1, condition=(0, 1))
```

In this example, qubit 1 is flipped only on shots where measuring qubit 0
produced `1`.

## Reset

{py:data}`~fatqat.operations.Reset` prepares its target in ``|0⟩``. It is useful after a
measurement or whenever a program needs to reuse a qubit:

```python
program.add(ops.Reset, 0)
program.add(ops.Reset, (0, 1))  # reset two qubits
```

Reset can use the same `condition=` argument on backends that support
feedforward.

## What this means for a run

Conditions, reset, and reuse of a measured qubit make later steps depend on an
earlier outcome. A backend either preserves that ordering or rejects the
program when it does not support feedforward. See the selected backend's API
page for supported operations and execution options.

See [Running and results](running-and-results.md) for count-string order and
for the limits on requesting a final state after a stochastic program.
