# Measurement and conditions

## Adding measurements

`add_measurement(qreg, clreg)` reads one or more qubits into matching
classical bits, in the order given:

```python
program = qs.Program(1, 1)
program.add(qs.ops.X, 0)
program.add_measurement(0, 0)
```

Grouped measurement reads several qubits into several clbits in one call —
`qreg` and `clreg` must have the same length:

```python
program.add_measurement((0, 1), (0, 1))
```

`measure_all()` is shorthand for measuring every qubit into every clbit, in
flat declaration order; it requires equal quantum and classical bit counts:

```python
program = qs.Program(2, 2)
program.add(qs.ops.H, 0)
program.add(qs.ops.CX, (0, 1))
program.measure_all()   # equivalent to add_measurement((0, 1), (0, 1))
```

## Feedforward conditions

`program.add(op, targets, condition=(clbit, value))` makes an operation
conditional on a classical bit already written earlier in the program — the
operation only applies if that bit equals `value` when execution reaches
it. Pass a sequence of `(clbit, value)` pairs to AND several conditions
together:

```python
program = qs.Program(2, 1)
program.add(qs.ops.H, 0)
program.add_measurement(0, 0)
program.add(qs.ops.X, 1, condition=(0, 1))   # flip qubit 1 iff clbit 0 == 1
```

## Reset

`qs.ops.Reset` reprepares one or more target subsystems in `|0>`. Unlike
every other operation in `qs.ops`, it isn't a unitary gate — it has no
matrix — so it can only be run on backends that special-case it (the
statevector backend does):

```python
program.add(qs.ops.Reset, 0)
```

`Reset` accepts more than one target at once, resetting every listed qubit:

```python
program.add(qs.ops.Reset, (0, 1))   # reset qubits 0 and 1 together
```

Like any operation, `Reset` can carry a `condition=` guard.

## What makes a program "dynamic"

Most programs — no reset, no conditions, no gate reapplied to an
already-measured qubit — are evaluated in one pass: the state is evolved
once, and any requested counts are sampled from the resulting distribution
afterward. A program becomes **dynamic** the moment it contains a
`condition=`, a `Reset`, or an operation that targets a qubit measurement
already wrote to. Dynamic programs are executed shot by shot, because a
later step's behavior can depend on an earlier measurement's outcome within
that same shot.

You don't choose the execution strategy — the backend picks it
automatically based on the program's shape — but it's worth knowing about,
because it's what makes conditions and reset behave correctly (each shot
gets its own classical-bit history) instead of just averaging over
independent evolutions.
