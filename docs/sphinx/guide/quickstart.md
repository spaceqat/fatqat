# Quickstart

This page uses the currently supported source-checkout installation route.
Run the commands from the repository root.

## Install

Creating a virtual environment keeps the project dependencies separate from
other Python projects:

```bash
python -m venv .venv
# Activate .venv with the normal command for your platform.
python -m pip install --upgrade pip
python -m pip install -e .
```

## Build and run a Bell-state program

Copy this complete example into a Python file or interpreter:

```python
import fatqat as fq

program = fq.Program(2, 2)
program.add(fq.ops.H, 0)
program.add(fq.ops.CX, (0, 1))
program.add_measurement((0, 1), (0, 1))

backend = fq.backends.SimulatorBackend()
job = backend.run(program, shots=1000)
result = job.result()
print(result.get_counts())
```

{py:data}`~fatqat.operations.H` puts qubit 0 into a superposition. {py:data}`~fatqat.operations.CX` entangles qubit 1 with it,
so measuring the pair produces either `00` or `11`. A typical result is
`{"00": 512, "11": 488}`; the two numbers naturally vary because
measurement is sampled 1,000 times.

## What each line does

| Part | Meaning |
| --- | --- |
| {py:class}`~fatqat.Program` (``Program(2, 2)``) | Create two quantum slots and two classical slots. |
| `program.add(...)` | Append a gate in execution order. Fixed gates such as `H` and `CX` are values, so they do not have parentheses. |
| `add_measurement(...)` | Copy each quantum outcome into the matching classical slot. |
| {py:class}`~fatqat.backends.SimulatorBackend` (``SimulatorBackend()``) | Use the general-purpose simulator. You do not construct or call an engine directly. |
| `shots=1000` | Repeat the measured program 1,000 times to collect counts. |
| `backend.run(...)` | Submit the program and receive a `Job`. |
| ``job.result()`` | Obtain the completed {py:class}`~fatqat.Result`, then use an accessor such as ``get_counts()``. |

For ordinary use, a `Job` is the hand-off between execution and results:
call `result()` and work with the returned `Result`. Its lifecycle/status
methods are separately documented as an experimental API.

The displayed count strings put classical bit 0 at the right. The
[results guide](running-and-results.md) gives a two-bit example and explains
how to request state data.

## Next steps

- Learn fixed versus parametric gates in [Gates](gates.md).
- Add mid-program measurement, reset, or classical conditions in
  [Measurement and conditions](measurement-and-conditions.md).
- Choose counts, a statevector, or a density matrix in
  [Running and results](running-and-results.md).
