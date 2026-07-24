# Running and results

The normal execution path is:

{py:class}`~fatqat.Program` → ``backend.run(...)`` → {py:class}`~fatqat.Job`
→ ``job.result()`` → {py:class}`~fatqat.Result` → ``get_*()``

A backend validates a program and returns a {py:class}`~fatqat.Job`. For ordinary use, call
``job.result()`` to obtain the {py:class}`~fatqat.Result` whose accessors return the fields you
requested. Job status and failure lifecycle controls are documented as an
experimental API rather than part of the normal workflow.

```python
backend = fq.backends.SimulatorBackend()
job = backend.run(program, shots=1000)
result = job.result()
```

``shots`` is the number of repetitions used to produce counts. Set ``seed``
when you want a sampled run to be reproducible.

## Choose output fields

Use ``result_config`` to make the output explicit. The useful fields depend
on the backend method:

| You need | Backend and request | Read it with |
| --- | --- | --- |
| sampled measurement counts | any simulator backend; `{"counts": True}` | `get_counts()` |
| a pure statevector | {py:class}`~fatqat.backends.SimulatorBackend`; ``{"counts": False, "statevector": True}`` | ``get_statevector()`` |
| an exact density matrix | {py:class}`~fatqat.backends.SimulatorBackend` (``method="density_matrix"``); ``{"counts": False, "density_matrix": True}`` | ``get_density_matrix()`` |

When `result_config` is omitted, counts are normally produced for a program
with measurements. An unmeasured, non-stochastic statevector program
normally produces its statevector. Make requests explicit in reusable
programs so the expected result is clear.

### Counts

```python
import fatqat as fq

program = fq.Program(1, 1)
program.add(fq.ops.X, 0)
program.add_measurement(0, 0)

result = fq.backends.SimulatorBackend().run(
    program,
    shots=100,
    result_config={"counts": True},
    seed=7,
).result()
print(result.get_counts())  # {"1": 100}
```

### Statevector

```python
import fatqat as fq

program = fq.Program(1)
program.add(fq.ops.H, 0)

result = fq.backends.SimulatorBackend().run(
    program,
    result_config={"counts": False, "statevector": True},
).result()
print(result.get_statevector())
```

A measurement, reset, or statevector noise can make the final state
stochastic. If you ask for a statevector in that situation, use `shots=1`
so that the result represents one shot.

### Density matrix

```python
import fatqat as fq

program = fq.Program(1)
program.add(fq.ops.H, 0)

backend = fq.backends.SimulatorBackend(method="density_matrix")
result = backend.run(
    program,
    result_config={"counts": False, "density_matrix": True},
).result()
print(result.get_density_matrix())
```

Use the density-matrix method when you need an exact mixed-state result,
such as a noisy distribution. It uses more memory than a statevector.

## Read count strings

{py:meth}`~fatqat.Result.get_counts` displays classical bits in little-endian order: classical
bit 0 is the character on the right. With two classical bits, a count key
of `"01"` means clbit 1 is `0` and clbit 0 is `1`.

Use {py:meth}`~fatqat.Result.get_counts_as_tuples` when a tuple in increasing classical-bit order
is clearer. The same `"01"` outcome is `(1, 0)`: first clbit 0, then
clbit 1.

## Check what is available

{py:attr}`~fatqat.Result.available_data` lists the fields the backend actually produced, and
{py:attr}`~fatqat.Result.metadata` records the effective request and run context. An accessor
such as {py:meth}`~fatqat.Result.get_statevector` raises {py:class}`~fatqat.errors.ResultFieldUnavailableError` rather
than returning an empty value when that field was not produced.

See [Troubleshooting](troubleshooting.md) if a requested result field is
unavailable.
