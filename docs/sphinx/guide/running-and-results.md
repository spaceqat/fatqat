# Running and results

## Running a program

```python
backend = qs.backends.StateVectorBackend()
result = backend.run(program, shots=1000).result()
```

{py:meth}`~qnsim.backends.StateVectorBackend.run` returns a
{py:class}`~qnsim.Job` immediately — Phase 1 jobs are already terminal, so
{py:meth}`~qnsim.Job.result` either returns a {py:class}`~qnsim.Result` or
re-raises the exception from a failed run. `run()` itself still raises
directly for validation failures (e.g. an unsupported operation, or `shots`
incompatible with the requested result fields).

Key arguments:

- `shots`: how many logical shots to run when counts are requested.
  Ignored for a purely deterministic statevector request.
- `result_config`: a plain dict selecting which fields to produce — see
  below.
- `seed`: a root seed. With a fixed seed, counts are reproducible
  regardless of whether shots ran serially or across worker processes.

## Choosing result fields

`result_config` accepts `counts` and `statevector`, each `True`, `False`,
or omitted for the backend default:

- `counts` defaults to `True` when the program has at least one
  measurement, `False` otherwise.
- `statevector` defaults to `True` only when the program is
  non-stochastic (no measurement, no reset) — a statevector is otherwise
  ambiguous, since each shot can end in a different state.
- Requesting `statevector=True` for a stochastic program is only valid for
  `shots == 1`, since only one shot's post-measurement state is returned.

```python
result = backend.run(
    program,
    result_config={"counts": False, "statevector": True},
).result()
```

## Reading a Result

```python
result.get_counts()             # {"00": 512, "11": 488}, little-endian keys
result.get_statevector()        # numpy array, if produced
result.available_data           # frozenset of field names actually present
result.metadata                 # shots, backend_name, effective result_config
```

Calling an accessor for a field that wasn't produced raises
{py:exc}`~qnsim.errors.ResultFieldUnavailableError` rather than returning
`None` — check `available_data` first if a field is optional in your
workflow.

If you request counts on a program where some declared clbit was never
written by a measurement, the backend still returns zero-filled counts for
that bit but raises a {py:exc}`~qnsim.errors.NoMeasurementWarning` —
usually a sign a measurement was forgotten.

## Advanced: qudits, custom implementations, parallel execution

Everything above applies equally to qudits (registers with `dim > 2`) —
{py:class}`~qnsim.operations.Shift`/{py:class}`~qnsim.operations.Clock`/
{py:data}`~qnsim.operations.Sum` generalize the qubit gates rather than
requiring separate handling.

{py:class}`~qnsim.backends.StateVectorBackend`'s `implementation_map=`
argument accepts a custom `MatrixImplementationMap` to control how
operations are lowered to matrices, or to add support for operations the
default map doesn't cover.

For programs on the dynamic path,
{py:class}`~qnsim.backends.StateVectorBackend`'s `options={...}` accepts
`max_workers` and `parallel_mode` (`"auto"`, `"serial"`,
`"multiprocessing"`, or `"loky"`) to control whether shots are distributed
across worker processes. This only affects execution strategy, never
numerical results.
