# Running and results

## Running a program

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))
program.measure((0, 1), (0, 1))

backend = fq.simulator.Simulator("SV")
result = backend.run(program, shots=1000).result()
```

{py:meth}`~fatqat.simulator.Simulator.run` returns a
{py:class}`~fatqat.Job` immediately. Phase 1 jobs are already terminal, so
{py:meth}`~fatqat.Job.result` either returns a {py:class}`~fatqat.Result` or
re-raises the exception from a failed run. `run()` itself still raises
directly for validation failures, such as an unsupported operation or a
configuration incompatible with the requested result fields.

Key arguments:

- `shots`: how many sampled executions to run when counts are requested. Ignored
  for a purely deterministic final-state request.
- `resource_layout`: an optional {py:class}`~fatqat.ResourceLayout` mapping
  every declared program quantum reference to a backend device operand. When
  omitted, the backend applies its documented default. This never selects
  numerical tensor indices.
- `simulation_config`: a plain dict for simulator-only controls: `seed`,
  `shot_parallelism`, `kernel_parallelism`, `max_workers`, and `fusion`.
  Both parallelism axes default to automatic selection; fusion defaults off. See
  [Advanced](advanced.md) for execution-mode semantics and the exact seed
  reproducibility contract.
- `result_config`: a plain dict selecting which result artifacts to produce.

## Choosing result fields

`result_config` accepts `counts` and `final_state`, each `True`, `False`, or
omitted for the backend default:

- `counts` defaults to `True` when the program has at least one measurement,
  `False` otherwise.
- `final_state` defaults to `True` only when the program is non-stochastic.
  Its concrete representation follows the backend's `method`: a statevector,
  density matrix, unitary, or super-operator.
- Requesting `final_state=True` for a stochastic program is only valid for
  `shots == 1`, since the result contains one trajectory's final state.

The program above is stochastic, so asking it for a final state needs
`shots=1`. Dropping its measurements makes it deterministic instead:

```python
bell = fq.Program(2)
bell.add(ops.H, 0)
bell.add(ops.CX, (0, 1))

state = (
    backend.run(bell, result_config={"counts": False, "final_state": True})
    .result()
    .get_statevector()
)
```

## Computing the program's map

`method="unitary"` and `method="superop"` return the program's operator
instead of a state under it. Both run one deterministic pass — no shots, no
sampling — so `final_state` defaults to `True` and `shots` is ignored:

```python
unitary = fq.simulator.Simulator("unitary").run(bell).result().get_unitary()
superop = fq.simulator.Simulator("superop").run(bell).result().get_superop()
```

`unitary[:, 0]` is the statevector the same program prepares. The public
super-operator uses column-stacking vectorization of density matrices:

```python
rho_out = (
    superop @ rho_in.reshape(-1, order="F")
).reshape(rho_in.shape, order="F")
```

Here, column-stacking describes the mathematical vectorization of `rho_in`,
not the NumPy memory layout of the returned matrix. For a noise-free program,
`superop` equals `numpy.kron(unitary.conj(), unitary)`.

**Migration note.** FATQAT previously returned row-stacked public
super-operator matrices. Raw matrices therefore change numerically for the
same channel; code that flattens or reshapes density matrices must now use
`order="F"`. The represented channel itself is unchanged.

Both methods reject anything an operator cannot express, before running:
measurement, feedforward conditions, and a `counts` request on either;
{py:data}`~fatqat.operations.Reset` and channel noise on `unitary` only —
`superop` applies both as exact channels. Memory grows as `4**n` for
`unitary` and `16**n` for `superop`, so keep `superop` circuits small.

## Reading a Result

The accessors below are a reference listing, not a runnable script — each
one is only valid when the run actually produced that field:

```python
result.get_counts()             # {"00": 512, "11": 488}, little-endian keys
result.get_statevector()        # numpy array, if produced
result.get_data("samples")      # backend-specific artifact, if produced
result.available_data           # frozenset of field names actually present
result.metadata                 # shots, backend name, normalized requests
```

Metadata retains the normalized requested ``simulation_config`` and
``result_config`` for provenance. It does not expose the private resolved shot
or kernel strategy, thread partition, or active worker count. Pulse-emulator
metadata also includes common solver facts, but does not duplicate model,
target, arrangement, or control-table data; retain those inputs separately or
attach application metadata when provenance is needed.

`final_state` is a request name, not an artifact name. When it is produced,
`available_data` contains the method-native name: `"statevector"`,
`"density_matrix"`, `"unitary"`, or `"superop"`, matching the backend's
`method`.

When a complete state or operator is returned, ``metadata["state_axes"]``
describes its physical subsystems in canonical order. Each entry contains a
public ``device_operand`` and a ``register_ref`` containing the program's actual
{py:class}`~fatqat.RegisterRef`, or ``None`` when that subsystem is not
addressed by the program. Because it preserves the public domain object,
``state_axes`` is not a JSON-serialization format. List position 0 is the
least-significant subsystem of a flat basis index; for local dimensions
``dims``, position ``q`` has place value ``prod(dims[:q])``. Density-matrix
rows and columns use the same basis order. These positions are not NumPy
reshape axes or solver-library tensor-factor numbers, and no private engine
index is published or accepted as a later selector. A physical model may
therefore report axes that have a device operand but no program reference.

Calling an accessor for a field that wasn't produced raises
{py:exc}`~fatqat.errors.ResultFieldUnavailableError` rather than returning
`None`; check `available_data` first if a field is optional in your workflow.

If a counts-only run contains a declared clbit that was never written by a
measurement, the backend still returns zero-filled counts for that bit but
emits a standard `UserWarning` — usually a sign a measurement was forgotten.

For qudits, custom matrix implementations, and execution strategies, see
[Advanced](advanced.md).
