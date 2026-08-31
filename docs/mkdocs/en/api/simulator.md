---
title: "Simulator"
---

# Simulator


[`Simulator`][fatqat.simulator.Simulator] runs gate-level programs with matrix operations and finite
noise channels. It supports qubits, qudits, mixed register dimensions, and
custom implementation maps. It runs the program as written; it does not
transpile or route it.

## Quick start


```python
import fatqat as fq
import fatqat.operations as ops

bell = fq.Program(2, 2)
bell.add(ops.H, 0)
bell.add(ops.CX, (0, 1))
bell.measure_all()

backend = fq.simulator.Simulator(method="statevector")
counts = backend.run(bell, shots=1000).result().get_counts()
```

The default implementation map covers FATQAT's built-in matrix gates.
State methods also support measurement, reset, and classical conditions.
`Barrier` has no numerical effect.

## Methods


Method names are case-insensitive. `SV` and `DM` are aliases; the
read-only [`Simulator.method`][fatqat.simulator.Simulator.method] property returns the full name. If the
program's Hilbert-space dimension is `D`:

**Simulation methods**

| Method | Result | Reset and finite channels | Restrictions |
| --- | --- | --- | --- |
| `statevector` / `SV` | `statevector`, shape `(D,)` | Samples one trajectory | A stochastic final state represents one shot |
| `density_matrix` / `DM` | `density_matrix`, shape `(D, D)` | Applies them exactly | Uses more memory than `statevector` |
| `unitary` | `unitary`, shape `(D, D)` | Rejected | Rejects measurement, conditions, counts, and `initial_state` |
| `superop` | `superop`, shape `(D**2, D**2)` | Applies them exactly | Rejects measurement, conditions, counts, and `initial_state` |

Super-operators use column-stacking vectorization:

```python
rho_out = (
    superop @ rho_in.reshape(-1, order="F")
).reshape(rho_in.shape, order="F")
```

For a noise-free program, `superop` equals
`numpy.kron(unitary.conj(), unitary)`. A unitary contains `4**n` complex
entries for `n` qubits, while a super-operator contains `16**n`; use the
operator methods only for programs small enough to hold the result.

## Runtime and execution


`runtime` is chosen when the backend is created. `"numba"` is the default
for [`Simulator`][fatqat.simulator.Simulator] and the superconducting profiles; it compiles kernels
on first use and supports threaded kernels. `"numpy"` runs directly without
compilation and is the default for [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator]. Both runtimes
support all four methods, but need not produce bit-identical floating-point or
sampled results.

`simulation_config` changes one call to [`Simulator.run`][fatqat.simulator.Simulator.run]. Its string
values are case-sensitive.

**Simulation controls**

| Key | Default | Accepted values and effect |
| --- | --- | --- |
| `seed` | `None` | Use a non-negative `int` or `None`; booleans are rejected. It controls measurement, reset, channel, loss, and readout sampling. A negative value is rejected when execution starts, so [`fatqat.Job.result`][fatqat.Job.result] raises `ValueError`. |
| `shot_parallelism` | `"auto"` | `"auto"`, `"serial"`, `"threads"`, or `"processes"`. Explicit parallel modes require an eligible counts-only, per-shot run with at least two shots and workers. Threads require a compatible Numba statevector run. |
| `kernel_parallelism` | `"auto"` | `"auto"`, `"serial"`, or `"threads"`. Threads require Numba and cannot be requested together with parallel shots. |
| `max_workers` | `None` | `None` or a positive `int`. It caps the selected parallel mode; `1` conflicts with an explicitly parallel request. |
| `fusion` | `False` | A `bool`. `True` combines compatible adjacent operations and is supported by Numba for `density_matrix`, `unitary`, and `superop`. |

Automatic selection uses at most one parallel axis. An explicit unsupported
choice raises an error instead of falling back. A run is eligible for explicit
shot parallelism only when it requests counts and must evolve shots
independently—for example, because it contains mid-circuit measurement, reset,
conditions, or stochastic channels. A circuit that evolves once and samples
only terminal measurements is not eligible. Threaded shots require a Numba
statevector run and do not support the atom-occupancy lifecycle; processes are
the alternative for other eligible workloads.

A fixed non-negative `seed` reproduces sampled results only when the Program,
complete configuration, FatQat version, and execution environment are also the
same. Changing the runtime or parallel execution mode can change random-number
consumption. Deterministic results do not depend on the seed and follow normal
floating-point tolerances.

See
[Performance and scaling](../guide/performance.md) for a practical benchmarking workflow; the table
above is the canonical configuration contract.

## Customize the backend


The constructor also accepts:

**Backend options**

| Argument | Meaning |
| --- | --- |
| `implementation_map` | Matrix rules for operations. `None` uses FATQAT's built-in gate set. |
| `noise` | A [`NoiseModel`][fatqat.NoiseModel] used by every run. `None` is ideal. |
| `channel_implementation_map` | Rules that turn supported channel descriptors into finite channels. `None` uses FATQAT's built-in rules. |

## Run inputs and results


Besides `simulation_config`, [`Simulator.run`][fatqat.simulator.Simulator.run] accepts:

**Run inputs**

| Argument | Default | Meaning |
| --- | --- | --- |
| `shots` | `1024` | Samples used for counts or a stochastic final state. A deterministic state-only or operator result does not use this value. |
| `resource_layout` | `None` | Assigns every program quantum reference to a device label. The generic simulator uses integer labels in declaration order. A supplied layout must be complete, one-to-one, and compatible with the backend. |
| `initial_state` | `None` | Starts every shot from this state rather than the all-zero state. `statevector` accepts shape `(D,)`; `density_matrix` accepts `(D,)` or `(D, D)`. Operator methods reject it. |

Only the initial state's shape is checked. You are responsible for
normalization and, for density matrices, Hermiticity and positivity.

`result_config` has two keys. Each accepts `True`, `False`, or `None`;
an omitted or `None` value uses the default shown below.

**Result fields**

| Key | Default | Constraint |
| --- | --- | --- |
| `counts` | Enabled when the program measures | Requires an integer `shots > 0` |
| `final_state` | Enabled when the method-native state or map is deterministic | A requested stochastic final state requires `shots == 1` |

The concrete final-state field is named `statevector`, `density_matrix`,
`unitary`, or `superop`. Check `fatqat.Result.available_data` before
reading a field that may not have been requested.

`run()` returns an eager [`Job`][fatqat.Job]. Program and option validation
errors normally raise directly. Errors during execution or result assembly are
stored on the job and re-raised by [`fatqat.Job.result`][fatqat.Job.result]. See
[Ask questions of a run](../guide/interpret-results.md) for result accessors and count-order
intuition. Exact state-axis metadata is specified in [Result](result.md).

## Noise


Matrix simulation has no physical timeline. Built-in damping and depolarizing
descriptors therefore use their probability form and apply at operation
boundaries. Rate forms, background sources, and
[`ThermalRelaxation`][fatqat.noise.ThermalRelaxation] are rejected. For a known qubit
operation duration, add probability-form
[`AmplitudeDamping`][fatqat.noise.AmplitudeDamping] and
[`PhaseDamping`][fatqat.noise.PhaseDamping] descriptors to the relevant operation.
Custom descriptors require a matching channel rule.
[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] additionally supports atom loss.

[`Simulator.validate_noise_model`][fatqat.simulator.Simulator.validate_noise_model] validates a model without running a
program. A method can still impose a stricter rule: for example, `unitary`
rejects a finite channel that the backend otherwise recognizes when that
channel matches the program. See [Noise](noise.md) for selectors and the support
table.

## Sweeps


[`Simulator.run_sweep`][fatqat.simulator.Simulator.run_sweep] binds each row of a complete object-keyed
parameter batch and returns one eager job containing an ordered
`list[Result]`. Batch and row validation errors raise directly; an execution
failure produces a failed sweep job, and no partial result list is returned.
It reuses a supplied seed for every row, so sampled errors can be correlated.
See [Simulate a quantum program](../guide/simulation.md) for a guided sweep. Accepted batch shapes are
specified above.

## API


::: fatqat.simulator.Simulator
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: true
      filters:
        - "!^_"
