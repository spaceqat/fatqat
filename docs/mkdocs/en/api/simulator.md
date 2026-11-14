---
title: "Simulator"
---

# Simulator


[`Simulator`][fatqat.simulator.Simulator] runs gate-level programs with matrix
operations and finite noise channels. It supports qubits, qudits, mixed register
dimensions, and custom implementation maps. Programs run as written, without
transpilation or routing.

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

All returned quantum arrays list subsystem 0 as the most-significant factor.
For two qubits, `X(q0)|00>` is nonzero at flat index
2, while `X(q1)|00>` is nonzero at index 1. For dimensions `(3, 2)`, digits
`(1, 0)` and `(0, 1)` likewise have indices 2 and 1. Ordinary NumPy C-order
reshape follows this order.

The same basis applies to input states and to operator rows and columns. For
an arbitrary state `psi`, a matrix acting on every subsystem satisfies
`unitary @ psi == state_out`; density matrices satisfy
`unitary @ rho @ unitary.conj().T == rho_out`.

## Runtime and execution


`runtime` accepts `"numpy"` or `"numba"` (case-insensitive). `None` (default)
uses the current simulator default, initially `"numba"`. See
[Simulation](../guide/simulation.md) to set the default for a script or notebook.

`simulation_config` changes one call to [`Simulator.run`][fatqat.simulator.Simulator.run]. Its string
values are case-sensitive.

| Key | Default | Accepted values and effect |
| --- | --- | --- |
| `seed` | `None` | Use a non-negative `int` or `None`; booleans are rejected. It controls measurement, reset, channel, loss, and readout sampling. A negative value fails when execution starts and follows `error_handling`. |
| `shot_parallelism` | `"auto"` | `"auto"`, `"serial"`, `"threads"`, or `"processes"`. Explicit parallel modes require an eligible counts-only, per-shot run with at least two shots and workers. Threads require a compatible Numba statevector run. |
| `kernel_parallelism` | `"auto"` | `"auto"`, `"serial"`, or `"threads"`. Threads require Numba and cannot be requested together with parallel shots. |
| `max_workers` | `None` | `None` or a positive `int`. It caps the selected parallel mode; `1` conflicts with an explicitly parallel request. |
| `error_handling` | `"raise"` | `"raise"` or `"capture"`; see the error-handling contract in [`Simulator.run`][fatqat.simulator.Simulator.run]. |
| `fusion` | `False` | A `bool`. `True` allows the engine to merge compatible adjacent operations. It is accepted for `statevector` and `density_matrix` on either runtime, and for `unitary` and `superop` on Numba; NumPy rejects it for `unitary` and `superop`. The engine may skip merging, for example on a small system. Merging follows program order and does not reorder gates. One merged matrix spans only a few qubits. On a wider register, gates on the same qubits merge best when they are adjacent: write `RX(q0), RZ(q0), RX(q1), RZ(q1), ...` rather than `RX(q0), RX(q1), ..., RZ(q0), RZ(q1), ...`. It is rejected when the run tracks atom occupancy, so `AtomArraySimulator` always rejects it. A custom operation that enables occupancy during a fused run raises `BackendExecutionError`. |

Shot parallelism requires counts and independent shot evolution, such as a
program with mid-circuit measurement, reset, conditions, or stochastic channels.
Programs with only terminal measurement sampling are not eligible. Threaded
shots require a Numba statevector run without atom-occupancy operations;
processes support other eligible workloads. Automatic selection uses at most
one parallel axis. Unsupported explicit choices raise an error.

A fixed `seed` reproduces sampled results when the program, configuration,
FatQat version, and execution environment are the same. Different runtimes or
parallel modes may produce different samples. Deterministic results are
independent of the seed and subject to floating-point tolerances.

See [Performance and scaling](../guide/performance.md) for benchmarking guidance.

## Customize the backend


The constructor also accepts:

| Argument | Meaning |
| --- | --- |
| `implementation_map` | Matrix, channel, and [custom-operation implementations](simulators/custom-operations.md). `None` uses FATQAT's built-in rules. |
| `noise` | A [`NoiseModel`][fatqat.NoiseModel] used by every run. `None` is ideal. |

## Run inputs and results


Besides `simulation_config`, [`Simulator.run`][fatqat.simulator.Simulator.run] accepts:

| Argument | Default | Meaning |
| --- | --- | --- |
| `shots` | `1024` | Samples used for counts or a stochastic final state. A deterministic state-only or operator result does not use this value. |
| `resource_layout` | `None` | Assigns every program quantum reference to a device label. The generic simulator uses integer labels in declaration order. A supplied layout must be complete, one-to-one, and compatible with the backend. |
| `initial_state` | `None` | Quantum array, [`BasisState`][fatqat.simulator.BasisState], or [`EvolutionState`][fatqat.simulator.EvolutionState] supplying quantum and classical data for each shot. `None` selects the ground state. Operator methods reject an initial state. |

A `BasisState` names one level per subsystem by register reference, so it does
not depend on subsystem order; unnamed subsystems start at level 0. Dense
quantum input uses the public most-significant-first basis. `statevector` accepts
shape `(D,)`; `density_matrix` accepts `(D,)` or `(D, D)`. Only the quantum shape is
checked; you are responsible for normalization and, for density matrices,
Hermiticity and positivity.

`result_config` accepts the keys below. `counts` and `final_state` accept
`True`, `False`, or `None`; an omitted or `None` value uses the default shown.
`final_classical_state` accepts only `True` or `False`.

| Key | Default | Constraint |
| --- | --- | --- |
| `counts` | Enabled when the program measures | Requires an integer `shots > 0` |
| `final_state` | Enabled when the method-native state or map is deterministic | A requested stochastic final state requires `shots == 1` |
| `final_classical_state` | Disabled | Requires `shots == 1` when the final state is stochastic; rejected by `unitary` and `superop` |

`final_classical_state=True` returns one shot's final [`ClassicalState`][fatqat.simulator.ClassicalState]
(register digits and site occupancy) through [`Result.get_classical_state`][fatqat.Result.get_classical_state].

The concrete final-state field is named `statevector`, `density_matrix`,
`unitary`, or `superop`. Check `fatqat.Result.available_data` before
reading a field that may not have been requested.

`run()` completes execution before returning a [`Job`][fatqat.Job]. Program and
option validation errors raise directly. Execution errors follow
`error_handling`, as specified in [`Simulator.run`][fatqat.simulator.Simulator.run]. See
[Result](result.md) for available fields, accessors, count ordering, and
state-axis metadata.

## Noise


Matrix simulation has no physical timeline. Built-in damping and depolarizing
descriptors therefore use their probability form and apply at operation
boundaries. [`AmplitudeDamping(p=...)`][fatqat.noise.AmplitudeDamping] is the
conventional finite qubit channel. Use
[`TransitionRelaxation(p=..., coefficients=...)`](noise/transition-relaxation.md)
for one explicit finite-dimensional jump. Matching transition descriptors are
applied sequentially in registration order.

Rate forms, background sources, and
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


[`Simulator.run_sweep`][fatqat.simulator.Simulator.run_sweep] runs a program for
each row of a parameter batch. It completes before returning a job containing
a `list[Result]` in row order. Batch and row validation errors raise directly;
execution errors follow `error_handling`. A failed sweep returns no partial
result list. A supplied seed is reused for every row, so sampled errors can be
correlated.

See [Simulate a quantum program](../guide/simulation.md) for an example and
[`run_sweep`][fatqat.simulator.Simulator.run_sweep] for accepted batch shapes.

## API


::: fatqat.simulator.get_default_runtime

::: fatqat.simulator.set_default_runtime

::: fatqat.simulator.Simulator
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: true
      filters:
        - "!^_"

::: fatqat.simulator.BasisState
