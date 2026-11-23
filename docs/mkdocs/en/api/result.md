---
title: "Result"
---

# Result

A [`Result`][fatqat.Result] contains the data produced by one execution, such as
measurement counts, a final state, or expectation values. Its `available_data`
lists the fields you can read. Accessing a missing field raises
[`ResultFieldUnavailableError`][fatqat.errors.ResultFieldUnavailableError].

## Find result fields

| Field | Accessor | Produced by |
| --- | --- | --- |
| `"counts"` | [`get_counts`][fatqat.Result.get_counts] or [`get_counts_as_tuples`][fatqat.Result.get_counts_as_tuples] | A backend run with measurement counts |
| `"statevector"` | [`get_statevector`][fatqat.Result.get_statevector] | A statevector run with final-state output enabled |
| `"density_matrix"` | [`get_density_matrix`][fatqat.Result.get_density_matrix] | A density-matrix run with final-state output enabled |
| `"fixed_photon_state"` | [`get_fixed_photon_state`][fatqat.Result.get_fixed_photon_state] | A linear-optics run that returns fixed-photon amplitudes |
| `"unitary"` | [`get_unitary`][fatqat.Result.get_unitary] | A unitary run with final-state output enabled |
| `"superop"` | [`get_superop`][fatqat.Result.get_superop] | A super-operator run with final-state output enabled |
| `"classical_state"` | [`get_classical_state`][fatqat.Result.get_classical_state] | A state-method simulator run with `result_config={"final_classical_state": True}` |
| `"expectation"` and `"standard_error"` | [`get_expectation`][fatqat.Result.get_expectation] and [`get_standard_error`][fatqat.Result.get_standard_error] | An [`Estimator`][fatqat.Estimator] run |
| Backend extension name | [`get_data`][fatqat.Result.get_data] | A backend extension |

A backend's `result_config` uses `"final_state"` to request a state or operator.
Look for its representation name in `available_data`, such as `"statevector"` or
`"fixed_photon_state"`. The [simulator](simulator.md#run-inputs-and-results) and
[pulse emulator](emulators/index.md#choose-a-result) pages describe the defaults
and how to request a single trajectory.

<a id="ordering-and-mutable-values"></a>

## Read counts and states

### Count keys

Choose [`get_counts()`][fatqat.Result.get_counts] for display strings or
[`get_counts_as_tuples()`][fatqat.Result.get_counts_as_tuples] for integer tuples.
Both follow classical register order. A tuple key `(1, 0)` is displayed as `"10"`
in a binary register.

A classical register member that no measurement writes stays at zero. Pulse
emulators also issue a `UserWarning` if you request counts without a final state
and leave a declared member unwritten.

<a id="draw-counts"></a>

A result with measurement counts can plot them with
[`draw()`][fatqat.Result.draw].

### State axes

Read `metadata["state_axes"]` to match state and operator entries to physical
subsystems. Each entry gives a device label (`device_operand`) and its program
reference (`register_ref`). An unused physical subsystem has reference `None`. The
first subsystem is the most significant in the basis index. For dimensions `dims`,
position `q` has place value `prod(dims[q + 1:])`. This order applies to
statevectors, both axes of density matrices and unitaries, and the underlying
basis of superoperators.

Simulators return NumPy arrays. For statevectors, density matrices, and unitaries,
pulse emulators return QuTiP `Qobj` values. The individual state accessors
document the shapes and how to convert to arrays.

## Result metadata

Built-in simulators and pulse emulators record the effective `execution_config`
and `result_config` in `metadata` for direct runs. The [`Result`][fatqat.Result]
reference also describes solver information available from pulse runs. If you plan
to edit a stored array or dictionary, copy it first to preserve the original.

## API reference { #detailed-reference }

::: fatqat.Result
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
