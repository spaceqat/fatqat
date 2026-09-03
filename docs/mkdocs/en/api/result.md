---
title: "Result"
---

# Result


A [`Result`][fatqat.Result] contains only the fields produced by one run.
Inspect `available_data` when a field is optional, then
use its accessor. An unavailable accessor raises
[`ResultFieldUnavailableError`][fatqat.errors.ResultFieldUnavailableError] instead of returning
`None`.

**Result fields**

| Field | Accessor | Produced by |
| --- | --- | --- |
| `"counts"` | [`get_counts`][fatqat.Result.get_counts] or [`get_counts_as_tuples`][fatqat.Result.get_counts_as_tuples] | A backend run with measurement counts |
| `"statevector"` | [`get_statevector`][fatqat.Result.get_statevector] | A statevector run with final-state output enabled |
| `"density_matrix"` | [`get_density_matrix`][fatqat.Result.get_density_matrix] | A density-matrix run with final-state output enabled |
| `"unitary"` | [`get_unitary`][fatqat.Result.get_unitary] | A unitary run with final-state output enabled |
| `"superop"` | [`get_superop`][fatqat.Result.get_superop] | A super-operator run with final-state output enabled |
| `"expectation"` and `"standard_error"` | [`get_expectation`][fatqat.Result.get_expectation] and [`get_standard_error`][fatqat.Result.get_standard_error] | An [`Estimator`][fatqat.Estimator] run |
| Backend extension name | [`get_data`][fatqat.Result.get_data] | A backend extension |

`"final_state"` is a request name, not an available-data name. A produced
state or operator uses its concrete representation name from the table.
Deterministic unmeasured runs enable method-native output by default. A
stochastic pulse-emulator state—caused by measurement, statevector reset, or
potentially active statevector Lindblad evolution—must be requested explicitly
with `shots=1`.

## Ordering and mutable values


[`get_counts`][fatqat.Result.get_counts] returns a new dictionary of display strings.
Classical slot 0 is on the left, matching tuple position 0 from
[`get_counts_as_tuples`][fatqat.Result.get_counts_as_tuples]. Thus tuple `(1, 0)` renders as
`"10"`, and `(0, 1)` renders as `"01"`. If any classical dimension is at
least 10, commas make multi-digit outcomes unambiguous: `(10, 3)` renders as
`"10,3"`.

Most other accessors return the value stored in the result. Copy arrays or
dictionaries before changing them if you need to preserve the original.
Metadata records the normalized `simulation_config` and `result_config`.
Backend extensions may add fields. Pulse-emulator metadata records
`runtime="qutip"` and a `runtime_details` mapping with `solver` and
`solver_options`. A run that invokes multiple solvers reports their names as
a tuple. These numerical-runtime details are informational. Keep the
model, arrangement, controls, and application metadata alongside a result when
they are needed to reproduce a physical run.

When a result includes a state or operator, `metadata["state_axes"]` lists
its physical subsystems from most to least significant. Each entry contains a
`device_operand` and its program `register_ref`; `register_ref` is
`None` when a physical model contains a subsystem the Program did not
address. Position 0 is the most-significant subsystem of a flat basis index.
For local dimensions `dims`, position `q` has place value
`prod(dims[q + 1:])`. Statevectors, both density-matrix and unitary axes, and
the global operator basis of super-operators use this same public order.

A counts-only run zero-fills every declared classical slot that was never
written by measurement and emits a standard `UserWarning`. This usually
indicates a missing measurement.

See [Ask questions of a run](../guide/interpret-results.md) for the guided interpretation workflow;
the conventions above are the canonical state-axis and count-order contract.

## Draw counts

A result containing counts can draw them directly:

```python
figure = result.draw()
figure = result.draw(stat="frequencies")
figure = result.draw(number_to_keep=20, sort="count")
```

The default `stat="counts"` plots raw occurrences. `stat="frequencies"`
divides each value by the total number of shots. Colors inherit Matplotlib's
active style and `rcParams`. `number_to_keep` retains the most frequent
outcomes and combines the remainder into an `other` bar.

## Detailed reference


::: fatqat.Result
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
