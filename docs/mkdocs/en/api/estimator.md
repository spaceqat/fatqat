---
title: "Observables and estimation"
---

# Observables and estimation


Use [`Estimator`][fatqat.Estimator] to evaluate one or more
[`Observable`][fatqat.Observable] values from a backend's final state. The program
must be unmeasured, fully bound, and qubit-only; the backend must return a
statevector or density matrix in the program's logical qubit space.

## Estimate an observable


```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2)
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))

estimator = fq.Estimator(fq.simulator.Simulator("SV"))
observable = fq.Observable([("ZZ", 1.0)])
result = estimator.run(program, observable).result()
expectation = result.get_expectation()
```

For a guided workflow, see [Ask questions of a run](../guide/interpret-results.md).

## Construct an observable


Dense labels run left to right in public qubit order: the character at
position `q` acts on qubit `q`. They accept `I`, `X`, `Y`, and `Z`. These
forms are equivalent:

```python
fq.Observable([("ZZ", 1.5)])
fq.Observable(["ZZ"], coeffs=[1.5])
```

[`from_sparse`][fatqat.Observable.from_sparse] names each non-identity factor and its
qubit explicitly. It also supports `ZERO` and `ONE` projectors:

```python
fq.Observable.from_sparse(
    [(["ONE", "Z"], (5, 3), 1.5)],
    num_qubits=6,
)
```

Coefficients must be real.

For example, `"XI"` means `X(q0) tensor I(q1)`. On public state `|10>`,
`<ZI> = -1` and `<IZ> = +1`. Dense labels therefore match an explicit
left-to-right Kronecker product; sparse labels retain the qubit associations
written in their `qubits` tuple.

## Exact and sampled results


Pass one observable to receive scalar expectation and standard-error values,
or a list or tuple to receive arrays in the same order.

`shots=0` computes an exact value. A positive `shots` value samples each
observable term, and
[`get_standard_error`][fatqat.Result.get_standard_error] reports the resulting
standard error. Set `simulation_config["seed"]` to reproduce a sampled run.

Configure the simulation method, runtime, and noise on the backend. Invalid
programs, observable widths, and shot values raise
[`BackendValidationError`][fatqat.errors.BackendValidationError] before a job is returned;
unsupported observable types raise `TypeError`. Later failures are raised by
[`result`][fatqat.Job.result].

Use a density-matrix backend when the program resets qubits or channel noise
applies.

For a program with `N` qubits, the returned statevector must have shape
`(2**N,)` or the density matrix must have shape `(2**N, 2**N)`. A backend
state with another shape raises [`BackendValidationError`][fatqat.errors.BackendValidationError]
before any Pauli expectation kernel runs. In particular, a full physical qutrit
state returned by a Transmon emulator is not implicitly projected into the
logical subspace. Inspect that physical state directly with the backend
[`Result`][fatqat.Result] until explicit leakage-aware observable semantics are
available.

Read estimator results with [`get_expectation`][fatqat.Result.get_expectation]
and [`get_standard_error`][fatqat.Result.get_standard_error]. Run the backend
separately if you also need its final state.

## Parameter sweeps


[`run_sweep`][fatqat.Estimator.run_sweep] evaluates binding rows in input order.
Validation errors raise directly; other row failures are raised by
[`result`][fatqat.Job.result]. No partial result list is returned. See
[Simulate a quantum program](../guide/simulation.md) for a guided parameter sweep; accepted binding
shapes and seed behavior are specified here.

## Detailed reference


::: fatqat.Observable
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.Estimator
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"
