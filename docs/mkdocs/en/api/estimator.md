---
title: "Observables and estimation"
---

# Observables and estimation


Use [`Estimator`][fatqat.Estimator] to evaluate one or more
[`Observable`][fatqat.Observable] values on a backend. The program must be
unmeasured and fully bound with no free Parameters.

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

For a guided workflow, see [Estimate observables](../guide/interpret-results.md).

## Construct an observable


An observable is a sum of real coefficients times products of
single-subsystem factors:

- the Pauli factors `X`, `Y`, and `Z`, which act on qubits only;
- [`Projector(level)`][fatqat.observable.Projector], the projector onto one
  level, with `ZERO` and `ONE` as names for `Projector(0)` and `Projector(1)`;
- [`NumberOperator()`][fatqat.observable.NumberOperator], `diag(0, 1, ..., d-1)`
  on a subsystem of dimension `d`: the mean level of a qudit, or `ONE` on a
  qubit.

A string label gives one letter per subsystem, from `I`, `X`, `Y`, and `Z`. The
character at position `q` acts on subsystem `q`, and subsystems past the end of
the label carry the identity. `"XI"` means `X(q0) tensor I(q1)`; on the state
`|10>`, `<ZI> = -1` and `<IZ> = +1`. These forms are equivalent:

```python
fq.Observable([("ZZ", 1.5)])
fq.Observable(["ZZ"], coeffs=[1.5])
```

[`from_sparse`][fatqat.Observable.from_sparse] names each non-identity factor
and its subsystem explicitly. A term's factors can be a letter string such as
`"XY"`, one factor object, or a sequence that mixes both. This is also how you
write `Projector` and `NumberOperator` factors:

```python
from fatqat.observable import NumberOperator, Projector

fq.Observable.from_sparse([(["ONE", "Z"], (5, 3), 1.5)])
fq.Observable.from_sparse([([Projector(2), "X"], (0, 1), 0.5)])
```

Coefficients must be real.

An observable has no width or subsystem dimensions of its own. The backend
checks each factor against the program: every index must name a program
subsystem, Pauli factors need a qubit, and a projector level must be below its
subsystem's dimension. This lets one observable describe qudit programs:

```python
qutrit = fq.Program([fq.QuantumRegister(1, dim=3)])
qutrit.add(ops.Shift(2), 0)  # |0> -> |2>

mean_level = fq.Observable.from_sparse([(NumberOperator(), (0,), 1.0)])
in_level_2 = fq.Observable.from_sparse([(Projector(2), (0,), 1.0)])
result = estimator.run(qutrit, [mean_level, in_level_2]).result()
result.get_expectation()  # [2.0, 1.0]
```

## Exact and sampled results


Pass one observable to receive scalar expectation and standard-error values,
or a list or tuple to receive arrays in the same order.

`shots=0` computes an exact value. A positive `shots` value samples each
nonconstant term independently, and
[`get_standard_error`][fatqat.Result.get_standard_error] reports the resulting
sample standard error. Constant terms contribute zero standard error; one
sample of a nonconstant term has undefined standard error and reports `nan`.
Set `simulation_config["seed"]` to reproduce a sampled run.

Invalid programs, factors that do not fit the program's subsystems, and invalid
shot values raise
[`BackendValidationError`][fatqat.errors.BackendValidationError] before a job
is returned; unsupported observable types raise `TypeError`. Execution errors
follow the backend's error-handling policy.

Configure the simulation method, runtime, and noise on the backend. The
Estimator forwards `simulation_config` to the backend and accepts the same keys
as its normal `run()` path; see the tables for
[matrix simulators](simulator.md#runtime-and-execution) and
[the two-level atom emulator](atom-emulators.md#run-configuration-and-results).
`result_config` is not accepted.

Read estimator results with [`get_expectation`][fatqat.Result.get_expectation]
and [`get_standard_error`][fatqat.Result.get_standard_error]. Run the backend
separately if you also need its final state. Result metadata is compact: backend
name, method, runtime, and shots, plus the pulse solver for pulse executions.

## Backend support


**Matrix simulators** support every factor, exact and sampled, with a
statevector or density-matrix method. Unitary and superoperator methods raise
[`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError] because
they produce maps rather than states. For an exact expectation with reset or an
applicable stochastic channel, use a density-matrix method; a statevector
method can run those programs with positive shots. Programs with carrier loss
are rejected.

**The two-level atom pulse emulator** evaluates observables on its qubit
programs with a statevector or density-matrix method. There, `NumberOperator()`
equals `ONE`, and only levels 0 and 1 are valid projector levels.

- Exact statevector evaluation rejects potentially active Lindblad evolution;
  use exact density-matrix execution or a positive-shot trajectory run.
- Positive-shot evaluation supports the diagonal factors `Z`, `Projector(0)`,
  `Projector(1)`, and `NumberOperator()`.
- Reset is unsupported.

Transmon and three-level atom emulators are rejected: their local dimensions
exceed the qubit observable space.

**Readout noise.** Exact evaluation rejects readout-confusion noise, since it
has no exact meaning. Positive-shot evaluation applies readout confusion to the
measurements. On matrix simulators, a sampled `X` or `Y` is measured after an
ideal basis change that gate noise does not affect.

## Parameter sweeps


[`run_sweep`][fatqat.Estimator.run_sweep] evaluates binding rows in input order.
Validation errors raise directly; execution errors follow the backend's
error-handling policy. No partial result list is returned. See
[Simulate a quantum program](../guide/simulation.md) for a guided parameter
sweep; accepted binding shapes and seed behavior are specified here.

## Detailed reference


::: fatqat.Observable
    options:
      inherited_members: true
      show_bases: true
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.observable.NumberOperator
    options:
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.observable.Projector
    options:
      show_bases: false
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
