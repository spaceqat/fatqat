---
title: "Observables and estimation"
---

# Observables and estimation


Use [`Estimator`][fatqat.Estimator] to evaluate one or more
[`Observable`][fatqat.Observable] values through a backend's native execution
path. The program must be unmeasured, fully bound, and qubit-only. Built-in
matrix simulators and the two-level atom pulse emulator implement this
interface; physical models with higher local dimensions have no implicit
qubit-observable embedding.

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
nonconstant term occurrence independently through the backend's normal
measurement path, and
[`get_standard_error`][fatqat.Result.get_standard_error] reports the resulting
sample standard error. Constant terms need no execution and contribute zero
standard error; one sample of a nonconstant term has undefined standard error
and therefore reports `nan`. Set `simulation_config["seed"]` to reproduce a
sampled run.

Configure the simulation method, runtime, and noise on the backend. Invalid
programs, observable widths, and shot values raise
[`BackendValidationError`][fatqat.errors.BackendValidationError] before a job is returned;
unsupported observable types raise `TypeError`. Later failures are raised by
[`result`][fatqat.Job.result].

The Estimator forwards `simulation_config` to the backend and accepts the same
keys as its normal `run()` path. See the canonical tables for
[matrix simulators](simulator.md#runtime-and-execution) and
[the two-level atom emulator](atom-emulators.md#run-configuration-and-results).
`result_config` is not an Estimator input because the backend chooses the
internal states or measurements required by the requested observables.

Matrix simulators support all observable factors for exact and sampled runs
with statevector or density-matrix methods. Unitary and superoperator methods
are rejected with [`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError]
before a job is returned because they produce maps rather than states. Use a
density-matrix method for an exact expectation when reset or an applicable
stochastic channel is present; a statevector method can execute those programs
with positive shots. Carrier-loss programs are rejected because the current
observable contract does not define expectations across changing occupancy.

The two-level atom pulse emulator supports exact qubit observables with
statevector or density-matrix methods. Exact statevector evaluation rejects
potentially active Lindblad evolution; use density-matrix exact execution or a
positive-shot trajectory run instead. Positive-shot pulse evaluation currently
supports diagonal `Z`, `ZERO`, and `ONE` factors. Pulse reset is unsupported.
Transmon and three-level atom emulators are rejected because their physical
local dimensions exceed the qubit observable space.

Exact evaluation rejects applicable readout-confusion noise because exact
readout-noise semantics are not defined. Positive-shot evaluation applies
readout confusion through the backend's normal measurement routing. On matrix
simulators, sampled `X` and `Y` factors use ideal Estimator-owned basis changes
before that routed measurement; those synthetic changes are not user-program
operations and do not receive operation-scoped noise.

Read estimator results with [`get_expectation`][fatqat.Result.get_expectation]
and [`get_standard_error`][fatqat.Result.get_standard_error]. Run the backend
separately if you also need its final state. Estimator metadata is intentionally
compact: backend name, method, runtime, and shots, plus the pulse solver when an
actual pulse execution reports one.

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
