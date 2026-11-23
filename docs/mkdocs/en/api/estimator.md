---
title: "Observables and estimation"
---

# Observables and estimation

[`Estimator`][fatqat.Estimator] computes expectation values for one or more
[`Observable`][fatqat.Observable] objects. Both entry points take a program
without measurement operations. Choose the entry point by how you want to supply
the program parameters.

<a id="parameter-sweeps"></a>

## Run once or sweep

- [`run()`][fatqat.Estimator.run] evaluates a single program. Bind all its
  parameters before calling it. The job returns one [`Result`][fatqat.Result].

- [`run_sweep()`][fatqat.Estimator.run_sweep] takes a parameterized program and a
  batch of complete `bindings` in the same call. The job returns one result per
  binding row, in input order.

## Construct an observable

Each term is the product of operators on individual subsystems, multiplied by a
real coefficient. Use a dense string of `I`, `X`, `Y`, and `Z` for Pauli operators
on qubits. The leftmost letter acts on subsystem 0. For example, `"XI"` applies X
to the first qubit and the identity to the second.

Use [`from_sparse()`][fatqat.Observable.from_sparse] to name the operators and
their subsystem indices directly. This form also accepts
[`Projector(level)`][fatqat.observable.Projector] for the probability of finding a
subsystem in a level and [`NumberOperator()`][fatqat.observable.NumberOperator]
for its mean level or photon number. Unspecified subsystems carry the identity.

## Estimate an expectation value { #estimate-an-observable }

For a three-level system prepared in level 2, this example reads the mean level
index and the probability of being in level 2.

```pycon
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> from fatqat.observable import NumberOperator, Projector
>>> program = fq.Program([fq.QuantumRegister(1, dim=3)])
>>> program.add(ops.Shift(2), 0)
>>> mean_level = fq.Observable.from_sparse([(NumberOperator(), (0,), 1.0)])
>>> in_level_2 = fq.Observable.from_sparse([(Projector(2), (0,), 1.0)])
>>> estimator = fq.Estimator(fq.simulator.Simulator(runtime="numpy"))
>>> result = estimator.run(program, [mean_level, in_level_2]).result()
>>> result.get_expectation().tolist()
[2.0, 1.0]
```

## Exact and sampled results

`shots=0` (the default) computes the expectation directly and reports zero
standard error. A positive `shots` sets the number of samples per nonconstant
term. Read the values with [`get_expectation()`][fatqat.Result.get_expectation]
and [`get_standard_error()`][fatqat.Result.get_standard_error].

## Backend support

Set the method, numerical runtime, and noise on the backend. Pass per-run options
such as the seed through `execution_config`. Simulators also accept an
`initial_state`.

- **Gate-level simulators:** Use `statevector` or `density_matrix`. For an exact
  result with stochastic reset or gate noise, use `density_matrix`. A statevector
  method can sample those runs with positive `shots`. Programs that use custom
  operations are unsupported.

- **Linear optics:** [`LOSimulator`][fatqat.simulator.LOSimulator] also supports
  `fixed_photon_state`. To evaluate an optical mode, use
  [`NumberOperator`][fatqat.observable.NumberOperator] or a
  [`Projector`][fatqat.observable.Projector] with a level at or below the photon
  cutoff. Set the input photons with `initial_state`.

- **Neutral-atom arrays:** On
  [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator], the observable must
  target loaded sites in a loss-free program.

- **Pulse emulation:** [`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator]
  supports both state methods, without reset. Use a density matrix for exact
  evaluation with Lindblad noise, or sample with positive `shots`. Sampled runs
  accept only diagonal operators, including `Z`, `Projector(0)`, `Projector(1)`,
  and `NumberOperator()`. Transmon and three-level atom pulse models are
  unsupported.

When readout confusion applies to an observable, use positive `shots`. Gate-level
simulators measure sampled `X` and `Y` terms after ideal basis changes. Gate noise
acts on the program's operations but leaves those basis changes ideal.

## API reference { #detailed-reference }

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
