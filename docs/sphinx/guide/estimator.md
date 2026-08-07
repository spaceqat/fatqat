# Expectation values

Counts answer "what did I measure?". An {py:class}`~fatqat.Observable` and an
{py:class}`~fatqat.Estimator` answer "what is the value of this quantity?" —
energies, magnetizations, site occupations, cost functions — without going
through bitstrings at all.

```python
import fatqat as fq
import fatqat.operations as op

program = fq.Program(2)          # no classical bits, and no measurement
program.add(op.H, 0)
program.add(op.CX, (0, 1))

estimator = fq.Estimator(fq.simulator.Simulator(method="SV"))
result = estimator.run(program, fq.Observable([("ZZ", 1.0)])).result()
print(result.get_expectation())  # 1.0
```

The backend is built exactly as for a counts run and owns the method, runtime,
and noise model. `fq.Estimator` wraps it and adds only the observable step, so
one backend can serve both `backend.run` for counts and `estimator.run` for
expectation values.

Note the program has **no measurement**. An expectation value is read from the
final state, and a measurement collapses that state, so the two are mutually
exclusive by construction — see **Restrictions** at the end of this page.

## Writing an observable

An observable is a weighted sum of terms, each a product of single-qubit
letters. Labels are little-endian, matching fatqat's counts strings: the
rightmost character is qubit 0.

```python
fq.Observable([("ZZ", 1.0)])                     # 1.0 * Z_1 Z_0
fq.Observable([("ZZ", 1.0), ("XX", 0.5)])        # a two-term sum
fq.Observable(["ZZ", "XX"], coeffs=[1.0, 0.5])   # the same, labels and coeffs
```

Coefficients must be real: every letter is Hermitian, so the observable is
Hermitian exactly when its coefficients are, and then the expectation value is
real too. A complex coefficient is rejected at construction.

For a wide register, name only the factors that are not the identity instead of
writing a label full of `I`s:

```python
# 1.5 * X_3 Y_7 on a 100-qubit register
fq.Observable.from_sparse([("XY", (3, 7), 1.5)], num_qubits=100)
```

Storage grows with the number of factors written, not with the qubit count: a
two-body term on 100 qubits costs the same as one on 4. The
`2**n x 2**n` matrix is never built — not during construction, and not during
evaluation.

### Letters

| letter | operator | notes |
| --- | --- | --- |
| `I` | identity | omitted from storage |
| `X`, `Y`, `Z` | Pauli operators | usable in a dense label |
| `ZERO` | `\|0><0\|` | projector; `from_sparse` only |
| `ONE` | `\|1><1\|` | projector; `from_sparse` only |

`ZERO` and `ONE` are multi-character names, so a dense label cannot hold them —
`"ONE"` would read as three separate letters. They are reachable through
`from_sparse`, where each factor is named separately. A term mixing projectors
and Paulis is written as an explicit list:

```python
occupation = fq.Observable.from_sparse([("ONE", (5,), 1.0)], num_qubits=8)
mixed = fq.Observable.from_sparse([(["ONE", "Z"], (5, 3), 1.0)], num_qubits=8)
```

The projectors make site occupation directly expressible: `<ONE_i>` is the
occupation number of qubit `i`, the quantity atom-array experiments report.
Writing it as `(I - Z_i)/2` also works, but costs two terms and an offset.

## Several observables at once

Pass a list to evaluate many observables against a single evolution:

```python
observables = [
    fq.Observable([("ZZ", 1.0)]),
    fq.Observable([("XX", 1.0)]),
    fq.Observable([("YY", 1.0)]),
]
values = estimator.run(program, observables).result().get_expectation()
```

The result shape mirrors the input shape: a float for a single observable, an
array for a sequence — including a one-element sequence.

This is where a simulator differs from hardware. Hardware must fan a
multi-basis observable out into one circuit per commuting group, each with its
own basis-rotation gates appended, and run each separately. A simulator holds
the final state and reads any observable off it directly, so the evolution is
paid for once no matter how many observables or terms follow. Passing a list
rather than calling `run` repeatedly is what claims that saving.

### A note on speed

Once the evolution is shared, the remaining cost is one sweep over the state
per term — so with many terms, evaluation rather than evolution dominates. That
sweep is compiled when the optional `numba` dependency is installed, which is
worth roughly an order of magnitude on a large state.

Nothing needs to be configured: the compiled kernel is used whenever `numba`
can be imported, and the NumPy path is a complete fallback otherwise. It is
**independent of the backend's `runtime=`** — that selects how the *program* is
evolved, while this is how the *observable* is evaluated, and a
`runtime="numpy"` backend benefits from the compiled kernel just as much.

The two agree to floating-point rounding rather than bit for bit, because NumPy
sums pairwise and a loop sums in order. The compiled kernel sums in blocks so
that its rounding error stays flat as the state grows rather than accumulating
across a million amplitudes.

## Exact and sampled

`shots=0` — the default — computes the value exactly from the final state. Note
this differs from {py:meth}`~fatqat.simulator.Simulator.run`, whose `shots`
defaults to 1024; an estimator's usual request is the exact value.

`shots > 0` reproduces the statistical error of a finite-shot experiment by
drawing real samples from each term's eigenvalue distribution, so a small shot
count shows the granularity a real experiment would have.

```python
tilted = fq.Program(2)          # a state the Bell example's +-1 values would hide
tilted.add(op.RY(1.0), 0)
tilted.add(op.CX, (0, 1))
observable = fq.Observable([("ZZ", 1.0), ("XX", 0.5)])

exact = estimator.run(tilted, observable).result()
sampled = estimator.run(tilted, observable, shots=1000,
                        simulation_config={"seed": 7}).result()

print(exact.get_expectation(), exact.get_std())      # 1.4207354924039484 0.0
print(sampled.get_expectation(), sampled.get_std())  # 1.437 0.008542929557921405
```

{py:meth}`~fatqat.Result.get_std` reports the standard error, mirroring
`get_expectation`'s shape. It is `0` for an exact run, which carries no
statistical error because nothing was sampled. The reported value is analytic —
`sqrt(sum_k c_k^2 Var(T_k) / shots)` — so it states the precision of the
request rather than adding a second layer of noise on top of the sample.

Terms are sampled independently, so `get_std` omits correlation between the
terms of one observable; in a real experiment they are measured on the same
state. Treat it as the precision of the request, not as a full error model.

A `seed` in `simulation_config` seeds the sampling as well as the backend, so a
seeded run reproduces:

```python
estimator.run(program, observable, shots=1000, simulation_config={"seed": 7})
```

## Noise

Noise is configured on the backend, exactly as for a counts run, and the value
stays exact:

```python
noise = fq.NoiseModel()
noise.add_channel(fq.noise.Depolarizing(p=0.1), operation=op.CX)

estimator = fq.Estimator(fq.simulator.Simulator(method="DM", noise=noise))
print(estimator.run(program, fq.Observable([("ZZ", 1.0)])).result().get_expectation())
# 0.8999999999999997  -- the exact value under the channel, no sampling
```

Use `method="DM"` for this. A density matrix applies each channel as an exact
map, so the noisy expectation value is computed rather than estimated — the
`0.9` above is not an average over runs.

## Restrictions

Each of the following is rejected because the expectation value would be
ill-defined, not merely unsupported.

**A program that measures.** The measurement collapses the state, so "the
expectation value of the final state" has no single meaning. Build the program
without measurements — usually without classical bits at all — or use
`backend.run` when you want counts. Qiskit's estimators reject measured
circuits for the same reason.

**A statevector run whose evolution is stochastic** — one carrying `Reset`, or
channel noise that actually reaches the program. Both sample one branch per
shot under statevector semantics, so there is no single final state to
evaluate, and none to sample from either. This is rejected at `shots > 0` as
well, since sampling one branch would quietly report that trajectory's
statistics instead of the noisy channel's. Use `method="DM"`, where both are
exact channels.

The test is what the run actually does, not what the noise model contains: a
channel registered on a gate the program never uses never fires, so such a run
stays deterministic and is accepted. The backend decides this, from the lowered
program.

**A width mismatch.** The observable's qubit count must equal the program's.

**Non-qubit registers.** The letter alphabet has no meaning for `dim > 2`.

## See also

- [Running and results](running-and-results.md) — counts, states, and
  `result_config`.
- [Noise](noise.md) — building the noise model an estimator's backend carries.
- {doc}`../api/estimator` — exact signatures.
