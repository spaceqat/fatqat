# Parameters and sweeps

Use parameters when the program structure stays fixed while numeric gate
angles change. A parameter is an identity-based placeholder: its name is a
label for people, not a lookup key.

## Reuse one parameter

```python
import fatqat as fq
import fatqat.operations as ops

theta = fq.Parameter("theta")
program = fq.Program(2)
program.add(ops.RX(theta), 0)
program.add(ops.RY(theta), 1)

bound = program.assign_parameters({theta: 0.25})
```

Both gates in `bound` use `0.25`. The original `program` still contains
`theta`. Binding always returns a new program and may be partial:

```python
angles = fq.ParameterVector("angles", 2)
template = fq.Program(2)
template.add(ops.RX(angles[0]), 0)
template.add(ops.RY(angles[1]), 1)

partly_bound = template.assign_parameters({angles[0]: 0.1})
fully_bound = partly_bound.assign_parameters({angles[1]: 0.2})
```

`ParameterVector` is an explicit group with stable element order. Two
parameters named `"theta"`, or two vectors named `"angles"`, remain distinct
objects and distinct mapping keys. String keys are not accepted.

Ordinary Simulator, Estimator, and pulse-emulator execution rejects any
remaining unbound parameter before numeric realization. QASM export does the
same before formatting numeric gate arguments.

## Simulator sweeps

`Simulator.run_sweep()` accepts a mapping from parameter objects to batches.
A single `Parameter` needs shape `(N,)`; a vector of length `M` needs shape
`(N, M)`:

```python
import numpy as np

backend = fq.simulator.Simulator("SV")
job = backend.run_sweep(
    template,
    {
        angles: np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]),
    },
    shots=0,
    result_config={"counts": False, "final_state": True},
)
results = job.result()

assert len(results) == 3
# results[i] is the ordinary Result for row i.
```

Every parameter in the template must be assigned exactly once. All batches
must have the same positive leading length. A one-point vector sweep is still
two-dimensional, for example `values[None, :]` with shape `(1, M)`; a bare
`(M,)` vector is not interpreted as one point.

A sweep requires at least one program parameter and at least one assignment.
Use ordinary `run()` for a parameter-free program.

The same options are forwarded to every row, including `shots`,
`resource_layout`, `simulation_config`, and `result_config`. Each list element
has the same fields as the equivalent direct `Simulator.run()` result.

## Estimator and QNN-style sweeps

Estimator sweeps use the identical binding shapes while preserving the normal
single-observable or multiple-observable result shape:

```python
features = fq.ParameterVector("features", 2)
weights = fq.ParameterVector("weights", 2)
qnn = fq.Program(2)
qnn.add(ops.RX(features[0]), 0)
qnn.add(ops.RY(weights[0]), 0)
qnn.add(ops.RX(features[1]), 1)
qnn.add(ops.RY(weights[1]), 1)

X = np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]])
weight_value = np.array([0.3, 0.6])
weight_batch = np.broadcast_to(weight_value, (len(X), len(weights)))

estimator = fq.Estimator(fq.simulator.Simulator("SV"))
observable = fq.Observable([("ZI", 1.0)])
results = estimator.run_sweep(
    qnn,
    observable,
    {features: X, weights: weight_batch},
).result()
expectations = np.array([result.get_expectation() for result in results])
```

For multiple observables, every `Result` contains an expectation array with
the same length as the observable sequence.

## Version 1 execution behavior

Version 1 validates the complete binding batch, then binds and calls the
existing `run()` once per row. It makes no promise of one-time lowering,
parallel row execution, vectorized engines, or faster execution than the
equivalent explicit Python loop. No partial result list is exposed if a later
row fails.

An explicit seed is forwarded unchanged to every row. This exactly matches
manually repeating `run()` with that seed, but it also reuses the same
pseudorandom stream positions. Sampled row errors are correlated: do not treat
their standard errors as independent or use this seed behavior to model
finite-difference gradient noise.

A reusable prepared-program handle, parameter expressions, gradients, and
per-target broadcasting syntax are future work.
