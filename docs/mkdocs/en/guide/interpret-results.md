# Estimate observables

Use [`Estimator`][fatqat.Estimator] to calculate expectation values, such as
spin correlations or energies. Define the quantity with an
[`Observable`][fatqat.Observable], then supply a Program and a
[compatible simulator or emulator](../api/estimator.md#backend-support).
You can calculate the expectation directly from the state or estimate it
from a finite number of measurement shots.

The examples below use the general-purpose `Simulator` and two Bell states.
Adding a `Z` operation to the Bell circuit changes its relative phase while
leaving its computational-basis probabilities unchanged:

```pycon
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> bell = fq.Program(2)
>>> bell.add(ops.H, 0)
>>> bell.add(ops.CX, (0, 1))
>>> phase_flipped = bell.copy()
>>> phase_flipped.add(ops.Z, 0)
>>> backend = fq.simulator.Simulator(method="statevector", runtime="numpy")
>>> state = backend.run(bell).result().get_statevector()
>>> flipped_state = backend.run(phase_flipped).result().get_statevector()
>>> np.round(np.abs(state) ** 2, 6).tolist()
[0.5, 0.0, 0.0, 0.5]
>>> np.round(np.abs(flipped_state) ** 2, 6).tolist()
[0.5, 0.0, 0.0, 0.5]
```

Both states give `00` or `11` with equal probability. Their state vectors are
\((|00\rangle + |11\rangle)/\sqrt{2}\) and
\((|00\rangle - |11\rangle)/\sqrt{2}\). To distinguish them, we need to
measure in another basis.

## Calculate correlations

A Pauli measurement along the X or Z axis has outcomes `+1` and `-1`.
For Z, `|0>` corresponds to `+1` and `|1>` to `-1`.
`ZZ` measures the product of the two qubits' Z outcomes; `XX` does the same
along X. An expectation of `+1` means the outcomes always agree, while `-1`
means they are always opposite.

Pass a list of observables to calculate both correlations in one request:

```pycon
>>> zz = fq.Observable([("ZZ", 1.0)])
>>> xx = fq.Observable([("XX", 1.0)])
>>> estimator = fq.Estimator(backend)
>>> bell_result = estimator.run(bell, [zz, xx], shots=0).result()
>>> flipped_result = estimator.run(phase_flipped, [zz, xx], shots=0).result()
>>> np.round(bell_result.get_expectation(), 6).tolist()
[1.0, 1.0]
>>> np.round(flipped_result.get_expectation(), 6).tolist()
[1.0, -1.0]
```

The results follow the order `[zz, xx]`. Both states have `ZZ = 1`: their Z
outcomes always agree. The `XX` values distinguish them. X outcomes always
agree for the original Bell state and are always opposite for the
phase-flipped state.

Each tuple in an `Observable` contains a Pauli label and its real coefficient.
Labels follow qubit order from left to right, and `I` is the identity:
`ZI` measures Z on qubit 0 alone. To calculate a weighted sum, such as an
energy, include its terms in the same `Observable`. The
[Estimator API](../api/estimator.md#construct-an-observable) describes the
other ways to write an observable.

`Estimator.run` uses `shots=0` by default, which calculates the expectation
directly from the state.

## Estimate from samples

Set `shots` to a positive integer to estimate an expectation from sampled
measurements. For `ZI` on the original Bell state, each shot gives `+1` or
`-1` with equal probability. The exact expectation is zero, but a finite
sample usually gives a small nonzero mean:

```pycon
>>> zi = fq.Observable([("ZI", 1.0)])
>>> exact = estimator.run(bell, zi, shots=0).result()
>>> round(float(exact.get_expectation()), 4)
0.0
>>> sampled = estimator.run(
...     bell,
...     zi,
...     shots=400,
...     simulation_config={"seed": 7},
... ).result()
>>> round(float(sampled.get_expectation()), 4)
-0.045
>>> round(float(sampled.get_standard_error()), 4)
0.05
```

For a single observable, `get_expectation()` and `get_standard_error()` each
return a scalar. This run gives an estimate of `-0.045` with a standard error
of about `0.05`. The standard error describes the expected spread of estimates
if you repeated the run with the same number of shots.

Increase `shots` for better precision. The plot below shows estimates from
100 to 6,400 shots, with error bars extending one standard error above and
below each estimate:

![Estimates of ZI are plotted against increasing shot counts with one-standard-error bars that shrink around the exact expectation of zero.](../assets/generated/guide/observable-uncertainty.png)

??? example "Reproduce this figure"

    ```python
    import matplotlib.pyplot as plt
    import fatqat as fq
    import fatqat.operations as ops

    bell = fq.Program(2)
    bell.add(ops.H, 0)
    bell.add(ops.CX, (0, 1))
    estimator = fq.Estimator(
        fq.simulator.Simulator(method="statevector", runtime="numpy")
    )
    zi = fq.Observable([("ZI", 1.0)])
    exact = estimator.run(bell, zi, shots=0).result().get_expectation()

    shot_counts = [100, 400, 1600, 6400]
    estimates = []
    standard_errors = []
    for index, shots in enumerate(shot_counts):
        result = estimator.run(
            bell, zi, shots=shots, simulation_config={"seed": 7 + index}
        ).result()
        estimates.append(result.get_expectation())
        standard_errors.append(result.get_standard_error())

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.errorbar(
        shot_counts, estimates, yerr=standard_errors,
        fmt="o", capsize=4, label="Estimate with one standard error",
    )
    axis.axhline(exact, color="C1", linestyle="--", label="Exact expectation")
    axis.set(xscale="log", xlabel="Shots", ylabel=r"$\langle Z_0 \rangle$")
    axis.set_xticks(shot_counts, [str(shots) for shots in shot_counts])
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    plt.show()
    ```

Quadrupling the shots roughly halves the standard error. Individual estimates
still fluctuate, so a larger sample can give a value farther from zero.
The exact value can also fall outside a one-standard-error bar.

Circuit noise can change the expectation itself. Configure noise on the
estimator's backend as described in [Ideal and noisy runs](ideal-and-noisy.md).
A density-matrix backend can calculate the expectation with noise channels
directly; increasing the shot count improves the precision of a sampled
estimate of that noisy expectation.

See the [Estimator API](../api/estimator.md) for supported backends, observable
construction, and run options.
