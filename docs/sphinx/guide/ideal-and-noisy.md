# Compare the same Program ideally and noisily

Start with an ideal run, then change only the execution model. A
{py:class}`~fatqat.NoiseModel` describes errors and where they act; a backend
decides how to realize them. The {py:class}`~fatqat.Program` remains unchanged,
so the difference between the two runs has a clear cause. Here, a measured Bell
Program picks up both operation noise and readout confusion.

## Establish an ideal baseline

Build the computation before configuring the ideal and noisy backends:

```{doctest}
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> bell = fq.Program(2, 2)
>>> bell.add(ops.H, 0)
>>> bell.add(ops.CX, (0, 1))
>>> bell.measure_all()
>>> ideal_backend = fq.simulator.Simulator(
...     method="density_matrix",
...     runtime="numpy",
... )
>>> ideal_counts = ideal_backend.run(
...     bell,
...     shots=4_000,
...     simulation_config={"seed": 7},
... ).result().get_counts()
>>> ideal_counts.get("01", 0) + ideal_counts.get("10", 0)
0
```

An ideal Bell run has no wrong-parity outcomes: its two classical digits
always agree. The split between `00` and `11` still fluctuates because
measurement is sampled.

## Change the execution, not the Program

Add a finite channel after `CX` and classical confusion at measurement, then
construct another backend:

```{doctest}
>>> noise = fq.NoiseModel()
>>> noise.add(
...     fq.noise.Depolarizing(p=0.12),
...     operation=ops.CX,
... )
>>> noise.add(
...     fq.noise.ReadoutConfusion(
...         [[0.98, 0.04], [0.02, 0.96]]
...     )
... )
>>> noisy_backend = fq.simulator.Simulator(
...     method="density_matrix",
...     runtime="numpy",
...     noise=noise,
... )
>>> noisy_counts = noisy_backend.run(
...     bell,
...     shots=4_000,
...     simulation_config={"seed": 7},
... ).result().get_counts()
>>> noisy_errors = noisy_counts.get("01", 0) + noisy_counts.get("10", 0)
>>> noisy_errors > 0
True
>>> sum(noisy_counts.values())
4000
```

The depolarizing channel changes the quantum state after the entangling gate.
The confusion matrix changes only the reported classical digit: in this
example, a true `0` is reported as `1` two percent of the time, while a true
`1` is reported as `0` four percent of the time. Both effects can create the
wrong-parity bars below.

```{eval-rst}
.. plot::
   :alt: Side-by-side Bell-state histograms show only zero-zero and one-one ideally, while the noisy run also contains zero-one and one-zero outcomes.
   :include-source: false

   import numpy as np
   import matplotlib.pyplot as plt
   import fatqat as fq
   import fatqat.operations as ops

   bell = fq.Program(2, 2)
   bell.add(ops.H, 0)
   bell.add(ops.CX, (0, 1))
   bell.measure_all()

   noise = fq.NoiseModel()
   noise.add(fq.noise.Depolarizing(p=0.12), operation=ops.CX)
   noise.add(
       fq.noise.ReadoutConfusion(
           [[0.98, 0.04], [0.02, 0.96]]
       )
   )

   ideal_backend = fq.simulator.Simulator(
       method="density_matrix", runtime="numpy"
   )
   noisy_backend = fq.simulator.Simulator(
       method="density_matrix", runtime="numpy", noise=noise
   )
   shots = 4_000
   run_options = {"shots": shots, "simulation_config": {"seed": 7}}
   ideal = ideal_backend.run(bell, **run_options).result().get_counts()
   noisy = noisy_backend.run(bell, **run_options).result().get_counts()

   labels = ["00", "01", "10", "11"]
   ideal_frequency = np.array([ideal.get(label, 0) for label in labels]) / shots
   noisy_frequency = np.array([noisy.get(label, 0) for label in labels]) / shots

   assert ideal.get("01", 0) + ideal.get("10", 0) == 0
   assert noisy.get("01", 0) + noisy.get("10", 0) > 0

   x = np.arange(len(labels))
   width = 0.36
   fig, ax = plt.subplots(figsize=(6.4, 3.5))
   ax.bar(
       x - width / 2,
       ideal_frequency,
       width,
       label="ideal",
       color="#3b6ea8",
   )
   ax.bar(
       x + width / 2,
       noisy_frequency,
       width,
       label="noisy",
       color="#d17a3a",
   )
   ax.set(
       xlabel="reported outcome",
       ylabel="frequency",
       xticks=x,
       xticklabels=labels,
       ylim=(0.0, 0.58),
   )
   ax.legend(frameon=False)
   ax.grid(axis="y", alpha=0.25)
   fig.tight_layout()
```

The comparison is controlled because both backends receive the same `bell`
object, shot count, and seed. Only the execution model changes.

## Density matrices and sampled trajectories

The density-matrix method applies supported finite channels as an exact mixed
evolution before measurement. Counts are still sampled because they describe
individual reported outcomes.

A statevector backend instead samples a channel trajectory when stochastic
noise reaches the Program. That can use less state storage, but each shot
represents one branch rather than the exact ensemble. Use density matrix when
the exact noisy state or expectation value is the answer; use statevector
trajectories when sampling branches is part of the intended study. The two
approaches should agree statistically on repeated measurement outcomes, not
shot for shot.

## Circuit channels and continuous noise

| Execution level | Noise description | Where it acts |
| --- | --- | --- |
| Circuit simulator | finite probabilities and channels | at matched operation boundaries |
| Physical emulator | rates and relaxation times | throughout elapsed Hamiltonian/Lindblad evolution |

FatQat does not invent a gate duration to convert between the two. Move to
[Hamiltonian-level emulation](hamiltonian-emulation.md) for pulse duration,
idle evolution, leakage, or continuous-time noise. For supported combinations,
selectors, and validation rules, see the {ref}`noise-backend-support` table
and {doc}`Noise model API <../api/noise/model>`.
