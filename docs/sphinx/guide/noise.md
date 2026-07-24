# Noise

A {py:class}`~fatqat.NoiseModel` describes noise separately from a
{py:class}`~fatqat.Program`. This lets you run the same program ideally or
noisily by choosing a different backend.

## A first noisy experiment

```python
import fatqat as fq

program = fq.Program(2, 2)
program.add(fq.ops.H, 0)
program.add(fq.ops.CX, (0, 1))
program.add_measurement((0, 1), (0, 1))

noise = fq.NoiseModel()
noise.add_noise(fq.ops.CX, fq.noise.Depolarizing(p=0.05))

ideal = fq.backends.SimulatorBackend(method="density_matrix")
noisy = fq.backends.SimulatorBackend(method="density_matrix", noise=noise)

print(ideal.run(program, shots=1000, seed=7).result().get_counts())
print(noisy.run(program, shots=1000, seed=7).result().get_counts())
```

The {py:meth}`add_noise <~fatqat.NoiseModel.add_noise>` call attaches noise immediately
after every ``CX`` operation. The noisy counts
can include outcomes that the ideal Bell-state program does not produce.

## Choose a simulation method

| Method | Good default use | Trade-off |
| --- | --- | --- |
| `method="density_matrix"` | Exact noisy distributions and density matrices. | A density matrix uses more memory. |
| `method="statevector"` or the default | Larger ideal programs, or sampled noisy trajectories. | Channel noise samples a branch per shot, so noisy execution is stochastic. |

For a noisy statevector run, request a final state only with `shots=1`.
Use [Running and results](running-and-results.md) to request a statevector
or density matrix explicitly.

## Built-in quantum channels

Attach a channel with {py:meth}`add_noise <~fatqat.NoiseModel.add_noise>` (``operation, channel``):

- {py:class}`~fatqat.noise.Depolarizing` (``p``) mixes the affected system with the maximally mixed
  state.
- {py:class}`~fatqat.noise.AmplitudeDamping` (``gammas=(gamma,)``) models qubit relaxation toward
  `|0⟩`. A higher-dimensional system needs one rate per adjacent-level
  transition.
- {py:class}`~fatqat.noise.PhaseDamping` (``p``) removes coherence without changing populations.

More than one call for the same gate adds independent mechanisms in the
order they were registered:

```python
noise = fq.NoiseModel()
noise.add_noise(fq.ops.H, fq.noise.AmplitudeDamping(gammas=(0.01,)))
noise.add_noise(fq.ops.H, fq.noise.PhaseDamping(p=0.02))
```

{py:func}`~fatqat.noise.relaxation_channels` (``t1, t2, duration``) builds the damping and
dephasing pair for a qubit gate with a known duration. Its API reference
documents the units and physical bounds.

## Target one qubit

Without `targets=`, a channel applies to every occurrence of its operation.
For a portable application program, target one of the program’s qubit
references:

```python
program = fq.Program(2, 2)
noise = fq.NoiseModel()
noise.add_noise(
    fq.ops.X,
    fq.noise.Depolarizing(p=0.1),
    targets=(program.qreg[0][0],),
)
```

This names the first quantum slot in the program, independent of how a
backend executes it. Backend-specific physical addressing is intentionally
outside the normal application workflow.

## Readout error

Readout error changes the reported classical value after measurement; it
does not change the underlying post-measurement quantum state. Supply a
column-stochastic confusion matrix where `C[reported, true]` is the
probability of reporting one value when another was measured:

```python
import numpy as np

noise = fq.NoiseModel()
noise.add_readout_error(
    np.array([[0.98, 0.05],
              [0.02, 0.95]]),
)
```

The {py:meth}`add_readout_error <~fatqat.NoiseModel.add_readout_error>` call applies this matrix.
Here a true ``0`` is reported as ``0`` with probability 0.98, while a true ``1``
is reported as `1` with probability 0.95. The error applies to every
measurement unless you pass a qubit reference.

The {doc}`noise API reference <../api/noise>` lists all supported
descriptors and their parameters.
