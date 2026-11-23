---
title: "Simulator"
---

# Simulator

[`Simulator`][fatqat.simulator.Simulator] runs a [`Program`][fatqat.Program] at
the gate level. It supports qubits, qudits, and registers whose members have
different dimensions. It executes the operations as written, without imposing
a device topology.

## Choose a simulation method { #methods }

Use `statevector` or `density_matrix` to follow the evolution of a state. Use
`unitary` or `superop` to compute the action of the entire program on an
arbitrary input. In the table, `D` is the product of the subsystem dimensions.

| Method | What it computes | Result shape |
| --- | --- | --- |
| `statevector` (default) | A final pure state. Reset and noise channels may sample a trajectory. | `(D,)` |
| `density_matrix` | A final density matrix. Reset and noise channels act directly on the density matrix. | `(D, D)` |
| `unitary` | The matrix of a program made up of unitary operations. | `(D, D)` |
| `superop` | The linear map on density matrices, including reset and finite noise channels. | `(D**2, D**2)` |

Both state methods support measurement and classical conditions. Both operator
methods require a program without measurements or classical conditions, and
return the map without taking an initial state or producing counts. Their
memory use grows with the size of the returned matrix.

To apply a `superop` to a density matrix `rho`, flatten it by stacking its
columns and reshape the result in the same order.

```python
rho_out = (superop @ rho.reshape(-1, order="F")).reshape(rho.shape, order="F")
```

For a program of unitary operations, the superoperator is
`numpy.kron(unitary.conj(), unitary)`.

See [Result](result.md#ordering-and-mutable-values) for the basis order used
by states and operators.

<a id="quick-start"></a>

## Run and read results { #run-inputs-and-results }

[`run()`][fatqat.simulator.Simulator.run] accepts a `Program` or an
[`ExecutableProgram`][fatqat.ExecutableProgram] that carries its own resource
layout. It returns a completed job. This example reads the measurement counts
from a Bell-state program.

```pycon
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> bell = fq.Program(2, 2)
>>> bell.add(ops.H, 0)
>>> bell.add(ops.CX, (0, 1))
>>> bell.measure_all()
>>> simulator = fq.simulator.Simulator(runtime="numpy")
>>> counts = simulator.run(
...     bell, shots=1000, execution_config={"seed": 42}
... ).result().get_counts()
>>> sorted(counts)
['00', '11']
>>> sum(counts.values())
1000
```

A program with measurements returns counts by default. A run with a
deterministic final state or operator returns that result by default. To inspect
the final state of a single sampled trajectory, request
`result_config={"final_state": True}` with `shots=1`. See [Result](result.md)
for the available fields and their accessors.

The simulator starts in the ground state by default. Pass a
[`BasisState`][fatqat.simulator.BasisState] as `initial_state` to select a
level for each subsystem by register reference.

<a id="sweeps"></a>

Use [`run_sweep()`][fatqat.simulator.Simulator.run_sweep] to run a parameterized
program for several sets of values and read the results in the same order.

## Configure a run { #runtime-and-execution }

The constructor's `runtime` selects NumPy or Numba for numerical execution.
If you omit it, new simulators use the current default, initially Numba.
Use [`set_default_runtime()`][fatqat.simulator.set_default_runtime] to change
the default for new simulators.

Pass `execution_config` to [`run()`][fatqat.simulator.Simulator.run] to set a
sampling seed, choose parallelism, or control error handling for that call.
The `run()` entry below lists the accepted keys, defaults, and restrictions.

With the automatic settings, the simulator uses at most one kind of parallelism,
either across shots or within one evolution. See
[Performance and scaling](../guide/performance.md#match-parallelism-to-the-work)
for examples of matching parallelism to a workload.

<a id="noise"></a>

Pass a [`NoiseModel`][fatqat.NoiseModel] through the constructor's `noise`
argument to use it on every run. Matrix simulation uses probability-form
channels applied at operation boundaries. See [Noise](noise.md#backend-support)
for supported noise sources.

<a id="customize-the-backend"></a>

Use `implementation_map` to supply your own matrices, channels, or
[custom operations](simulators/custom-operations.md).

## API reference { #api }

::: fatqat.simulator.Simulator
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: true
      filters:
        - "!^_"

::: fatqat.simulator.BasisState

::: fatqat.ExecutableProgram
    options:
      heading_level: 3

::: fatqat.simulator.get_default_runtime

::: fatqat.simulator.set_default_runtime
