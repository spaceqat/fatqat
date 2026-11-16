# Performance and scaling

Simulation cost depends on the state representation, requested results, and
how you organize repeated runs. This guide covers these choices for
[`Simulator`][fatqat.simulator.Simulator] and [`Estimator`][fatqat.Estimator].

## Choose the state and results you need

For subsystem dimensions `d0, d1, ...`, the total state-space dimension is
`D = d0 * d1 * ...`. Each extra qubit doubles `D`; each extra qutrit triples
it. The simulation method determines the number of entries you store:

| Method | Complex entries | Use it to |
| --- | --- | --- |
| `statevector` | `D` | Evolve a pure state or sample noisy trajectories |
| `density_matrix` | `D**2` | Evolve a mixed state and apply noise channels directly |
| `unitary` | `D**2` | Find the circuit's action on every input state |
| `superop` | `D**4` | Find the full channel acting on density matrices |

For example, 20 qubits need about 16 MiB for a complex128 statevector alone.
A density matrix of the same system needs about 16 TiB. Temporary arrays add
to these storage costs. For noisy simulations, this makes statevector
trajectories useful at sizes where a density matrix is too large; the
tradeoff is sampling uncertainty.

Choose results to match the question. Use counts for a measurement
distribution, an Estimator for correlations or energies, and a final state
when you need the quantum state itself. A counts-only run can use
`result_config={"counts": True, "final_state": False}`. This avoids retaining
a final quantum array in the result, although simulation still needs a state
while it runs.

Specialized representations can save much more than runtime tuning. For
linear optics, `LOSimulator` uses fixed-photon states by default: four photons
in twelve modes require just 1,365 amplitudes. See
[Simulate larger interferometers](linear-optics.md#simulate-larger-interferometers).
For physical models, first [choose how much physics to model](execution-models.md);
extra physical levels and pulse integration change the cost of the problem.

## Compare NumPy and Numba on your workload

FatQat offers two CPU runtimes. NumPy starts executing directly, which makes
it useful when compilation startup would dominate the workload. Numba
compiles numerical kernels on first use and spreads that startup cost over
repeated work. The initial simulator default is Numba; set `runtime`
explicitly when comparing them.

The runtime and the execution workflow serve different roles. A **kernel**
does numerical work, such as applying a gate to a state. A **workflow**
sequences those operations and handles repeated shots or parameter points.
NumPy uses Python workflows with NumPy numerical operations. Numba uses
compiled kernels, with compiled workflows for eligible statevector
trajectories and parameter batches.

For statevector simulation, choosing `runtime="numba"` does not require the
whole program to fit a compiled workflow. If an otherwise supported operation
cannot run in that workflow, the simulator uses a Python workflow with Numba
kernels for supported numerical operations. Custom Python callbacks still
execute in Python. Python workflows also support process-based shots and
threading within kernels.

For repeated runs, parameter sweeps, or many independent shots, start with
Numba and automatic execution settings. Even small circuits can benefit from
compiled workflows and shot parallelism when each shot evolves a separate
noisy trajectory or follows measurement-dependent control flow.

Workflow selection is automatic: you choose the runtime and parallelism
options, and the simulator selects the workflow. For parallelism choices, see
[Match parallelism to the work](#match-parallelism-to-the-work).

The following example compares warmed execution times and checks that both
runtimes agree within floating-point tolerance. Times are in seconds.

```python
import statistics
import time

import numpy as np
import fatqat as fq
import fatqat.operations as ops


def median_seconds(run, repeats=5):
    run()  # Warm up compilation and other first-use setup.
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


program = fq.Program(8)
for _ in range(6):
    for target in range(8):
        program.add(ops.RY(0.17), target)
    for control in range(7):
        program.add(ops.CX, (control, control + 1))

state_only = {"counts": False, "final_state": True}
states = {}
for runtime in ("numpy", "numba"):
    backend = fq.simulator.Simulator("statevector", runtime=runtime)
    run = lambda: backend.run(program, result_config=state_only).result()
    print(runtime, median_seconds(run))
    states[runtime] = run().get_statevector()

assert np.allclose(states["numpy"], states["numba"])
```

Measure a representative circuit with the same method and result request.
For startup latency, time the first call in a fresh Python process separately.

## Sweep parameters without rebuilding

Parameter scans and variational algorithms repeatedly run the same circuit
with different angles. Express those angles as `Parameter` objects and pass
the values to `run_sweep()`: the simulator can share preparation across the batch.

Requesting final states retains one state per parameter point. Use an
estimator sweep when you only need energies or correlations. See
[Sweep without rebuilding](simulation.md#sweep-without-rebuilding) for a
worked example.

## Evaluate observables together

An Estimator computes expectation values directly from structured
[`Observable`][fatqat.Observable] objects. For matrix simulation, it evaluates
their local factors without constructing a dense operator over the full
system. Pass related observables together so they share the state evolution.

`shots=0` calculates expectations without shot sampling. Positive shots
introduce measurement sampling and give a standard error alongside each estimate. For exact
expectations with stochastic channels or reset, choose a density-matrix
backend; use positive shots for noisy statevector trajectories.

`Estimator.run_sweep()` retains expectation results instead of a final state
for every point, saving memory in large scans. See
[Estimate observables](interpret-results.md) for examples and uncertainty,
and the [Estimator API](../api/estimator.md#parameter-sweeps) for sweeps.

## Match parallelism to the work

Start with automatic execution settings. For explicit control, set these
options in `execution_config`:

| Workload | Parallelism |
| --- | --- |
| General use | Keep the default `"auto"` settings; use `max_workers` to cap the worker count |
| Large state with Numba | `kernel_parallelism="threads"`, `shot_parallelism="serial"` |
| Independent shots, counts only | `shot_parallelism="threads"` for compatible Numba statevector runs, or `"processes"` for other supported workloads; keep kernel parallelism serial |
| Little total work, or a serial timing baseline | Set both `kernel_parallelism` and `shot_parallelism` to `"serial"` |

A deterministic circuit with terminal measurements evolves once and samples
its output distribution. Increasing its shot count does not repeat the whole
circuit, so explicit shot parallelism does not apply to that workload.

Parallelism pays off when there is enough work across state entries or
independent shots to outweigh worker overhead. Choose one parallel axis at a
time. The [Simulator API](../api/simulator.md#runtime-and-execution) lists the supported
combinations.

When comparing sampled runs, set `execution_config["seed"]` and compare
observables or distributions: changing runtime or parallel mode can change
the individual samples. For deterministic runs, compare final states within
floating-point tolerance.
