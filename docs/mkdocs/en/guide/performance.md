# Performance and scaling

Before tuning runtimes, estimate how much state FatQat must carry. Subsystem
dimensions and the chosen representation usually matter sooner than a runtime
switch.

First choose the simplest execution level that contains the effect you need;
[Choose how much physics to model](execution-models.md) draws that boundary.
Then use the estimates and benchmark pattern below on your own
[`Program`][fatqat.Program].

## Count the state space before running

If the local dimensions are `d0, d1, ...`, the state-space dimension is their
product. This makes FatQat's mixed-dimensional Program support useful and also
makes its scaling explicit:

```pycon
>>> import math
>>> local_dimensions = (2, 2, 2, 2, 3, 3)  # four qubits and two qutrits
>>> dimension = math.prod(local_dimensions)
>>> dimension
144
>>> dimension**2
20736
```

A statevector stores one complex entry per basis state. A density matrix or
unitary stores a square array, and a super-operator is square in the already
squared density-matrix space. In terms of total dimension `D`, their entry
counts scale as `D`, `D**2`, `D**2`, and `D**4`, respectively. Temporary work
space and backend overhead add to these lower-level counts.

The growth is exponential in both subsystem count and local dimension:

![Logarithmic curves show statevector and density-matrix entry counts growing faster for qutrits than for qubits as subsystem count increases.](../assets/generated/guide/performance-1.png)

??? example "Reproduce this figure"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    subsystems = np.arange(1, 9)
    qubit_state = 2 ** subsystems
    qutrit_state = 3 ** subsystems
    qubit_density = qubit_state ** 2
    qutrit_density = qutrit_state ** 2

    assert np.all(np.diff(qubit_state) > 0)
    assert np.all(np.diff(qutrit_density) > 0)

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.semilogy(
        subsystems,
        qubit_state,
        marker="o",
        label="qubit statevector",
    )
    ax.semilogy(
        subsystems,
        qubit_density,
        marker="o",
        label="qubit density matrix",
    )
    ax.semilogy(
        subsystems,
        qutrit_state,
        marker="s",
        label="qutrit statevector",
    )
    ax.semilogy(
        subsystems,
        qutrit_density,
        marker="s",
        label="qutrit density matrix",
    )
    ax.set(
        xlabel="number of equal-dimension subsystems",
        ylabel="complex array entries",
        xticks=subsystems,
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, ncols=2, fontsize="small")
    fig.tight_layout()
    ```

This plot counts entries rather than bytes so it does not assume a dtype or
allocator. Estimate the Program you actually intend to run, including physical
levels that an emulator models even when the logical Program addresses only
qubits.

## Request only the answer you need

Result choice can dominate scaling. A complete unitary or super-operator asks
for the action on every input, while a state run asks for one input state.
Likewise, a full state is unnecessary when you only need counts or a few
expectation values.

Shot cost also depends on the Program. A circuit that evolves deterministically
and measures only at the end can reuse more work than one with mid-circuit
measurement, reset, feedforward, or stochastic trajectories. Avoid estimating
shot cost from `shots` alone; benchmark the same Program and result request you
will use in practice.

See [Ask questions of a run](interpret-results.md) for choosing an answer and
the [Simulator API](../api/simulator.md) for exact method and result
constraints.

## Compare NumPy and Numba on your workload

FatQat's general simulator offers two CPU runtimes:

- NumPy executes directly and avoids JIT compilation startup.
- Numba compiles numerical kernels on first use and can reuse compatible
  compiled work on later calls.

Neither choice changes the Program or the modeled mathematics. Compilation,
array-library behavior, CPU, operating system, Program shape, and repetition
count all affect the result, so benchmark rather than assuming one runtime is
always preferable. FatQat does not currently provide a GPU runtime.

The following harness separates one untimed warm-up from repeated measurements
and compares like-for-like final states:

```python
import statistics
import time

import numpy as np
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(8)
for _ in range(6):
    for target in range(8):
        program.add(ops.RY(0.17), target)
    for control in range(7):
        program.add(ops.CX, (control, control + 1))

result_config = {"counts": False, "final_state": True}
numpy_backend = fq.simulator.Simulator("SV", runtime="numpy")
numba_backend = fq.simulator.Simulator("SV", runtime="numba")

def warm_and_measure(backend, repeats=7):
    # Keep compilation and other first-use setup outside steady-state samples.
    backend.run(program, result_config=result_config).result()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        backend.run(program, result_config=result_config).result()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)

numpy_state = numpy_backend.run(program, result_config=result_config).result()
numba_state = numba_backend.run(program, result_config=result_config).result()
assert np.allclose(
    numpy_state.get_statevector(),
    numba_state.get_statevector(),
)

numpy_seconds = warm_and_measure(numpy_backend)
numba_seconds = warm_and_measure(numba_backend)
print({"numpy": numpy_seconds, "numba": numba_seconds})
```

Treat the printed values as local evidence, not package guarantees. If startup
latency matters, measure the first run separately instead of discarding it.
If sustained throughput matters, increase the repetition count and Program
size to match the intended workload.

## Tune parallelism and fusion only after measuring

Automatic execution settings are the appropriate baseline. Manual shot
parallelism, kernel parallelism, worker limits, and operation fusion apply only
to compatible workloads, and overhead can outweigh saved numerical work.

When tuning:

1. Keep the Program, method, runtime, seed, and result request fixed.
2. Warm any compiled path before measuring steady state.
3. Measure several repetitions and report a robust statistic such as the
   median.
4. Change one execution choice at a time.
5. Verify the resulting state, counts distribution, or observable before
   accepting the timing.
6. Repeat at the problem sizes that matter; a small example may rank choices
   differently from the target workload.

For eligible combinations and error behavior, see
[Simulator runtime and execution](../api/simulator.md). Fusion is opt-in,
and explicit parallel modes can be rejected when the Program cannot use them.

## Account for physical emulation separately

Hamiltonian emulators add costs that a circuit-level array-size estimate does
not capture: the model's physical levels, unaddressed modeled subsystems,
time-dependent controls, scheduling, integration intervals, and open-system
evolution. A logical qubit may therefore contribute three physical levels in a
transmon or three-level atom model.

Benchmark the actual model, arrangement, controls, duration, solver settings,
and requested result. Do not extrapolate an emulator run from a gate-level
Simulator timing. The [emulator API](../api/emulators/index.md) lists the
solver and schedule controls available to each physical model.
