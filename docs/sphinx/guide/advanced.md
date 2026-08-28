# Advanced

Use this page when you need qudits, custom operation implementations, or
explicit execution controls.

## Qudits

Registers with `dim > 2` are qudits and use the same `Program` and backend API
as qubits. {py:class}`~fatqat.operations.Shift`,
{py:class}`~fatqat.operations.Clock`, and {py:data}`~fatqat.operations.Sum`
generalize the qubit `X`/`Z`/`CX` gates rather than requiring separate
handling. See [Gates](gates.md) for the full qudit gate list.

Registers of different dimensions can also coexist in the same program.
You can operate on each register independently, and operations such as
{py:class}`~fatqat.operations.CClock` can couple targets of different
dimensions:

```python
import fatqat as fq
import fatqat.operations as ops

qubit = fq.QuantumRegister(1, name="qubit")            # dim=2
qutrit = fq.QuantumRegister(1, dim=3, name="qutrit")
cbit = fq.ClassicalRegister(1, name="c_qubit")
ctrit = fq.ClassicalRegister(1, dim=3, name="c_qutrit")

program = fq.Program([qubit, qutrit], [cbit, ctrit])
program.add(ops.X, qubit[0])            # qubit: |0> -> |1>
program.add(ops.Shift(1), qutrit[0])     # qutrit: |0> -> |1>
program.add(ops.CClock(1), (qubit[0], qutrit[0]))
program.measure(qubit[0], cbit[0])
program.measure(qutrit[0], ctrit[0])

result = fq.simulator.Simulator("SV").run(program, shots=100).result()
print(result.get_counts())   # {"11": 100}
```

Each target carries its register's dimension, so use the register references
directly when adding gates or measurements. See [Gates](gates.md) for target
ordering and the available qudit gates.

## Custom implementations

Pass a custom {py:class}`~fatqat.implementation.MatrixImplementationMap` as
{py:class}`~fatqat.simulator.Simulator`'s `implementation_map=` argument when
you need to supply a matrix for your own operation class.

### Example: adding a new fixed-matrix gate

Start from `default_matrix_implementation_map()` so built-in gates keep
working, then register your operation on top of it. This example defines a
square-root-of-Y gate:

```python
import numpy as np
import fatqat as fq
import fatqat.operations as ops
from fatqat.implementation import FixedMatrix, default_matrix_implementation_map


class SqrtY(ops.Operation):
    name = "SqrtY"
    num_subsystems = 1


SQRT_Y_MATRIX = 0.5 * (1 + 1j) * np.array(
    [[1, -1], [1, 1]],
    dtype=complex,
)

implementation_map = default_matrix_implementation_map()
implementation_map.add(SqrtY, FixedMatrix(SQRT_Y_MATRIX))

backend = fq.simulator.Simulator("SV", implementation_map=implementation_map)

program = fq.Program(1)
program.add(SqrtY(), 0)
program.add(SqrtY(), 0)   # two SqrtY == one Y

statevector = (
    backend.run(program, result_config={"counts": False, "final_state": True})
    .result()
    .get_statevector()
)
print(statevector)   # [0, 1j]
```

For a multi-subsystem custom operation, preserve the ordered target tuple when
defining matrix factors. For two qubit targets `(t0, t1)`, the local basis is
`|00>, |01>, |10>, |11>`: the last target changes fastest. See
{py:meth}`~fatqat.implementation.MatrixImplementationMap.add` for the exact
mixed-radix contract and [Running and results](running-and-results.md) for the
full-system basis order.

A bare `np.ndarray` is accepted in place of `FixedMatrix`, so the registration
above could instead be written as
`implementation_map.add(SqrtY, SQRT_Y_MATRIX)`. The custom operation class
declares its display name and fixed number of targets. FATQAT checks the matrix
against those targets' dimensions when the program runs.

If the matrix depends on operation parameters or target dimensions, register a
callable instead of a fixed matrix. See
{py:meth}`~fatqat.implementation.MatrixImplementationMap.add` for the accepted
callable signatures.

### Custom pulse implementations

Pulse emulators use a `PulseImplementationMap` where `Simulator` uses a
`MatrixImplementationMap`. Its rules return `PulseDefinition` values instead
of matrices. See {doc}`../api/pulse-control/gate-realization` for the pulse
rule contract and the emulator API pages for their built-in maps.

## Execution configuration

The matrix simulator separates two kinds of parallel work: independent sampled
executions belong to **shot parallelism**, while numerical work within one
evolution belongs to **kernel parallelism**. Kernel parallelism never overlaps
ordered program operations or executes them out of order.

The defaults are:

```python
simulation_config = {
    "seed": None,
    "shot_parallelism": "auto",
    "kernel_parallelism": "auto",
    "max_workers": None,
    "fusion": False,
}
```

Leave both parallelism settings at `"auto"` unless you have measured a reason
to choose. FATQAT uses at most one parallel axis.

`shot_parallelism` controls independent sampled executions. `"threads"` and
`"processes"` are available only for eligible counts-only runs that need a
separate evolution for each shot, such as programs with mid-program
measurement, reset, conditions, or stochastic channels. They require at least
two shots and two workers. Threaded shots require a Numba statevector run and
do not support the atom occupancy lifecycle; use processes for other eligible
methods or programs. A circuit that evolves once and samples only at the end
is not eligible for explicit shot parallelism.

`kernel_parallelism` controls numerical work within one evolution. Its values
are `"auto"`, `"serial"`, and `"threads"`; threads require Numba. You cannot
request threaded kernels together with threaded or process shots.

`max_workers` is `None` or a positive integer. It limits the chosen parallel
mode but cannot make an ineligible run parallel. Setting it to `1` is useful
with automatic selection to keep a run serial, but conflicts with an explicit
parallel request. An explicit mode that the runtime or program cannot support
raises an error rather than silently falling back.

For example:

```python
# Fully serial.
fully_serial = {
    "shot_parallelism": "serial",
    "kernel_parallelism": "serial",
}

# Use processes for an eligible counts-only workload.
parallel_shots = {
    "shot_parallelism": "processes",
    "kernel_parallelism": "serial",
    "max_workers": 4,
}
```

Parallel overhead can make a small workload slower, so benchmark the program
and runtime you intend to use.

### Fusion

`fusion` is independent of parallelism and defaults to `False`. `True` can
combine compatible adjacent operations for Numba density-matrix, unitary, and
super-operator runs. NumPy and Numba statevector reject it. Fusion is not
always faster, so benchmark the intended workload.

### Seed reproducibility

A fixed non-negative `seed`, together with the same program, complete
configuration, FATQAT version, and execution environment, reproduces sampled
results. `None` uses fresh entropy. A negative integer is rejected when
execution starts, so the returned job is failed and `Job.result()` raises
`ValueError`. Changing the runtime or execution mode can change the order in
which random values are consumed, so the same seed alone does not promise
identical samples across those changes. Deterministic results follow the normal
numerical tolerance and do not depend on the seed.
