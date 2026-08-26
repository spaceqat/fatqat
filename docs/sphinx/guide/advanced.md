# Advanced

Everything in the rest of the guide applies beyond the plain-qubit case.
This page covers three extension points: qudits, custom matrix
implementations, and execution strategies.

## Qudits

Registers with `dim > 2` are qudits, and use the same `Program` and backend
API as qubits — {py:class}`~fatqat.operations.Shift`,
{py:class}`~fatqat.operations.Clock`, and {py:data}`~fatqat.operations.Sum`
generalize the qubit `X`/`Z`/`CX` gates rather than requiring separate
handling. See [Gates](gates.md) for the full qudit gate list and a worked
example on qutrits.

Registers of different dimensions can also coexist in the same program.
There's no mixed qubit-qutrit gate yet, but a qubit register and a qutrit
register can each be driven independently with their own gates:

```python
import fatqat as fq
import fatqat.operations as op

qubit = fq.QuantumRegister(1, name="qubit")            # dim=2
qutrit = fq.QuantumRegister(1, dim=3, name="qutrit")
cbit = fq.ClassicalRegister(1, name="c_qubit")
ctrit = fq.ClassicalRegister(1, dim=3, name="c_qutrit")

program = fq.Program([qubit, qutrit], [cbit, ctrit])
program.add(op.X, program.quantum_registers[0][0])            # qubit: |0> -> |1>
program.add(op.Shift(1), program.quantum_registers[1][0])     # qutrit: |0> -> |1>
program.measure(program.quantum_registers[0][0], program.classical_registers[0][0])
program.measure(program.quantum_registers[1][0], program.classical_registers[1][0])

result = fq.simulator.Simulator("SV").run(program, shots=100).result()
print(result.get_counts())   # {"11": 100}
```

Each register keeps its own dimension; `add()` and `measure()`
resolve targets against whichever register a `RegisterRef` names, so mixed
dimensions require no special handling beyond addressing the right register
explicitly (see [Gates](gates.md) for how targets are addressed).

## Custom implementations

{py:class}`~fatqat.simulator.Simulator`'s `implementation_map=`
argument accepts a custom `MatrixImplementationMap` to control how
operations are lowered to matrices, or to add support for operations the
default map doesn't cover.

### Example: adding a new fixed-matrix gate

Start from the default map — `default_matrix_implementation_map()` — rather
than an empty `MatrixImplementationMap()`, so the built-in gates keep
working, then register your gate's matrix on top of it. This example adds a
`SqrtX` gate (the square root of `X`, i.e. applying it twice is equivalent to
`X`) that has no built-in equivalent:

```python
import numpy as np
import fatqat as fq
import fatqat.operations as op
from fatqat.implementation import FixedMatrix, default_matrix_implementation_map


class SqrtX(op.Operation):
    name = "SqrtX"
    _num_subsystems = 1


SQRT_X_MATRIX = 0.5 * np.array(
    [[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]],
    dtype=complex,
)

implementation_map = default_matrix_implementation_map()
implementation_map.add(SqrtX, FixedMatrix(SQRT_X_MATRIX))

backend = fq.simulator.Simulator("SV", implementation_map=implementation_map)

program = fq.Program(1)
program.add(SqrtX(), 0)
program.add(SqrtX(), 0)   # two SqrtX == one X

statevector = (
    backend.run(program, result_config={"counts": False, "final_state": True})
    .result()
    .get_statevector()
)
print(statevector)   # [0, 1]
```

A few things worth noting about the shape of this API:

- `add()` accepts a bare `np.ndarray` too — it auto-wraps into
  `FixedMatrix`, so `implementation_map.add(SqrtX, SQRT_X_MATRIX)` works
  just as well as wrapping it explicitly.
- A custom `Operation` subclass only needs `name` and `_num_subsystems`
  (the fixed arity the matrix implementation map requires — variable-arity
  operations aren't supported here).
- The matrix's shape is checked against the target dimensions at lowering
  time, so registering a 2x2 matrix for an operation applied to a qudit
  register raises a clear error rather than an opaque shape mismatch deeper
  in the engine.
- `implementation_map` is copied internally by the backend
  (`Simulator` never mutates the map you pass in), so the same map
  can be reused across multiple backend instances.

If the matrix instead depends on the operation's parameters (e.g. a rotation
angle) or on target dimension (e.g. a qudit gate), `add()` a bare callable —
`f(op)` or `f(op, targets)` — instead of a `FixedMatrix`; the map detects
which shape you passed by inspecting the callable's signature.

### Custom pulse implementations

`fq.emulator.TransmonEmulator` has the same customization shape for its own
native operations, through a `PulseImplementationMap` instead of an
`MatrixImplementationMap`: build
`default_transmon_gate_implementation_map(model=..., calibration=...)`,
register a replacement rule, and construct the backend with
`gate_implementation_map=`. See
[Superconducting pulse simulation](superconducting-pulse.md) for the
pulse-specific rule contract. A pulse rule receives `operation` and optionally
keyword-only `device_operands`; it returns a claim-free `PulseDefinition`
(duration, sampled controls, frame actions) rather than a matrix.

This public gate-map customization hook belongs to all three pulse emulators.
`Atom2LevelEmulator` has an empty built-in map, so custom rules can add ordinary
gate behavior without changing its global direct-control path. Direct
`PulseOperation` values on every system bypass gate lookup.

Pulse maps support a fixed definition, an operand-unaware callable registered
for explicit `device_operands`, or an operand-aware reusable callable with an
explicit `device_operands` parameter. Call `remove(operation)` before changing
an operation family between unconstrained and device-specific registration
modes; this applies equally to the standard CZ tables and unconstrained RX
rules.

## Execution configuration

The matrix simulator separates two kinds of parallel work: independent sampled
executions belong to **shot parallelism**, while numerical work within one
evolution belongs to **kernel parallelism**. Kernel parallelism never overlaps
ordered program operations or executes them out of order.

The normalized per-run configuration is:

```python
simulation_config = {
    "seed": None,
    "shot_parallelism": "auto",
    "kernel_parallelism": "auto",
    "max_workers": None,
    "fusion": False,
}
```

`kernel_parallelism` controls parallel numerical work inside the state or
operator kernels of one evolution. The supported requests are:

| Shot request | Kernel request | Meaning and constraints |
| --- | --- | --- |
| `"auto"` | `"auto"` | Choose the best-known validated strategy, using at most one parallel axis. |
| `"serial"` | `"serial"` | Run locally with sequential shots and one numerical worker. |
| `"serial"` | `"auto"` | Replay trajectories in order; supported kernels remain adaptive. |
| `"serial"` | `"threads"` | Replay trajectories in order and thread supported kernels; Numba only. |
| `"threads"` | `"serial"` or `"auto"` | Thread independent shots through a compatible compiled Numba counts-only plan; at least two shots and workers are required. Kernel `"auto"` yields to the explicit shot axis. |
| `"processes"` | `"serial"` or `"auto"` | Shard independent shots across reusable worker processes; at least two shots and workers plus a shardable result are required. Every child uses one kernel thread. |
| `"auto"` | `"serial"` | Auto may parallelize shots; kernels stay serial. |
| `"auto"` | `"threads"` | Force supported Numba kernels; shot auto yields to the explicit kernel axis. |

Threaded or process shots apply only to programs that need independent
per-shot evolution, such as dynamic measurement, reset, or feedforward. A
static circuit that evolves once and samples terminal measurements rejects
those explicit requests. Threaded shots also require a compatible compiled
plan. Explicit kernel threads are rejected by the NumPy runtime or when fewer
than two threads are available; an exactly empty plan is accepted as a no-op.
Requests that would nest threaded or process shots with threaded kernels are
rejected. A supported explicit mode is honored even when it is slower; an
inapplicable request fails before numerical materialization instead of silently
changing strategy.

`max_workers` is `None` or a positive integer and limits whichever axis becomes
parallel. It cannot create independent work. An explicit parallel request with
`max_workers=1` is contradictory and is rejected; `1` remains useful with
`"auto"` to force a serial choice. With explicit Numba threads and no ceiling,
FATQAT uses the available Numba capacity. With `"auto"` and no ceiling, it
preserves the caller's active Numba thread mask. With no ceiling, process mode
resolves a stable CPU-count worker limit for the reusable executor and only
reduces the number of submitted batches when there are fewer shots.

`"auto"` applies FATQAT's current conservative selection heuristics; it is not a
guarantee that the selected strategy is fastest. The heuristics may evolve
between releases, while the explicit mode meanings remain stable. Omitting the
two parallelism fields preserves current `main` behavior, including its active
Numba mask and automatic strategy choices. Fusion is a separate choice and
deliberately defaults to the opt-in value `False`; omitted configuration is
therefore not otherwise promised to be identical to current `main`.

The three common configurations are:

```python
# Fully serial.
fully_serial = {
    "shot_parallelism": "serial",
    "kernel_parallelism": "serial",
}

# Ordered shots with adaptive numerical kernels.
serial_adaptive = {
    "shot_parallelism": "serial",
    "kernel_parallelism": "auto",
}

# Let FATQAT select at most one parallel axis (also the defaults).
automatic = {
    "shot_parallelism": "auto",
    "kernel_parallelism": "auto",
}
```

For a dynamic or noisy workload with independent trajectories, setting
`shot_parallelism="processes"` and `kernel_parallelism="serial"` explicitly
selects process sharding. Process startup can dominate small workloads, so use
it only when measurement evidence supports the choice.

### Fusion

`fusion` is independent of concurrency and defaults to `False`. `True` may
merge adjacent operations for Numba density-matrix, unitary, and superoperator
execution, reducing kernel launches and numeric passes on sufficiently long
plans. The wider intermediate work and preparation are not universally
profitable, so benchmark the intended workload. NumPy and Numba statevector
reject it. Compiled multi-shot statevector execution is a different mechanism
and works with fusion disabled.

### Seed reproducibility

The seed contract depends on whether a strategy change also changes the
sampling algorithm:

| Comparison | Guarantee |
| --- | --- |
| Same fixed integer seed, complete configuration, versions, and execution environment | The sample is reproducible. |
| Compiled multi-shot statevector counts with `kernel_parallelism="serial"` pinned, changing only `shot_parallelism` between `"serial"` and `"threads"`, or changing its ceiling | Seeded counts are identical. Omitting the kernel setting with serial shots selects ordered replay instead of the compiled loop. |
| NumPy per-shot replay, changing only local execution versus process batching | Seeded counts are identical. |
| Compiled multi-shot statevector noisy trajectories versus ordinary per-shot replay | The distributions agree; identical per-seed counts are not promised. |
| Deterministic execution | Results follow the existing numerical-tolerance contract; the seed is irrelevant. |
