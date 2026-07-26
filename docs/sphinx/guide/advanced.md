# Advanced

Everything in the rest of the guide applies beyond the plain-qubit case.
This page covers three extension points: qudits, custom matrix
implementations, and parallel shot execution.

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

qubit = fq.QuantumRegister(1, name="qubit")            # dim=2
qutrit = fq.QuantumRegister(1, dim=3, name="qutrit")
cbit = fq.ClassicalRegister(1, name="c_qubit")
ctrit = fq.ClassicalRegister(1, dim=3, name="c_qutrit")

program = fq.Program([qubit, qutrit], [cbit, ctrit])
program.add(fq.ops.X, program.qreg[0][0])            # qubit: |0> -> |1>
program.add(fq.ops.Shift(1), program.qreg[1][0])     # qutrit: |0> -> |1>
program.add_measurement(program.qreg[0][0], program.clreg[0][0])
program.add_measurement(program.qreg[1][0], program.clreg[1][0])

result = fq.backends.SimulatorBackend("SV").run(program, shots=100).result()
print(result.get_counts())   # {"11": 100}
```

Each register keeps its own dimension; `add()` and `add_measurement()`
resolve targets against whichever register a `RegisterRef` names, so mixed
dimensions require no special handling beyond addressing the right register
explicitly (see [Gates](gates.md) for how targets are addressed).

## Custom implementations

{py:class}`~fatqat.backends.SimulatorBackend`'s `implementation_map=`
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
from fatqat.implementation import FixedMatrix, default_matrix_implementation_map


class SqrtX(fq.ops.Operation):
    name = "SqrtX"
    _num_subsystems = 1


SQRT_X_MATRIX = 0.5 * np.array(
    [[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]],
    dtype=complex,
)

implementation_map = default_matrix_implementation_map()
implementation_map.register(SqrtX, FixedMatrix(SQRT_X_MATRIX))

backend = fq.backends.SimulatorBackend("SV", implementation_map=implementation_map)

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

- `register()` accepts a bare `np.ndarray` too — it auto-wraps into
  `FixedMatrix`, so `implementation_map.register(SqrtX, SQRT_X_MATRIX)` works
  just as well as wrapping it explicitly.
- A custom `Operation` subclass only needs `name` and `_num_subsystems`
  (the fixed arity the matrix implementation map requires — variable-arity
  operations aren't supported here).
- The matrix's shape is checked against the target dimensions at lowering
  time, so registering a 2x2 matrix for an operation applied to a qudit
  register raises a clear error rather than an opaque shape mismatch deeper
  in the engine.
- `implementation_map` is copied internally by the backend
  (`SimulatorBackend` never mutates the map you pass in), so the same map
  can be reused across multiple backend instances.

If the matrix instead depends on the operation's parameters (e.g. a rotation
angle) or on target dimension (e.g. a qudit gate), register a bare callable —
`f(op)` or `f(op, targets)` — instead of a `FixedMatrix`; the map detects
which shape you passed by inspecting the callable's signature.

## Parallel execution

For programs on the dynamic path, `run()`'s `simulation_config={...}` accepts
`max_workers` and `parallel_mode` (`"auto"`, `"serial"`,
`"multiprocessing"`, or `"loky"`) to control whether shots are distributed
across worker processes; it also accepts `seed` for reproducible sampling.
This only affects execution strategy and random streams, never deterministic
numerical semantics.
