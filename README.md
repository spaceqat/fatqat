# fatqat

fatqat is a quantum circuit simulator built around a clean three-part split:
a `Program` describes *what* to run (registers, gates, measurements,
feedforward), a backend decides *how* to run it (state representation,
execution technology, noise), and a `Result` reports what came out (counts,
statevector, density matrix). Programs are backend-agnostic: the same
program runs pure or mixed, ideal or noisy, NumPy or JIT-compiled, without
changes.

## Quick start

```python
import fatqat as fq
import fatqat.operations as op

program = fq.Program(2, 2)          # 2 qubits, 2 clbits
program.add(op.H, 0)
program.add(op.CX, (0, 1))
program.measure((0, 1), (0, 1))

backend = fq.simulator.Simulator(method="SV")
result = backend.run(program, shots=1000, simulation_config={"seed": 7}).result()
print(result.get_counts())          # {'00': 502, '11': 498}
```

## Simulation methods

`method` selects the state representation, Qiskit style: `"SV"` /
`"statevector"` for pure-state simulation, `"DM"` / `"density_matrix"` for
exact mixed-state simulation. Each method can export its native state
instead of (or alongside) counts:

```python
bell = fq.Program(2)
bell.add(op.H, 0)
bell.add(op.CX, (0, 1))

rho = (
    fq.simulator.Simulator(method="DM")
    .run(bell, result_config={"counts": False, "final_state": True})
    .result()
    .get_density_matrix()
)
print(rho.diagonal().real)          # [0.5 0.  0.  0.5]
```

## Runtimes

`runtime` selects the execution technology for the chosen representation —
`"numpy"` (the default) or `"numba"` for JIT-compiled kernels (both methods;
requires the optional `numba` dependency). The runtime never changes
results, only how fast they are computed:

```python
backend = fq.simulator.Simulator(method="SV", runtime="numba")
noisy = fq.simulator.Simulator(method="DM", runtime="numba")
```

## Dynamic circuits

Measurement, feedforward, and reset are first-class program constructs; a
`Barrier` is a compiler-facing marker with no simulation effect. The backend
automatically switches between a fast single-evolution path and per-shot
replay (parallelized across processes for large shot counts) depending on
what the program needs:

```python
dyn = fq.Program(2, 2)
dyn.add(op.H, 0)
dyn.measure(0, 0)                # mid-circuit measurement
dyn.add(op.X, 1, condition=(0, 1))   # applied only when clbit 0 read 1
dyn.add(op.Reset, 0)                 # reprepare q0 in |0>
dyn.add(op.Barrier, (0, 1))          # compiler marker, no-op here
dyn.measure(1, 1)
```

## Noise

Noise lives in a `NoiseModel`, built separately from the program and passed
to the backend, so the same program runs ideal or noisy without changes.
Quantum channels attach to gate occurrences; readout error is classical
(the collapse stays true, only the reported bit is resampled):

```python
import numpy as np

noise = fq.NoiseModel()
noise.add_channel(op.CX, fq.noise.Depolarizing(p=0.05))
damping, dephasing = fq.noise.relaxation_channels(t1=60e-6, t2=80e-6, duration=2e-6)
noise.add_channel(op.H, damping)
noise.add_channel(op.H, dephasing)
noise.add_readout_error(np.array([[0.98, 0.05], [0.02, 0.95]]))

backend = fq.simulator.Simulator(method="DM", noise=noise)
```

Under `method="DM"` channels apply exactly (one evolution); under
`method="SV"` each shot samples one Kraus branch (quantum trajectories) —
both converge to the same counts. A device backend can also author its own
calibration-derived model:
`SCQubitIBMSimulator.default_noise_model()`. The full noise guide belongs
to the Sphinx docs.

## Expectation values

An `Observable` plus an `Estimator` reads a quantity off the final state
instead of going through counts. The program carries no measurement — a
measurement would collapse the state the value is read from:

```python
program = fq.Program(2)             # no clbits, no measurement
program.add(op.H, 0)
program.add(op.CX, (0, 1))

estimator = fq.Estimator(fq.simulator.Simulator(method="SV"))
observables = [fq.Observable([("ZZ", 1.0)]), fq.Observable([("XX", 0.5)])]
print(estimator.run(program, observables).result().get_expectation())
# [1.  0.5]
```

Labels are little-endian, matching the counts strings, and coefficients
must be real. For a wide register name only the non-identity factors —
`fq.Observable.from_sparse([("XY", (3, 7), 1.5)], num_qubits=100)` — which
also reaches the `ZERO`/`ONE` projectors, so site occupation is written
directly as `<ONE_i>`. The `2**n x 2**n` matrix is never built.

Passing a list evaluates every observable against **one** evolution. This is
where a simulator beats hardware, which must fan a multi-basis observable out
into one circuit per commuting group and run each separately.

`shots=0` (the default, unlike `Simulator.run`) computes the value exactly,
including under `method="DM"` noise. `shots > 0` draws real samples to
reproduce a finite-shot experiment's statistical error, and
`result.get_std()` reports the standard error.

## Qudits

Registers take a per-slot dimension; gates like `Shift`, `Clock`, `Sum`, and
`Fourier` resolve at whatever dimension their targets have:

```python
qutrits = fq.Program([fq.QuantumRegister(2, dim=3)], [fq.ClassicalRegister(2, dim=3)])
qutrits.add(op.Fourier, 0)
qutrits.add(op.Sum, (0, 1))          # |i, j> -> |i, i+j mod 3>
qutrits.measure_all()
# counts over trits: {'00': 314, '11': 293, '22': 293}
```

## Device backends

Two configurable-grid prototype superconducting targets ship with distinct
native gate sets: `SCQubitIBMSimulator` (`X`, `SX`, `RZ`, nearest-neighbor
`CZ`) and `SCQubitGoogleSimulator` (`RX`, `RY`, `RZ`, nearest-neighbor
`iSwap` and `CZ`). Each implementation map is introspectable, so a compiler
can discover the device's constraints instead of hardcoding them:

```python
fake = fq.simulator.SCQubitIBMSimulator()
impl_map = fake.implementation_map
sorted(op.name for op in impl_map.supported_operations())   # ['CZ', 'RZ', 'SX', 'X']
impl_map.supports(op.CX)                                     # False
```

## OpenQASM

Programs translate to and from OpenQASM 2.0/3.0 text:

```python
from fatqat.qasm import from_qasm, to_qasm

program = from_qasm(
    'OPENQASM 2.0; include "qelib1.inc"; '
    "qreg q[2]; creg c[2]; h q[0]; cx q[0],q[1]; measure q -> c;"
)
print(to_qasm(program))             # OpenQASM 3.0 by default
```

## Dev setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```sh
uv sync                 # install runtime + dev dependencies into .venv
uv run pytest           # run the test suite
uv sync --group numba   # optional: enables runtime="numba"
```

## Documentation

The user guide and API reference live in `docs/sphinx`. Build them locally
with warnings-as-errors, so a missing docstring or broken cross-reference
fails the build instead of silently vanishing:

```sh
uv sync --group docs
uv run sphinx-build -b html -W docs/sphinx docs/sphinx/_build
```

Then open `docs/sphinx/_build/index.html`. There's no CI job or hosting for
this yet — it's for local/internal use.

## Project layout

- `src/fatqat/` — package source: `Program`, `operations` (gates,
  measurement, reset, barrier), `simulator` (the gate-level `Simulator`, the
  fake devices, and `simulator._engine` — the NumPy and Numba `MatrixEngine`
  implementations that own the state), `emulator` (the pulse-level `Emulator`
  and its `PulseEngine`), `_backends` (private infrastructure both families
  share: resolved execution steps, the backend/engine contract, lowering
  helpers), `implementation` (gate-matrix rules and registry), `noise`
  (channels, `NoiseModel`, readout error), registers, layout, jobs, results,
  QASM translation, errors.
- `tests/` — pytest suite.
- `docs/sphinx/` — user guide and API reference (see above).
