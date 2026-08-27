# fatqat

fatqat is a quantum circuit simulator built around a clean three-part split:
a `Program` describes *what* to run (registers, gates, measurements,
feedforward), a backend decides *how* to run it (state representation,
execution technology, noise), and a `Result` reports what came out (counts,
statevector, density matrix). Programs are backend-agnostic: the same
program runs pure or mixed, ideal or noisy, NumPy or JIT-compiled, without
changes.

## Installation

fatqat is not yet published on PyPI. Until the first package release, install
it from a source checkout:

```sh
git clone https://github.com/BoxiLi/fatqat.git
cd fatqat
python -m pip install .
```

## Quick start

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)          # 2 qubits, 2 clbits
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))
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
bell.add(ops.H, 0)
bell.add(ops.CX, (0, 1))

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
`"numba"` (the default, using JIT-compiled kernels) or `"numpy"`. Both runtimes
support the statevector, density-matrix, unitary, and superoperator methods.
They implement the same simulation: deterministic arrays agree to
floating-point rounding and sampled results agree in distribution:

```python
backend = fq.simulator.Simulator(method="SV", runtime="numba")
noise = fq.NoiseModel()
noise.add(fq.noise.Depolarizing(p=0.05), operation=ops.CX)
noisy = fq.simulator.Simulator(method="DM", runtime="numba", noise=noise)
```

Numba remains the recommended runtime for repeated circuits and sustained
workloads. Its first use may spend time compiling kernels or loading their
on-disk cache; later calls reuse those signatures. Changing circuit gates,
topology, or qubit count normally does not recompile a compatible execution
path, because Numba specializes on array types and ranks rather than circuit
contents. Choose NumPy when predictable latency for a small one-off run matters
more than throughput.

Matrix execution exposes `shot_parallelism` for work across sampled shots and
`kernel_parallelism` for numerical work inside one evolution. Both default to
`"auto"`, which applies FATQAT's current conservative selection heuristics while
avoiding nested worker pools. `max_workers` is a ceiling for whichever axis is
selected; it does not make program operations overlap or run out of order. See
[Advanced user topics](docs/sphinx/guide/advanced.md#execution-configuration) for
the supported combinations.

Operation fusion is a separate opt-in rewrite. `fusion=False` is the default.
`fusion=True` is supported by the Numba density-matrix, unitary, and
superoperator methods. It can improve throughput on longer plans by reducing
numeric passes over the state or operator, while changing floating-point
association order:

```python
fused = noisy.run(
    bell,
    simulation_config={"fusion": True},
)
```

Fusion does not select Numba's compiled multi-shot statevector execution; that
path also works with the default unfused plan.

## Dynamic circuits

Measurement, feedforward, and reset are first-class program constructs; a
`Barrier` is a compiler-facing marker with no simulation effect. The backend
automatically switches between a fast single-evolution path and per-shot
or compiled multi-shot execution, and chooses an execution strategy based on
the requested results and workload:

```python
dyn = fq.Program(2, 2)
dyn.add(ops.H, 0)
dyn.measure(0, 0)                # mid-circuit measurement
dyn.add(ops.X, 1, condition=(0, 1))   # applied only when clbit 0 read 1
dyn.add(ops.Reset, 0)                 # reprepare q0 in |0>
dyn.add(ops.Barrier, (0, 1))          # compiler marker, no-op here
dyn.measure(1, 1)
```

## Noise

Noise lives in a `NoiseModel`, built separately from the program and passed
to the backend, so the same program runs ideal or noisy without changes.
Quantum channels attach to gate occurrences; readout confusion is classical
(the collapse stays true, only the reported bit is resampled):

```python
import numpy as np

noise = fq.NoiseModel()
noise.add(fq.noise.Depolarizing(p=0.05), operation=ops.CX)
damping, dephasing = fq.noise.ThermalRelaxation(t1=60e-6, t2=80e-6).as_channels(2e-6)
noise.add(damping, operation=ops.H)
noise.add(dephasing, operation=ops.H)
noise.add(fq.noise.ReadoutConfusion(np.array([[0.98, 0.05], [0.02, 0.95]])))

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
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))

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
qutrits.add(ops.Fourier, 0)
qutrits.add(ops.Sum, (0, 1))          # |i, j> -> |i, i+j mod 3>
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
sorted(operation.name for operation in impl_map.supported_operations())  # ['CZ', 'RZ', 'SX', 'X']
impl_map.supports(ops.CX)                                     # False
```

The neutral-atom target `AtomArraySimulator` (`RX`, `RY`, `RZ`, `CZ`) has no
fixed topology at all: `ops.Put` loads atoms into sites, and a `CZ` is legal
only while its two atoms are connected by `ops.Pair` (undone by `ops.Unpair`),
so connectivity is rearranged mid-circuit. An unpaired `CZ` is rejected with
`fq.errors.BackendValidationError`.

## Physics emulators

Three pulse-resolved physics systems live under `fq.emulator`. Gate-authored
and direct-control programs are independent capabilities; supporting one does
not determine whether a backend supports the other.

| System | Physical basis | Gate-authored programs | Direct controls | Gate map |
|---|---|---|---|---|
| `TransmonEmulator` | three-level transmons | yes | yes | public, optional `gate_implementation_map=` with a built-in default |
| `Atom3LevelEmulator` | `\|0>, \|1>, \|r>` atoms | yes | yes | public, optional `gate_implementation_map=` with a built-in default |
| `Atom2LevelEmulator` | `\|g>, \|r>` atoms | custom rules only | yes, global drive/detuning | public, optional `gate_implementation_map=` with an empty built-in map |

The common workflow hides nominal package calibration defaults:

```python
transmons = fq.emulator.TransmonEmulator(transmon_model)
atoms = fq.emulator.Atom3LevelEmulator(atom_model, arrangement=arrangement)
```

For explicit calibration, construct a complete calibration document, compile
it with the corresponding required-input standard map builder, then pass only
that map to the emulator. Calibration is portable map-construction data, not
mutable emulator state. Package defaults are simulation baselines rather than
hardware-fidelity guarantees. See the Sphinx emulator guides for the complete
workflow, physical controls, noise, and result semantics.

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

## Development

Install the source tree in editable mode together with the development
dependencies, then run the test suite:

```sh
python -m pip install --upgrade pip
python -m pip install --editable . --group dev
python -m pytest
```

## Documentation

The user guide and API reference live in `docs/sphinx`. Build them locally
with warnings-as-errors, so a missing docstring or broken cross-reference
fails the build instead of silently vanishing:

```sh
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r docs/requirements.txt
python -m sphinx -b html -W docs/sphinx docs/sphinx/_build/html
```

Then open `docs/sphinx/_build/html/index.html`. This matches the existing
`docs/sphinx/Makefile` and `make.bat` `<build-dir>/<builder>` layout; direct
commands and Read the Docs use the same convention. Read the Docs builds the
published HTML with warnings treated as errors. The documentation environment
has its own Python 3.12 pin in `docs/requirements.txt`, keeping documentation
builds reproducible independently of the contributor environment.

## Project layout

- `src/fatqat/` — package source: `Program`, `operations` (gates,
  measurement, reset, barrier), `simulator` (the gate-level `Simulator`, the
  fake devices, and `simulator._engine` — the NumPy and Numba `MatrixEngine`
  implementations that own the state), `emulator` (the transmon, three-level
  atom, and two-level atom physics emulators and their shared `PulseEngine`),
  `_backends` (private infrastructure both backend families
  share: resolved execution steps, the backend/engine contract, lowering
  helpers), `implementation` (gate-matrix rules and registry), `noise`
  (physical noise declarations, `NoiseModel`, readout confusion), registers,
  layout, jobs, results,
  QASM translation, errors.
- `examples/` — runnable end-to-end scripts (smoke-tested by `tests/test_examples.py`).
- `tests/` — pytest suite.
- `docs/sphinx/` — user guide and API reference (see above).
