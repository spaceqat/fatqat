# FatQat

FatQat is a quantum-computing toolkit built around one authoring interface:
`Program`. Write the computation once, then choose how closely to model the
machine beneath it.

| Execution level | Start here when you want to… |
| --- | --- |
| General simulation | study logical states, samples, observables, noise, or parameter sweeps |
| Hardware-profile simulation | check native operations, placement, connectivity, capacity, or atom occupancy |
| Hamiltonian emulation | follow pulses, coupling, leakage, timing, and continuous-time noise |

The execution targets accept the same `Program` type and return results through
the same `Job`/`Result` workflow. Each target still validates what it can
physically or mathematically realize.

## Installation

FatQat requires Python 3.12 or newer and is not yet published on PyPI. Install
it from a source checkout:

```sh
git clone https://github.com/BoxiLi/fatqat.git
cd fatqat
python -m pip install .
```

## Run a first Program

This Bell-state example contains the complete circuit-level workflow:

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))
program.measure_all()

result = fq.simulator.Simulator().run(
    program,
    shots=1000,
    simulation_config={"seed": 7},
).result()

print(result.get_counts())
```

Only `00` and `11` appear: the measured bits agree because the two qubits are
entangled. The [quickstart](docs/sphinx/guide/quickstart.md) draws this Program,
runs it, and turns the counts into a plot.

## Grow the same authoring model

`Program` records registers and ordered instructions. It supports gates,
measurement, reset, classical conditions, reusable parameters, logical qudits,
mixed local dimensions, circuit drawing, and direct physical controls. These
features stay together instead of splitting into separate circuit and pulse
languages.

The [Program guide](docs/sphinx/guide/program.md) builds those ideas step by
step. [Choose how much physics to model](docs/sphinx/guide/execution-models.md)
then runs one unchanged rotation through all three execution levels.

From there:

- [Simulate a quantum program](docs/sphinx/guide/simulation.md) for states,
  sampling, and parameter sweeps.
- [Ask questions of a run](docs/sphinx/guide/interpret-results.md) for counts,
  states, maps, and observables.
- [Compare ideal and noisy execution](docs/sphinx/guide/ideal-and-noisy.md).
- [Measure performance and scaling](docs/sphinx/guide/performance.md).
- [Test a hardware profile](docs/sphinx/guide/hardware-profile-simulation.md).
- [Follow a Program into physical dynamics](docs/sphinx/guide/hamiltonian-emulation.md),
  then continue with the transmon or neutral-atom workflow.
- [Connect OpenQASM and Qiskit](docs/sphinx/guide/interoperability.md).

The [tutorials](docs/sphinx/tutorials/index.rst) are longer algorithm and
physics case studies. The [API reference](docs/sphinx/api/index.rst) contains
the exact signatures, supported operations, shapes, units, and validation
contracts.

## Development

Install the source tree in editable mode with the development dependencies,
then run the tests:

```sh
python -m pip install --upgrade pip
python -m pip install --editable . --group dev
python -m pytest
```

Build the documentation with warnings treated as errors:

```sh
python -m pip install --editable . --group docs
python -m sphinx -b html -W --keep-going -E -a docs/sphinx docs/sphinx/_build/html
```

The main repository directories are:

- `src/fatqat/` — package source.
- `tests/` — behavior-focused test suite.
- `tutorials/` — executable case studies used by Sphinx-Gallery.
- `docs/sphinx/` — user guide, tutorials, and API reference.
