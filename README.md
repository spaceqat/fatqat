# FatQat

FatQat is a quantum-computing toolkit built around one authoring interface:
`Program`. Write the computation once, then choose how closely to model the
machine beneath it.

> **Development status:** FatQat is under active development, and its interfaces
> may change between releases. Pin an exact version when reproducibility matters.

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
git clone https://github.com/spaceqat/fatqat.git
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
entangled. The
[quickstart](https://fatqat.readthedocs.io/en/latest/guide/quickstart/)
draws this Program, runs it, and turns the counts into a plot.

## Grow the same authoring model

`Program` records registers and ordered instructions. It supports gates,
measurement, reset, classical conditions, reusable parameters, logical qudits,
mixed local dimensions, circuit drawing, and direct physical controls. These
features stay together instead of splitting into separate circuit and pulse
languages.

The
[Program guide](https://fatqat.readthedocs.io/en/latest/guide/program/)
builds those ideas step by step.
[Choose how much physics to model](https://fatqat.readthedocs.io/en/latest/guide/execution-models/)
then runs one unchanged rotation through all three execution levels.

From there:

- [Simulate a quantum program](https://fatqat.readthedocs.io/en/latest/guide/simulation/) for states,
  sampling, and parameter sweeps.
- [Ask questions of a run](https://fatqat.readthedocs.io/en/latest/guide/interpret-results/) for counts,
  states, maps, and observables.
- [Compare ideal and noisy execution](https://fatqat.readthedocs.io/en/latest/guide/ideal-and-noisy/).
- [Measure performance and scaling](https://fatqat.readthedocs.io/en/latest/guide/performance/).
- [Test a hardware profile](https://fatqat.readthedocs.io/en/latest/guide/hardware-profile-simulation/).
- [Follow a Program into physical dynamics](https://fatqat.readthedocs.io/en/latest/guide/hamiltonian-emulation/),
  then continue with the transmon or neutral-atom workflow.
- [Connect OpenQASM and Qiskit](https://fatqat.readthedocs.io/en/latest/guide/interoperability/).

The [tutorial gallery](https://fatqat.readthedocs.io/en/latest/tutorials/)
contains longer algorithm and physics case studies. The
[API reference](https://fatqat.readthedocs.io/en/latest/api/)
contains the exact signatures, supported operations, shapes, units, and
validation contracts.

## Development

Install the source tree in editable mode with the development dependencies,
then run the tests:

```sh
python -m pip install --upgrade pip
python -m pip install --editable . --group dev
python -m pytest
```

Before submitting a change, read
[Contributing to FatQat](https://github.com/spaceqat/fatqat/blob/main/CONTRIBUTING.md),
including the policy for AI-assisted work. AI tools are permitted, but every
contributor must understand, own, and lead the work they submit and the project
conversations around it.

For documentation changes, follow the
[pinned setup and build workflow](https://github.com/spaceqat/fatqat/blob/main/docs/mkdocs/README.md).

The main repository directories are:

- `src/fatqat/` — package source.
- `tests/` — behavior-focused test suite.
- `docs/mkdocs/` — Material user guide, executable tutorials, and API reference.
