<div align="center">

<img src="https://raw.githubusercontent.com/spaceqat/fatqat/main/docs/mkdocs/en/assets/images/space-qat-logo.png" alt="" width="120">

# FatQat

**One Program, from algorithm to hardware.**

[![Tests](https://github.com/spaceqat/fatqat/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/spaceqat/fatqat/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/fatqat)](https://pypi.org/project/fatqat/)
[![Python](https://img.shields.io/pypi/pyversions/fatqat)](https://pypi.org/project/fatqat/)
[![Documentation](https://img.shields.io/readthedocs/fatqat)](https://fatqat.readthedocs.io/en/latest/)
[![License](https://img.shields.io/pypi/l/fatqat)](https://github.com/spaceqat/fatqat/blob/main/LICENSE)

[Quickstart](https://fatqat.readthedocs.io/en/latest/guide/quickstart/) · [Documentation](https://fatqat.readthedocs.io/en/latest/) · [Case studies](https://fatqat.readthedocs.io/en/latest/tutorials/)

English | [简体中文](README.zh-CN.md) | [한국어](README-KR.md)

</div>

FatQat is a Python toolkit for building, compiling, and running quantum
programs. Write a `Program` once and run it through the same interface on every
backend, from an ideal simulator to a physical model of the device.

- **Algorithm developers** test their algorithms under realistic device
  conditions and prepare them for hardware execution.
- **Hardware engineers** simulate how their devices respond to controls,
  coupling, and noise, and find what limits performance.
- **Students** learn quantum computing hands-on, from first circuits to the
  physics of real devices, through guided examples and case studies.

The current backends offer three levels of physical detail:

| Model | Question it answers |
| --- | --- |
| General simulation | Does the algorithm work, and how does noise change it? |
| Hardware-profile simulation | Does it fit the device's native gates, connectivity, and atom occupancy? |
| Hamiltonian emulation | What happens during the pulses: leakage, coupling, and timing? |

A `Program` is more than a gate circuit. It can hold:

- gates, mid-circuit measurement, feedforward, and parameters on qubits,
  qudits, or optical modes;
- device operations such as loading, pairing, and losing atoms;
- control pulses written directly, alongside calibrated gates.

FatQat also compiles logical circuits for superconducting and neutral-atom
targets, and exchanges circuits with OpenQASM and Qiskit.

## Install

FatQat requires Python 3.12 or newer:

```sh
python -m pip install fatqat
```

## Run a first Program

```python
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))
program.measure_all()

result = fq.simulator.Simulator().run(program, shots=1000).result()
print(result.get_counts())  # only '00' and '11'
```

## Learn more

- [Quickstart](https://fatqat.readthedocs.io/en/latest/guide/quickstart/):
  build, draw, and run a first Program.
- [User guide](https://fatqat.readthedocs.io/en/latest/guide/):
  simulation, hardware constraints, and pulse dynamics, task by task.
- [Case studies](https://fatqat.readthedocs.io/en/latest/tutorials/):
  error correction, quantum simulation, optimization, and pulse control.
- [API reference](https://fatqat.readthedocs.io/en/latest/api/):
  exact signatures, units, and validation rules.

Release notes are in the
[changelog](https://github.com/spaceqat/fatqat/blob/main/CHANGELOG.md).

## Contribute

AI makes code easier to write. Your questions, use cases, and reviews help
decide what is worth writing.
[Read how you can make a difference](https://fatqat.readthedocs.io/en/latest/contributing/contributing-in-the-ai-era/),
then see
[Contributing to FatQat](https://github.com/spaceqat/fatqat/blob/main/CONTRIBUTING.md)
for development setup and the policy on AI-assisted work.

FatQat is licensed under the
[Apache License 2.0](https://github.com/spaceqat/fatqat/blob/main/LICENSE).
