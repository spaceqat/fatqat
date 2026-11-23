---
title: "API reference"
---

# API reference


Use this reference to look up FatQat's public classes, functions, and
exceptions. For a complete example, see the [Quickstart](../guide/quickstart.md).

## Build programs

- [Program](program.md)
- [Registers](registers.md)
- [LogicalProgram](logical-program.md)
- [Compiler](compiler.md) – Compile logical circuits or OpenQASM for a device model.
    - [SCTarget](sc-target.md)
    - [Compiler pipelines & IR](compiler-pipelines.md)
- [Operations](operations.md)
- [Pulse control](pulse-control/index.md) – Specify waveforms and map gates to pulse definitions.

## Run and inspect

- [Simulators](simulators/index.md) – Run gate-level and linear-optical programs.
- [Emulators](emulators/index.md) – Follow Hamiltonian evolution under physical controls.
- [Noise](noise.md) – Choose a noise model and check backend support.
- [Backend](backend.md)
- [Estimator](estimator.md) – Compute expectation values of observables.
- [Job](job.md)
- [Result](result.md)

## Integrate and extend

- [OpenQASM & Qiskit](interoperability/index.md) – Exchange circuits or run FatQat through Qiskit's backend interface.
- [Matrix implementation maps](implementation.md) – Supply custom matrices and channel rules.

## Errors and stability

- [Exceptions](exceptions.md)
- [Experimental APIs](experimental.md)
