# Changelog

## 0.1.0a2

This alpha updates the documentation and examples, and includes the changes
merged to `main` since 0.1.0a1.

- Expand the compiler guide, add a dedicated `LogicalProgram` reference, and
  compile the homepage Grover example from one shared logical circuit. The SC
  compiler now decomposes Toffoli (`CCX`) gates before routing.
- Add tutorials on atom-loss erasure, Rydberg CZ pulse control, and quantum
  circuit search for molecular-property prediction. Organize tutorials by
  topic, add suggested reading paths, and add a version-aware language menu.
- Expose the `Backend` protocol for custom execution backends, read-only
  program instruction snapshots, and simulator discovery through versioned
  plugin entry points.

**Compatibility change:** `compile_to_sc()` and `compile_to_na()` now require
an exact `LogicalProgram` input. In 0.1.0a1 they also accepted `Program` and
converted it automatically. Construct a `LogicalProgram` before calling either
compiler entry point. OpenQASM compiler entry points remain available.

Requires Python 3.12 or newer. This is an alpha release; pin the exact version
for reproducible studies.

## 0.1.0a1

First alpha release of FatQat, a quantum-computing toolkit built around
the `Program` authoring interface.

- Author gates, measurements, reset, classical conditions, parameterized
  programs, logical qudits, and direct physical controls.
- Run general simulations, hardware-profile simulations, and Hamiltonian
  emulations through a shared `Job`/`Result` workflow.
- Explore noise, parameter sweeps, states, samples, and observables.
- Compile supported logical programs for hardware targets, including
  neutral-atom placement and routing.
- Work with OpenQASM and optional Qiskit integration, draw programs, and
  follow executable algorithm and physics tutorials.

Requires Python 3.12 or newer. This is an alpha release: interfaces may change
between releases. Pin the exact package version for reproducible studies.
