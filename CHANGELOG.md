# Changelog

## 0.1.0a3 - 2026-11-16

- Added linear-optics simulation with optical gates, mode and count registers,
  and `LOSimulator`, including fixed-photon-number states.
- Faster Numba statevector simulation, parameter sweeps, noisy trajectories,
  and expectation values. A new performance guide explains how to choose
  runtimes and parallelism settings.
- Custom operations can now update both quantum and classical state. Gates
  and noise channels share an implementation map with target-specific rules.
- Added measurement destinations via `Program.measure_all(outputs=...)`,
  final classical-state output via `Result.get_classical_state()`, and qudit
  observables for expectation values.
- Added `fatqat.simulator.set_default_runtime()` to choose the runtime for
  new simulators.
- Atom arrangements can now be created directly from coordinates.
- Fixed multi-block pulse composition and the DRAG virtual-Z frame-update sign.
- Added installation guidance, Chinese repository documentation, a Korean
  README, and tutorials on surface-code memory, qutrit-assisted Toffoli gates,
  QAOA, and pulse calibration.

**Compatibility and migration:**

- `execution_config=` replaces `simulation_config=`, which is deprecated and
  will be removed before 0.1.0. Do not pass both. Result metadata includes both
  keys during the transition.
- Pulse emulators now return QuTiP `Qobj` values for final states and unitaries.
  Use `.full()` to obtain a NumPy array. Gate-level simulators still return arrays.
- Custom noise rules now belong in `MatrixImplementationMap`, passed through
  `implementation_map=`. This replaces `ChannelImplementationMap`,
  `default_channel_implementation_map()`, and `channel_implementation_map=`.
  Use `default_matrix_implementation_map()` to keep the built-in rules.
- `AtomArraySimulator` now rejects finite noise channels acting on multiple
  subsystems. `Loss` still supports multiple targets, sampled independently.

## 0.1.0a2 - 2026-10-04

This alpha updates the documentation and examples, with targeted compiler
and extension API changes.

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

## 0.1.0a1 - 2026-09-12

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
