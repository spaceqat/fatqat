# Changelog

## 0.1.0a3 - 2026-11-16

This alpha adds linear-optics simulation, speeds up circuit execution and
expectation values, and expands control over measurements and custom operations.

- Simulate optical modes with `ModeRegister`, `CountRegister`, optical gates,
  and `LOSimulator`, including fixed-photon-number state evolution.
- Improve Numba statevector execution, parameter sweeps, noisy trajectories,
  and expectation-value evaluation. Add a performance guide covering runtime
  selection, parallelism, repeated runs, and result storage.
- Configure the runtime used by newly constructed simulators with
  `fatqat.simulator.set_default_runtime()`. Numba remains the initial default.
- Add custom operations that can update quantum and classical state, with
  target-specific implementation rules shared by gates and noise channels.
- Choose measurement destinations with `Program.measure_all(outputs=...)`,
  retrieve final classical state with `Result.get_classical_state()`, and use
  qudit observables in expectation values.
- Construct atom arrangements from coordinates. Fix multi-block pulse
  composition and the sign of the DRAG virtual-Z frame update.
- Expand installation and first-run guidance, add Chinese repository documents
  and a Korean README, and add tutorials on surface-code memory, qutrit-assisted
  Toffoli gates, QAOA, and pulse calibration.

**Compatibility and migration:**

- Use `execution_config=` for per-run execution settings. `simulation_config=`
  remains a deprecated alias and is scheduled for removal before 0.1.0;
  supplying both is an error. Result metadata now includes `execution_config`
  and retains a copy under the deprecated `simulation_config` key.
- Pulse emulators now return QuTiP `Qobj` values for final states and unitaries.
  Use `.full()` when a NumPy array is needed. Gate-level simulators continue
  to return NumPy arrays.
- Register custom noise channels in `MatrixImplementationMap` and pass it as
  `implementation_map=`. This replaces `ChannelImplementationMap`,
  `default_channel_implementation_map()`, and `channel_implementation_map=`;
  start with `default_matrix_implementation_map()` to retain built-in rules.
- `AtomArraySimulator` accepts only single-subsystem finite noise channels;
  multi-subsystem channels are rejected. `Loss` can still target multiple
  subsystems, sampling each present atom independently.

Requires Python 3.12 or newer. This is an alpha release; pin the exact version
for reproducible studies.

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

Requires Python 3.12 or newer. This is an alpha release; pin the exact version
for reproducible studies.

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

Requires Python 3.12 or newer. This is an alpha release: interfaces may change
between releases. Pin the exact package version for reproducible studies.
