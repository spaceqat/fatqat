# Changelog

## 0.1.0a3 - 2026-11-16

This release adds linear-optics simulation, improves simulation performance, and extends support for custom operations, measurements, and observables.

### New features

- **Superconducting compilation:** added `SCTarget` to describe device sites and CZ couplings without constructing a simulator. Both `compile_to_sc()` and `compile_qasm_to_sc()` accept it, so the executor can be chosen afterward. Thanks to @yjmaxpayne ([#47](https://github.com/spaceqat/fatqat/pull/47)).
- **Linear optics:** added `LOSimulator`, optical gates, `ModeRegister`, and `CountRegister`, including simulation in a fixed-photon-number basis.
- **Custom operations:** operations can update both quantum and classical state. Gates and noise channels share an implementation map, with rules that can be registered for specific physical targets.
- **Measurement outputs:** `Program.measure_all(outputs=...)` accepts explicit measurement destinations. Thanks to @279wb ([#68](https://github.com/spaceqat/fatqat/pull/68)).
- **Final classical state:** simulator runs can return the final classical state through `Result.get_classical_state()` when requested with `result_config={"final_classical_state": True}`.
- **Qudit observables:** expectation-value calculations now support qudit observable factors.
- **Runtime defaults:** `fatqat.simulator.set_default_runtime()` sets the runtime for newly constructed simulators. Numba remains the initial default.
- **Atom arrangements:** arrangements can be constructed from coordinates and combined into labeled groups. Thanks to @chenx820 ([#11](https://github.com/spaceqat/fatqat/pull/11)).
- Improved Numba statevector execution, parameter sweeps, and noisy trajectories, reducing repeated setup and parameter-binding work.
- Improved expectation-value evaluation and reduced intermediate state storage during parameter sweeps.

### Fixes

- Fixed pulse contributions leaking across control blocks. Thanks to @fykuang02 ([#46](https://github.com/spaceqat/fatqat/pull/46)).
- Corrected the sign of the DRAG virtual-Z frame update.
- Fixed process-worker sizing to respect the CPU affinity mask. Thanks to @oscar-chw ([#52](https://github.com/spaceqat/fatqat/pull/52)).

### Documentation

- Expanded installation and first-run guidance, and added Chinese repository documentation. Thanks to @phoenix-chenli ([#70](https://github.com/spaceqat/fatqat/pull/70), [#93](https://github.com/spaceqat/fatqat/pull/93)).
- Added a Korean README. Thanks to @Enhyeok-Jang ([#96](https://github.com/spaceqat/fatqat/pull/96)).
- Added a surface-code memory tutorial.
- Updated existing QAOA and pulse-calibration tutorials, including reduced repeated work in the islanding QAOA example.
- Added a linear-optics guide and API reference.
- Added a performance guide covering runtime selection, parallelism, repeated runs, and result storage.

### Compatibility and migration

- **Execution settings:** use `execution_config=` instead of `simulation_config=`. The old keyword remains a deprecated alias and will be removed before 0.1.0. Passing both is an error. Result metadata includes both keys during the transition.
- **Pulse-emulator results:** final states and unitaries are now returned as QuTiP `Qobj` values. Call `.full()` when a NumPy array is needed. Gate-level simulators continue to return NumPy arrays.
- **Custom noise implementations:** register channel rules in `MatrixImplementationMap` and pass it through `implementation_map=`. This replaces `ChannelImplementationMap`, `default_channel_implementation_map()`, and `channel_implementation_map=`. Start with `default_matrix_implementation_map()` to retain built-in rules.
- **Atom-array noise:** `AtomArraySimulator` now rejects finite noise channels acting on multiple subsystems. Register separate single-subsystem channels using `target_positions`. `Loss` still supports multiple targets, sampling each present atom independently.

This is an alpha release; pin the exact version for reproducible studies.

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

This is an alpha release; pin the exact version for reproducible studies.

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

This is an alpha release; pin the exact version for reproducible studies.
