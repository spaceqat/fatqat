---
title: "Emulators"
---

# Emulators


A control pulse can rotate a qubit, couple two transmons, or drive an array of
atoms. FatQat's pulse emulators calculate how the quantum state changes during
those controls. They use a Hamiltonian for the physical system, so the pulse
shape, strength, and timing can all affect the result.

| Emulator | What it models |
| --- | --- |
| [`TransmonEmulator`](../pulse-emulator.md) | Three levels per transmon, with calibrated gates and channel-addressed pulses. Use it to study rotations, coupled gates, and leakage. |
| [`Atom2LevelEmulator`](../atom-emulators.md) | Two-level Rydberg atoms at fixed positions. Use it to study how global laser pulses and distance-dependent interactions affect an array. |

See [Pulse control](../pulse-control/index.md) to choose how to specify a pulse,
or [Write and run pulse programs](../../guide/hamiltonian-emulation.md) for a
complete example.

## Choose a result { #choose-a-result }

Both emulators take a `Program` and return a completed job from `run()`. The
`method` selects the form of the state or operator you can read from
`job.result()`.

| Method | Use it for | Result accessor |
| --- | --- | --- |
| `statevector` (default) | Follow a pure state or sample one trajectory when Lindblad noise can act. | [`get_statevector()`][fatqat.Result.get_statevector] |
| `density_matrix` | Compute the ensemble state under supported Lindblad noise. | [`get_density_matrix()`][fatqat.Result.get_density_matrix] |
| `unitary` | Compute the coherent evolution operator for a program without measurement, reset, classical conditions, or potentially active Lindblad noise. | [`get_unitary()`][fatqat.Result.get_unitary] |

These accessors return QuTiP `Qobj` values. Use `.full()` for a NumPy array.
Measurement counts are returned by default when the program measures. To retain a
stochastic final state from one shot, pass `result_config={"final_state": True}`
and `shots=1`.

For the run options, see
[`TransmonEmulator.run()`][fatqat.emulator.TransmonEmulator.run] or
[`Atom2LevelEmulator.run()`][fatqat.emulator.Atom2LevelEmulator.run].
