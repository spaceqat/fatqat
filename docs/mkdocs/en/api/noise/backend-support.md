---
title: "Backend support"
---

<a id="noise-backend-support"></a>


# Backend support


Use these tables to see which built-in backends accept each noise type and
form. **Built in** means the standard backend accepts it. **Unsupported** means
the backend family rejects it.

For one configured backend, use
[`validate_noise_model`][fatqat.simulator.Simulator.validate_noise_model] on a simulator or
[`validate_noise_model`][fatqat.emulator.TransmonEmulator.validate_noise_model] on a pulse
emulator. These methods validate a model without a program. FATQAT checks
references, device labels, operation matches, dimensions, and execution-method
limits when it runs a concrete program.

<a id="noise-simulator-support"></a>


## Simulators


[`Simulator`][fatqat.simulator.Simulator],
[`SCQubitSimulator`][fatqat.simulator.SCQubitSimulator], and
[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] share these channel rules. The two hardware profiles may impose
additional operation, placement, and dimension limits.

**Built-in simulator channels**

| Noise type and form | Support |
| --- | --- |
| [`Depolarizing`][fatqat.noise.Depolarizing] `(p)` | **Built in.** Applies jointly to the selected operands after a matching operation. |
| [`PauliChannel`][fatqat.noise.PauliChannel] | **Built in.** The string width must equal the number of selected qubit operands. |
| [`AmplitudeDamping`][fatqat.noise.AmplitudeDamping] `(p)` | **Built in for qubits.** Applies the conventional finite amplitude-damping channel to one selected operand. |
| [`TransitionRelaxation`][fatqat.noise.TransitionRelaxation] `(p)` | **Built in.** Applies one explicit local jump; matching declarations are applied sequentially in registration order. |
| [`PhaseDamping`][fatqat.noise.PhaseDamping] `(p)` | **Built in.** Applies to one selected operand of any finite local dimension. |
| Built-in rate forms and [`ThermalRelaxation`][fatqat.noise.ThermalRelaxation] | **Unsupported.** Simulators have no physical timeline and do not convert rates or times into a channel application. |
| [`ReadoutConfusion`][fatqat.noise.ReadoutConfusion] | **Built in.** Applies universally or to one measured operand. The matrix size must equal the reported digit dimension; [`SCQubitSimulator`][fatqat.simulator.SCQubitSimulator] therefore requires `2 x 2`. |

[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] also supports
[`Loss`][fatqat.noise.Loss], sampling each selected carrier that is present
after a matching operation. Sites begin empty and
[`Put`][fatqat.operations.Put] loads
them, so loss attached to `Put` can model a loading failure. Other simulators
reject `Loss`. Because an empty site produces no physical readout digit, its
erasure value `2` bypasses readout confusion.

Attach every supported probability-form channel above to an operation; matrix
backends reject background channels.
A custom [`ChannelImplementationMap`][fatqat.noise.ChannelImplementationMap] can add a finite
channel type or replace the rule for an existing one. It cannot override the
rejection of built-in rate forms, background channels, or
`ThermalRelaxation`. A custom type's parameter names are interpreted by its
rule, whose result still represents one finite channel application. A supplied
map replaces the default map and is copied when the backend is constructed.

Execution method also matters. `statevector` samples one Kraus trajectory
per noisy shot, while `density_matrix` and `superop` apply
probability-form channels exactly. `unitary` rejects any channel that
matches the program. Only `statevector` and `density_matrix` support the
AtomArray occupancy lifecycle; requesting a final state with loss requires
`shots == 1`.

<a id="noise-emulator-support"></a>


## Pulse emulators


Pulse emulators use local Lindblad operators built from supported rates and
times. They do not infer a rate from a probability. Built-in probability
forms, [`PauliChannel`][fatqat.noise.PauliChannel], and [`Loss`][fatqat.noise.Loss]
are unsupported. Each emulator family owns its continuous-noise realizations.

**Emulator support**

| Backend | Built-in Lindblad-operator behavior | Readout and limits |
| --- | --- | --- |
| [`TransmonEmulator`][fatqat.emulator.TransmonEmulator] | Background or operation scope: [`TransitionRelaxation`][fatqat.noise.TransitionRelaxation] `(rate)` with explicit coefficients, [`PhaseDamping`][fatqat.noise.PhaseDamping] `(rate or t_phi)`, and [`Depolarizing`][fatqat.noise.Depolarizing] `(rate)` over the full qutrit. | Readout confusion is built in and binary. Construction and [`validate_noise_model`][fatqat.emulator.TransmonEmulator.validate_noise_model] require a `2 x 2` matrix. |
| [`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator] | Background scope: qubit [`AmplitudeDamping`][fatqat.noise.AmplitudeDamping] `(rate)`, explicit [`TransitionRelaxation`][fatqat.noise.TransitionRelaxation] `(rate)`, [`PhaseDamping`][fatqat.noise.PhaseDamping] `(rate or t_phi)`, qubit [`ThermalRelaxation`][fatqat.noise.ThermalRelaxation], and [`Depolarizing`][fatqat.noise.Depolarizing] `(rate)`. | Operation-scoped continuous noise is unsupported. Readout confusion is built in and `2 x 2` only. |

Each supported background declaration selects one site. Transmon
operation-scoped noise is active only during matching pulse windows. Readout
confusion may apply to every measurement or one operand; correlated
multi-operand readout is not supported.

Rates use the inverse of the model's time unit. `t_phi`, `t1`, and `t2`
use that unit directly on families that accept them. The reference
[`time_unit`][fatqat.emulator.TransmonModel.time_unit] is nanoseconds,
while [`time_unit`][fatqat.emulator.Atom2LevelModel.time_unit] is microseconds. Read
the chosen model's `time_unit` rather than guessing from a value's magnitude.

Unsupported continuous declarations cannot be enabled through emulator
construction. Use `NoiseModel` for the declarations listed above and call
the family's `validate_noise_model()` method for an eager support check.
