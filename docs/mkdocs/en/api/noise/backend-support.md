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


[`Simulator`][fatqat.simulator.Simulator], the two superconducting profiles, and
[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] share these channel rules. The
profiles may impose additional operation, placement, and dimension limits.

**Built-in simulator channels**

| Noise type and form | Support |
| --- | --- |
| [`Depolarizing`][fatqat.noise.Depolarizing] `(p)` | **Built in.** Applies jointly to the selected operands after a matching operation. |
| [`PauliChannel`][fatqat.noise.PauliChannel] | **Built in.** The string width must equal the number of selected qubit operands. |
| [`AmplitudeDamping`][fatqat.noise.AmplitudeDamping] `(p)` | **Built in.** Applies to one selected operand; dimension $d$ requires $d-1$ adjacent-transition probabilities. |
| [`PhaseDamping`][fatqat.noise.PhaseDamping] `(p)` | **Built in.** Applies to one selected operand of any finite local dimension. |
| Built-in rate forms and [`ThermalRelaxation`][fatqat.noise.ThermalRelaxation] | **Unsupported.** Simulators have no physical timeline and do not convert rates or times into a channel application. |
| [`ReadoutConfusion`][fatqat.noise.ReadoutConfusion] | **Built in.** Applies universally or to one measured operand. The matrix size must equal the reported digit dimension; the superconducting profiles therefore require `2 x 2`. |

[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] additionally supports
[`Loss`][fatqat.noise.Loss] after matching operations. It samples each selected
present carrier independently. A matching registration, even with `p=0`,
enables explicit occupancy; loss after `Put` can model a loading failure.
Other simulators reject `Loss`. Empty-site erasure outcome `2` bypasses
readout confusion because no physical digit was measured.

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
| [`TransmonEmulator`][fatqat.emulator.TransmonEmulator] | Background or operation scope: [`AmplitudeDamping`][fatqat.noise.AmplitudeDamping] `(rate)` with two rates for the three-level model, [`PhaseDamping`][fatqat.noise.PhaseDamping] `(rate or t_phi)`, [`ThermalRelaxation`][fatqat.noise.ThermalRelaxation], and [`Depolarizing`][fatqat.noise.Depolarizing] `(rate)` over the full qutrit. | Readout confusion is built in and binary. Construction and [`validate_noise_model`][fatqat.emulator.TransmonEmulator.validate_noise_model] require a `2 x 2` matrix. |
| [`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator] | Background scope: [`Depolarizing`][fatqat.noise.Depolarizing] `(rate)`, [`AmplitudeDamping`][fatqat.noise.AmplitudeDamping] `(rate)` with one rate, [`PhaseDamping`][fatqat.noise.PhaseDamping] `(rate or t_phi)`, and [`ThermalRelaxation`][fatqat.noise.ThermalRelaxation]. | Operation-scoped continuous noise is unsupported. Readout confusion is built in and `2 x 2` only. |

Each supported background declaration selects one site. Transmon
operation-scoped noise is active only during matching pulse windows. Readout
confusion may apply to every measurement or one operand; correlated
multi-operand readout is not supported.

Rates use the inverse of the model's time unit. `t_phi`, `t1`, and `t2`
use that unit directly. The reference
[`time_unit`][fatqat.emulator.TransmonModel.time_unit] is nanoseconds,
while [`time_unit`][fatqat.emulator.Atom2LevelModel.time_unit] is microseconds. Read
the chosen model's `time_unit` rather than guessing from a value's magnitude.

Unsupported continuous declarations cannot be enabled through emulator
construction. Use `NoiseModel` for the declarations listed above and call
the family's `validate_noise_model()` method for an eager support check.
