---
title: "Superconducting pulse emulator"
---

# Superconducting pulse emulator


[`TransmonEmulator`][fatqat.emulator.TransmonEmulator] models each transmon with
three energy levels. Use it to study how a control pulse rotates a qubit, drives a
coupled gate, or moves population into the third level. You can run calibrated
gates or specify the control waveforms directly.

## Configure the transmon model { #create-the-emulator }

<a id="physics-model-and-calibration"></a>

### Model documents { #model-documents }

A model describes the transmons, their frequencies, and which pairs can be driven
by an exchange control. Load `transmon.reference` with
[`load_model_document()`][fatqat.emulator.load_model_document], then turn the
document into a runtime model with
[`TransmonModel.from_document()`][fatqat.emulator.TransmonModel.from_document].
Use [`available_model_documents()`][fatqat.emulator.available_model_documents] to
find other packaged models. You can also pass a decoded dictionary from your own
model file.

An edge permits a driven exchange on that pair. The resonant carriers come from
the model frequencies, and the exchange coefficient comes from the control
waveform. Keep the source document if you need the model ID, revision, or
parameters after creating the runtime model.

### Generate a transmon grid { #generate-transmon-grid-documents }

[`generate_transmon_grid_documents()`][fatqat.emulator.generate_transmon_grid_documents]
gives you a nearest-neighbor grid model and a matching reference calibration. Both
are editable dictionaries. The generated pulse parameters are analytic starting
values that you can refine for your model.

### Calibrated gates { #pulse-implementation-maps }

<a id="calibration-recipes"></a>

To run ordinary gates, the emulator uses a gate-to-pulse map. The built-in map
includes `RX`, `RY`, virtual `RZ`, `iSwap`, and `CZ`. To use your own calibration,
pass a complete document to
[`TransmonCalibration`][fatqat.emulator.TransmonCalibration], then give
[`default_transmon_gate_implementation_map()`][fatqat.emulator.default_transmon_gate_implementation_map]
the model and calibration. Pass the resulting map as `gate_implementation_map`.

## Address control channels { #direct-controls }

For a directly authored [`PulseOperation`][fatqat.operations.PulseOperation], use
`model.control` to address physical channels. `drive` and `detuning` take one
transmon label. `exchange` takes two labels on a declared coupling.
`model.frame(...)` selects the phase reference for a virtual drive update.

### Units { #units }

Times are in nanoseconds and the built-in waveforms use `rad/ns`. Convert a value
in GHz to an angular rate with
[`angular_rate_from_ghz()`][fatqat.emulator.superconducting.angular_rate_from_ghz].
The drive envelope is the full complex Rabi rate \(\Omega(t)\), with
\(\operatorname{Re}\Omega/2\) and \(\operatorname{Im}\Omega/2\) in the two
quadratures. Detuning and exchange use real Hamiltonian coefficients.

Use `model.available_controls` to inspect the available selector kinds and their
`coefficient_unit`. See [Pulse control](pulse-control/index.md) for the waveform
and timing interfaces.

## Run a transmon program { #run }

Each run starts in the physical ground state. Program qubits bind to
`model.subsystem_ids` in declaration order. To choose different positions, pass a
[`ResourceLayout`][fatqat.ResourceLayout] whose device labels are model subsystem
IDs. The [`run()`][fatqat.emulator.TransmonEmulator.run] entry lists the
scheduling and output options.

### Lindblad noise { #lindblad-noise }

Pass a `noise` model to add continuous evolution from background or
operation-scoped noise. Rates are in inverse nanoseconds. Rate-form depolarization
acts on all three levels and can populate the third level. For accepted forms and
model validation, see [Pulse emulator
noise](noise/backend-support.md#noise-emulator-support).

## Read transmon results { #simulation-methods }

With any [result method](emulators/index.md#choose-a-result), the result covers
all transmons in the model. The state-space dimension for `m` transmons is `3**m`,
even if the program addresses fewer qubits. The basis of each transmon is `("0",
"1", "2")`. Use `metadata["state_axes"]` for the order of the transmons.

Returned states and operators incorporate the final virtual-frame changes in the
model's rotating-frame basis. To compare an operator with an ideal qubit gate such
as `RZ`, allow for a global phase.

Measurement maps physical levels `0`, `1`, and `2` to the reported digits `0`,
`1`, and `1`, before readout noise is applied. To examine leakage into level `2`,
read the physical state from an unmeasured run. Reset prepares the physical ground
state.

<a id="neutral-atom-pulse-emulator"></a>

## API reference { #api }


::: fatqat.emulator.TransmonEmulator
    options:
      members:
        - "method"
        - "model"
        - "run"
        - "validate_noise_model"
      inherited_members: true
      show_bases: false
      merge_init_into_class: false

::: fatqat.emulator.TransmonModel
    options:
      members:
        - "from_document"
        - "basis_order"
        - "time_unit"
        - "subsystem_ids"
        - "control"
        - "available_controls"
        - "frame"
      inherited_members: false
      show_bases: false
      merge_init_into_class: false

::: fatqat.emulator.TransmonCalibration
    options:
      inherited_members: true
      show_bases: false
      merge_init_into_class: false
      filters:
        - "!^_"

::: fatqat.emulator.generate_transmon_grid_documents

::: fatqat.emulator.available_model_documents

::: fatqat.emulator.load_model_document

::: fatqat.emulator.default_transmon_calibration

::: fatqat.emulator.superconducting.angular_rate_from_ghz

::: fatqat.emulator.default_transmon_gate_implementation_map
