---
title: "Neutral-atom emulator"
---

# Neutral-atom emulator


[`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator] models atoms at fixed
positions with a ground level and a Rydberg level. Use it to study how a laser
pulse drives the whole array and how the distance between atoms changes their
interactions.

<a id="two-level-atom-emulator"></a>

## Configure the model and arrangement { #arrangements-and-program-resources }

The model defines the energy levels, controls, and interaction law. The
arrangement defines where the atoms sit. Load `atom2level.reference` with
[`load_model_document()`][fatqat.emulator.load_model_document] and pass the
decoded model document to
[`Atom2LevelModel.from_document()`][fatqat.emulator.Atom2LevelModel.from_document].

Use [`AtomArrangement.chain()`][fatqat.emulator.AtomArrangement.chain] or
[`rectangular()`][fatqat.emulator.AtomArrangement.rectangular] for regular
layouts. Use
[`from_coordinates()`][fatqat.emulator.AtomArrangement.from_coordinates] for
explicit 2D or 3D sites and [`combine()`][fatqat.emulator.AtomArrangement.combine]
to join labeled layouts. Coordinates are in micrometres, and their order defines
the site indices.

Declare one qubit per site in the program. The default binding uses program
declaration order. An explicit [`ResourceLayout`][fatqat.ResourceLayout] must
cover every site exactly once.

## Address global controls { #model-and-controls }

Both global channels act on every atom. Use `model.control.drive()` for a complex
Rabi envelope and `model.control.detuning()` for a real detuning waveform. Time is
in microseconds and both waveforms use `rad/us`. Add the
[`PulseOperation`][fatqat.operations.PulseOperation] directly to the program.

Place measurements after the pulse blocks. The two-level emulator runs
unconditional operations. To use an ordinary gate, supply its pulse rule in
`gate_implementation_map`.

Use [Pulse control](pulse-control/index.md) to specify the waveform and timing.
For an example that compares drives at different atom separations, see [Emulate
neutral atoms with laser pulses](../guide/neutral-atom-emulation.md).

## Run on an atom arrangement { #run-configuration-and-results }

Each run starts with every atom in the ground level. Use
[`run()`][fatqat.emulator.Atom2LevelEmulator.run] to schedule the program and
request output. The per-run interaction cutoff lets you change which couplings the
Hamiltonian includes.

### Interaction cutoff { #interaction-cutoff }

Set `execution_config={"interaction_cutoff": ...}` to keep only pairs at or below
a distance in micrometres. `None` keeps the full interaction Hamiltonian and `0.0`
disables pair interactions. On a rectangular grid with equal spacing, using that
spacing as the cutoff keeps the horizontal and vertical nearest neighbors.

### Lindblad noise { #lindblad-noise }

For continuous noise, register a background source for each site you want to
affect. Rates are in inverse microseconds and relaxation times are in
microseconds. See [Pulse emulator
noise](noise/backend-support.md#noise-emulator-support) for the supported channel
forms and readout noise.

## Read atom array results { #read-atom-array-results }

With any [result method](emulators/index.md#choose-a-result), every site
contributes two basis levels in the order `("g", "r")`. The state dimension is
`2**N` for `N` sites. Use `metadata["state_axes"]` for the order of the sites in
the returned state or operator.

## API reference { #api }


::: fatqat.emulator.AtomArrangement
    options:
      members:
        - "from_coordinates"
        - "chain"
        - "rectangular"
        - "combine"
        - "group"
        - "label"
        - "groups"
        - "row_groups"
        - "column_groups"
        - "rows"
        - "cols"
        - "spacing"
        - "coordinates"
        - "spatial_dimension"
        - "num_sites"
        - "distance_unit"
      inherited_members: true
      show_bases: false
      merge_init_into_class: false

::: fatqat.emulator.Atom2LevelEmulator
    options:
      members:
        - "method"
        - "model"
        - "arrangement"
        - "run"
        - "validate_noise_model"
      inherited_members: true
      show_bases: false
      merge_init_into_class: false

::: fatqat.emulator.Atom2LevelModel
    options:
      members:
        - "from_document"
        - "control"
        - "available_controls"
        - "basis_order"
        - "time_unit"
      inherited_members: false
      show_bases: false
      merge_init_into_class: false
