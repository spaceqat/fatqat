---
title: "Neutral-atom emulator"
---

# Neutral-atom emulator


The neutral-atom pulse emulator follows the standard
[`Simulator`][fatqat.simulator.Simulator] workflow. Pass a
[`Program`][fatqat.Program] to `run()`, then call `job.result()` on the
eager [`Job`][fatqat.Job] to get a [`Result`][fatqat.Result]. It is a
pulse-resolved physical emulator rather than a mode of
[`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator].

Use [`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator] for directly authored
global controls in the physical `|g>, |r>` model. The executable workflow is
in [Choose and run a neutral-atom workflow](../guide/neutral-atom-emulation.md).

## Arrangements and program resources


The backend requires a regular
[`AtomArrangement`][fatqat.emulator.AtomArrangement]. Coordinates are row-major,
`(column * spacing, row * spacing, 0)`, and the current atom model interprets
spacing in micrometres. A program must declare exactly one dimension-two
quantum resource per site; declaration order binds resources to coordinates.
The arrangement describes fixed geometry; it does not track atom loading or
loss.
`arrangement.num_sites` and `len(arrangement)` both return the coordinate
count, which a pulse program must match exactly.
By contrast, `AtomArraySimulator(num_sites=6)` declares a maximum gate-level
device capacity and accepts programs with at most six resources; omitting its
`num_sites` argument leaves that simulator unbounded.

::: fatqat.emulator.AtomArrangement
    options:
      members:
        - "chain"
        - "rectangular"
        - "num_sites"
        - "distance_unit"
      inherited_members: true
      show_bases: false
      merge_init_into_class: false

## Run configuration and results


The `run()` method has the signature `(program, *, shots=1024, resource_layout=None, simulation_config=None, result_config=None)`. The
optional layout must still cover every arrangement site exactly once; the
default uses declaration order. Validation errors are raised
directly from `run()` before a job is returned. A failure after execution
starts is represented by a failed job, and `job.result()` raises
[`BackendExecutionError`][fatqat.errors.BackendExecutionError].

`simulation_config` accepts only `seed` and `schedule_mode`:

**Simulation configuration**

| Key | Type | Default | Effect and constraints |
| --- | --- | --- | --- |
| `seed` | `int` or `None`; not `bool` | `None` | Random seed for measurement, readout, and statevector trajectory sampling. Integers must be non-negative; `None` chooses a fresh seed. |
| `schedule_mode` | `"ASAP"` or `"ALAP"` | `"ASAP"` | Place operations as early or as late as their dependencies allow. |

`result_config` accepts only these keys:

**Result configuration**

| Key | Type | Default | Effect and constraints |
| --- | --- | --- | --- |
| `counts` | `bool` or `None` | `None` | `True` requests classical counts, `False` suppresses them, and `None` enables them when measurement exists. Counts require a positive integer `shots` value. |
| `final_state` | `bool` or `None` | `None` | `True` requests the method-native state or operator, `False` suppresses it, and `None` enables deterministic unmeasured output. A stochastic final state requires an explicit request and `shots == 1`. |

Both configuration arguments must be a `dict` or `None`; unknown keys
are rejected.

Each run starts from the fixed product state `|g>` on every site. The
constructor does not accept an `initial_state` argument.

The constructor accepts the case-insensitive values `method="statevector"`
(the default), `"density_matrix"`, or `"unitary"`. `"SV"` and `"DM"`
are aliases.
The read-only `backend.method` property and `result.metadata["method"]`
use the canonical full name. A method selects the mathematical representation
and predictable Result accessor, not an internal solver:

**Final-state representations**

| Method | Result accessor | Shape for `N` sites | Interpretation |
| --- | --- | --- | --- |
| `"statevector"` | [`get_statevector`][fatqat.Result.get_statevector] | `(d**N,)` | A pure coherent state, or one sampled posterior/trajectory when the execution is stochastic. |
| `"density_matrix"` | [`get_density_matrix`][fatqat.Result.get_density_matrix] | `(d**N, d**N)` | The exact ensemble state under supported Lindblad evolution. |
| `"unitary"` | [`get_unitary`][fatqat.Result.get_unitary] | `(d**N, d**N)` | The complete coherent operator in the canonical terminal frame. |

Here `d=2` for [`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator]. The backend
does not expose QuTiP values, `superop`, or internal solver names as public
methods.

Measurement makes a retained final state stochastic. Potentially active
Lindblad noise does too. In those cases the default request returns counts when
measurement exists and otherwise metadata only; request a single seeded final
state explicitly with
`result_config={"final_state": True}`, `shots=1`, and
`simulation_config={"seed": ...}`. Select `method="density_matrix"` for
an exact supported Lindblad ensemble.

Use `method="unitary"` through `run()` and
[`get_unitary`][fatqat.Result.get_unitary]. It rejects measurement, reset,
conditions, counts, and potentially active Lindblad evolution. There is no
separate public propagator API.

## Two-level atom emulator


[`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator] requires an
`Atom2LevelModel` and an arrangement. Its global drive and detuning channels
act on every site; see [PulseOperation](pulse-control/pulse-operation.md) for how to add a
direct pulse block. A terminal measurement may follow the pulse program, and
barriers are ignored. The built-in gate map is empty, so ordinary gates
require a custom map. Reset, conditions, per-site controls, mid-circuit
measurement, and pulses after measurement are not supported.

### Construction and execution


::: fatqat.emulator.Atom2LevelEmulator
    options:
      members:
        - "method"
        - "model"
        - "arrangement"
        - "interaction_cutoff"
        - "run"
        - "validate_noise_model"
      inherited_members: true
      show_bases: false
      merge_init_into_class: false

### Model and controls


The runtime model exposes basis order `("g", "r")`, the pulse time unit,
and global control selectors. It contains no geometry or calibration. Retain
the decoded source document when application code needs persisted species,
state labels, signed `C6`, interaction-law metadata, parameter units, or
channel bounds. Derive the local dimension as `len(model.basis_order)`.

### class `fatqat.emulator.Atom2LevelModel` { #fatqat.emulator.Atom2LevelModel }

Create instances with
[`from_document`][fatqat.emulator.Atom2LevelModel.from_document]; direct
construction is not supported.

::: fatqat.emulator.Atom2LevelModel.from_document

::: fatqat.emulator.Atom2LevelModel.control

::: fatqat.emulator.Atom2LevelModel.available_controls

::: fatqat.emulator.Atom2LevelModel.basis_order

::: fatqat.emulator.Atom2LevelModel.time_unit

The global drive accepts a complex [`SampledWaveform`][fatqat.emulator.SampledWaveform];
its complex values encode amplitude and phase together. The global detuning
accepts real samples. Both use `rad/us` and apply to every arrangement site.
The selector's `coefficient_unit` property is the runtime source for that
unit.

### Interaction cutoff


The default `interaction_cutoff=None` keeps every pair and preserves the
complete `C6/R^6` Hamiltonian. A finite nonnegative cutoff keeps pairs whose
Euclidean distance is at or below that value in
`arrangement.distance_unit` (currently micrometres); `0.0` disables pair
interactions. For a rectangular
arrangement, `interaction_cutoff=arrangement.spacing` keeps only horizontal
and vertical nearest pairs. This is a numerical Hamiltonian truncation, not a
physical blockade radius.

### Lindblad noise


The built-in forms are listed at [Pulse emulators](noise/backend-support.md#noise-emulator-support). Each background
registration names one site; enumerate sites explicitly to apply the same noise
at several sites. Rates use inverse microseconds and relaxation times use
microseconds. Finite `p` forms are not converted with a pulse duration.
Binary [`ReadoutConfusion`][fatqat.noise.ReadoutConfusion] is a classical report channel
applied only to the reported digit after physical collapse, not a Lindblad
operator.

The family-owned built-ins are amplitude damping, phase damping, thermal
relaxation, and depolarizing noise. They accept background declarations only;
operation-scoped continuous noise is unsupported.

With `method="statevector"`, resolved Lindblad noise that can act during a
nonzero-duration block uses seeded trajectories. Because this family accepts
background declarations only, one is conservatively considered active when
any nonzero-duration block exists. Zero-rate declarations still count. Use
`method="density_matrix"` for the exact ensemble. A zero-time measured
program samples the initial state without time evolution.

See [Pulse control](pulse-control/index.md) for direct pulse authoring and
[Choose and run a neutral-atom workflow](../guide/neutral-atom-emulation.md) for the complete two-level workflow.
