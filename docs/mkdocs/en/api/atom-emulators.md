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
global controls in the physical `|g>, |r>` model. See
[Choose and run a neutral-atom workflow](../guide/neutral-atom-emulation.md)
for a complete executable example.

## Arrangements and program resources


The backend requires a regular
[`AtomArrangement`][fatqat.emulator.AtomArrangement]. Coordinates are row-major,
`(column * spacing, row * spacing, 0)`, and the atom model interprets
spacing in micrometres. A program must declare exactly one dimension-two
quantum resource per site; declaration order binds resources to coordinates.
The arrangement describes fixed geometry; it does not track atom loading or
loss.
`arrangement.num_sites` and `len(arrangement)` both return the site count,
which must exactly match the number of quantum resources declared by a pulse
program.
By contrast, [`AtomArraySimulator`][fatqat.simulator.AtomArraySimulator] takes
its gate-level site count directly from the quantum resources declared by each
[`Program`][fatqat.Program]. It has no separate capacity argument.

## Run configuration and results


The `run()` method has the signature `(program, *, shots=1024, resource_layout=None, simulation_config=None, result_config=None)`. The
optional layout must still cover every arrangement site exactly once; the
default uses declaration order. Validation errors are raised
directly from `run()` before a job is returned. A failure after execution
starts is represented by a failed job, and `job.result()` raises
[`BackendExecutionError`][fatqat.errors.BackendExecutionError].

`simulation_config` accepts the shared pulse controls plus an Atom2-specific
Hamiltonian cutoff:

**Simulation configuration**

| Key | Type | Default | Effect and constraints |
| --- | --- | --- | --- |
| `seed` | `int` or `None`; not `bool` | `None` | Random seed for measurement, readout, and statevector trajectory sampling. Integers must be non-negative; `None` chooses a fresh seed. |
| `schedule_mode` | `"ASAP"` or `"ALAP"` | `"ASAP"` | Place operations as early or as late as their dependencies allow. |
| `interaction_cutoff` | finite nonnegative `Real` or `None`; not `bool` | `None` | For this run, retain interaction pairs at or below this distance in the arrangement's distance unit. `None` keeps all pairs; `0.0` disables pair interactions. |

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
and its corresponding Result accessor, not an internal solver:

**Final-state representations**

| Method | Result accessor | Shape for `N` sites | Interpretation |
| --- | --- | --- | --- |
| `"statevector"` | [`get_statevector`][fatqat.Result.get_statevector] | `(d**N,)` | A pure coherent state, or one sampled posterior/trajectory when the execution is stochastic. |
| `"density_matrix"` | [`get_density_matrix`][fatqat.Result.get_density_matrix] | `(d**N, d**N)` | The exact ensemble state under supported Lindblad evolution. |
| `"unitary"` | [`get_unitary`][fatqat.Result.get_unitary] | `(d**N, d**N)` | The complete coherent operator in the canonical terminal frame. |

Here `d=2` for [`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator]. Its
results use FATQAT and NumPy types; the public API does not expose QuTiP
objects, `superop`, or internal solver names.

Measurement and potentially active Lindblad noise make a retained final state
stochastic. In those cases, the default request returns counts when the
program contains measurement and otherwise returns metadata only. Request a
single seeded final state explicitly with
`result_config={"final_state": True}`, `shots=1`, and
`simulation_config={"seed": ...}`. Select `method="density_matrix"` for
an exact supported Lindblad ensemble.

For unitary execution, construct the backend with `method="unitary"`, call
`run()`, and retrieve the result with
[`get_unitary`][fatqat.Result.get_unitary]. Unitary execution rejects
measurement, reset, conditions, counts, and potentially active Lindblad
evolution. There is no separate public propagator API.

## Two-level atom emulator


[`Atom2LevelEmulator`][fatqat.emulator.Atom2LevelEmulator] requires an
`Atom2LevelModel` and an arrangement. Its global drive and detuning channels
act on every site; see [PulseOperation](pulse-control/pulse-operation.md) for how to add a
direct pulse block. A terminal measurement may follow the pulse program, and
barriers are ignored. The built-in gate map is empty, so ordinary gates
require a custom map. Reset, conditions, per-site controls, mid-circuit
measurement, and pulses after measurement are not supported.

### Model and controls


The runtime model exposes basis order `("g", "r")`, the pulse time unit, and
global control selectors, but it contains no geometry or calibration. Retain
the decoded source document when application code needs persisted species
data, state labels, signed `C6`, interaction-law metadata, parameter units, or
channel bounds. Derive the local dimension as `len(model.basis_order)`.

Create instances with
[`from_document`][fatqat.emulator.Atom2LevelModel.from_document]; direct
construction is not supported.

The global drive accepts a complex [`SampledWaveform`][fatqat.emulator.SampledWaveform];
its values encode amplitude and phase together. The global detuning accepts
real samples. Both controls use `rad/us` and apply to every arrangement site.
At runtime, read this unit from each selector's `coefficient_unit` property.

### Interaction cutoff


Set `interaction_cutoff` per run through `simulation_config`. The default
`None` keeps every pair and preserves the complete `C6/R^6` Hamiltonian. A
finite nonnegative cutoff keeps pairs whose Euclidean distance is at or below
that value in `arrangement.distance_unit` (currently micrometres); `0.0`
disables pair interactions. For a rectangular arrangement,
`simulation_config={"interaction_cutoff": arrangement.spacing}` keeps only
horizontal and vertical nearest pairs. This is a numerical Hamiltonian
truncation, not a physical blockade radius. One emulator can therefore be
reused to compare several truncations without rebuilding its model or
arrangement.

### Lindblad noise


The built-in forms are listed under
[Pulse emulators](noise/backend-support.md#noise-emulator-support). Each
background registration names one site; enumerate sites explicitly to apply
the same noise at several sites. Rates use inverse microseconds and relaxation
times use microseconds. Finite-probability (`p`) forms are not converted into
rates from the pulse duration.
Binary [`ReadoutConfusion`][fatqat.noise.ReadoutConfusion] is a classical report channel
applied only to the reported digit after physical collapse, not a Lindblad
operator.

The emulator provides built-in amplitude damping, phase damping, thermal
relaxation, and depolarizing noise. These forms accept background declarations
only; operation-scoped continuous noise is unsupported.

With `method="statevector"`, resolved Lindblad noise that can act during a
nonzero-duration block uses seeded trajectories. Because the atom emulator
accepts background declarations only, a declaration is conservatively
considered active whenever a nonzero-duration block exists. Zero-rate
declarations are still considered active. Use `method="density_matrix"` for
the exact ensemble. A zero-duration measured program samples the initial
state without time evolution.

See [Pulse control](pulse-control/index.md) for direct pulse authoring. For the
complete two-level workflow, see
[Choose and run a neutral-atom workflow](../guide/neutral-atom-emulation.md).

## API


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
