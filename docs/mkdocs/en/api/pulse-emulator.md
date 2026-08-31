---
title: "Superconducting pulse emulator"
---

# Superconducting pulse emulator


The [`TransmonEmulator`][fatqat.emulator.TransmonEmulator] runs a
[`Program`][fatqat.Program] against a three-level transmon model. It evolves
sampled controls over the full physical model, and `run()` returns an eager
[`Job`][fatqat.Job].

See [Emulate a superconducting system](../guide/transmon-emulation.md) for a
walkthrough covering calibrated gates and direct drive.

Ordinary gates use `gate_implementation_map`. A
[`PulseOperation`][fatqat.operations.PulseOperation] contains its own physical
channels and does not use that map. See [Pulse control](pulse-control/index.md) for direct
pulse authoring.

Unless a path is shown explicitly, the APIs on this page come from
`fatqat.emulator`; the GHz conversion helper comes from
`fatqat.emulator.superconducting`.

## Create the emulator


Load the packaged model and construct the emulator:

```python
import json
import fatqat as fq

model_document = fq.emulator.load_model_document("transmon.reference")
model = fq.emulator.TransmonModel.from_document(model_document)
backend = fq.emulator.TransmonEmulator(model)
```

To use an explicit calibration, build its gate map before constructing the
emulator:

```python
with open("calibration.json", encoding="utf-8") as stream:
    calibration = fq.emulator.TransmonCalibration(json.load(stream))
gate_map = fq.emulator.default_transmon_gate_implementation_map(
    model=model,
    calibration=calibration,
)
backend = fq.emulator.TransmonEmulator(
    model,
    gate_implementation_map=gate_map,
)
```

The packaged calibration is a reference configuration for simulation, not a
hardware calibration. To customize it, supply a complete calibration document
rather than a partial patch.

By default, qubits in a [`Program`][fatqat.Program] bind in declaration order
to `model.subsystem_ids`. Pass an explicit
[`ResourceLayout`][fatqat.ResourceLayout] to `run(resource_layout=...)` to
override that binding; its device labels must be model subsystem IDs.
Unaddressed transmons remain part of the full physical state and still
contribute factors of three to state and operator dimensions. The ordered
public identities of all model transmons appear in the result's `state_axes`
metadata.

`TransmonEmulator(...)` accepts these optional arguments:

**Constructor options**

| Argument | Meaning |
| --- | --- |
| `method` | Mathematical representation and method-native result field. Accepted case-insensitive values are `"statevector"` (the default), `"density_matrix"`, and `"unitary"`; `"SV"` and `"DM"` are aliases. |
| `noise` | A [`NoiseModel`][fatqat.NoiseModel]. `None` means no noise. |
| `gate_implementation_map` | A [`PulseImplementationMap`][fatqat.emulator.PulseImplementationMap] mapping operation families and device labels to pulse definitions. `None` uses the built-in map. |

## Execution methods


`method` names a mathematical representation, not the internal differential-
equation solver. The read-only `backend.method` property always returns the
canonical full name, and `result.metadata["method"]` records it.

**Method-native results**

| Method | Result accessor | Shape for `m` transmons | Meaning |
| --- | --- | --- | --- |
| `"statevector"` | [`get_statevector`][fatqat.Result.get_statevector] | `(3**m,)` | A pure coherent state, or one seeded trajectory when Lindblad noise can act. |
| `"density_matrix"` | [`get_density_matrix`][fatqat.Result.get_density_matrix] | `(3**m, 3**m)` | The exact ensemble state under supported Lindblad evolution. |
| `"unitary"` | [`get_unitary`][fatqat.Result.get_unitary] | `(3**m, 3**m)` | The complete coherent operator, including the canonical terminal virtual-frame transformation. |

All three representations cover the complete physical qutrit model, not just
the logical resources declared by the program. Pulse emulators do not expose
`superop` or internal solver names as methods.

An unmeasured statevector run with potentially active Lindblad noise is
stochastic, so the default result request retains metadata only. Request one
reproducible trajectory with
`result_config={"final_state": True}`, `shots=1`, and a non-negative
`simulation_config["seed"]`. Choose `method="density_matrix"` for the
exact ensemble instead.

Each transmon statevector trajectory is an independent physical evolution, so
the cost of a measured trajectory run grows with `shots`. Use
`method="density_matrix"` when an exact ensemble result is more useful.

`method="unitary"` is available through `run()` and
[`get_unitary`][fatqat.Result.get_unitary]; there is no separate public
propagator API. Unitary execution rejects measurement, reset, classical
conditions, counts, and programs in which resolved Lindblad noise can act
during nonzero-duration evolution. The check is deliberately structural:
background noise is considered active whenever any nonzero-duration block
exists, whereas operation-scoped noise is considered active only when attached
to such a block. Conditioned blocks and zero-rate declarations still count as
active; unmatched declarations and declarations scoped to zero-duration blocks
do not. The check does not analyze reachability, trajectories, jump
probability, or internal solver coefficients. Readout-only noise does not
affect an unmeasured unitary run.

The unitary uses per-subsystem near-resonant rotating frames and may differ
from a conventional qubit `RZ` by global phase; compare ideal operators up to
global phase.

## Run


[`run`][fatqat.emulator.TransmonEmulator.run] accepts these
`simulation_config` keys:

**`simulation_config` keys**

| Key | Type | Default | Effect and constraints |
| --- | --- | --- | --- |
| `seed` | `int` or `None`; not `bool` | `None` | Seed stochastic sampling for measurement, reset, readout, and statevector trajectories. Use a non-negative integer; `None` uses fresh entropy. |
| `schedule_mode` | `"ASAP"` or `"ALAP"` | `"ASAP"` | Place operations as early or as late as possible while preserving dependencies and physical-resource conflicts. |

These are the only two keys for `TransmonEmulator`. Pulse emulators reject
the matrix backend's `shot_parallelism`, `kernel_parallelism`, `max_workers`,
and `fusion` settings.

**`result_config` keys**

| Key | Type | Default | Effect and constraints |
| --- | --- | --- | --- |
| `counts` | `bool` or `None` | `None` | `True` requests sampled classical counts, `False` suppresses them, and `None` enables them when measurement exists. Counts require a positive integer `shots` value. |
| `final_state` | `bool` or `None` | `None` | `True` requests the method-native state or operator, `False` suppresses it, and `None` enables deterministic unmeasured output. A stochastic final state requires an explicit request and `shots == 1`. |

Both configuration arguments must be a `dict` or `None`; unknown keys
are rejected.

Every run begins in the product state with each transmon in physical
`|0>`. Pulse emulators do not accept an `initial_state` argument.

Measurement first samples a physical level, maps `0, 1, 2` to `0, 1, 1`,
then applies any classical readout-confusion matrix. Reset prepares physical
`|0>`.

Result metadata includes the effective run and result settings, but not the
model or calibration documents.

`run()` raises validation errors before returning a job. If execution fails
after a job is returned, `job.result()` raises
[`BackendExecutionError`][fatqat.errors.BackendExecutionError].

## Physics model and calibration


`TransmonModel.from_document(...)` accepts a decoded JSON-compatible model
mapping; direct model construction is not supported. The
[`TransmonCalibration`][fatqat.emulator.TransmonCalibration] constructor
separately accepts a decoded calibration mapping. Use
`load_model_document("transmon.reference")` for the packaged reference, or
`json.load` or another JSON reader for custom documents. Documents must use a
supported `format` ID and version. Missing or
unknown keys, unsupported versions, non-finite values, and values outside the
documented JSON-compatible types are rejected.

Control and frame addresses name model resources. Invalid addresses are
reported when you call `run()`.

The built-in model contains fixed qutrit transmons and an arbitrary undirected
coupling graph. A coupling declares where controlled exchange operations may
be driven; it is not a residual always-on exchange Hamiltonian. Frequencies
define the implicit resonant carriers.

### Model documents


Use [`available_model_documents`][fatqat.emulator.available_model_documents] to
discover the packaged model IDs and
[`load_model_document`][fatqat.emulator.load_model_document] to load one. Create
runtime models with
[`from_document`][fatqat.emulator.TransmonModel.from_document]; direct construction
is not supported.

Retain the decoded source documents when application code needs the persisted
model identity, physical parameters, or topology. In a model document,
`model` contains the ID and revision, `system.control_edges` contains the
coupling graph, and `parameters.subsystems` contains frequencies and
anharmonicities. The runtime model exposes ordered device labels through
`subsystem_ids`, but it does not expose normalized subsystem or coupling
records.

### Generate transmon grid documents

[`generate_transmon_grid_documents`][fatqat.emulator.generate_transmon_grid_documents]
is a shortcut for setting up a rectangular nearest-neighbor grid. It returns a
model document and a matching calibration document, so you can start a
simulation without writing either document by hand:

```python
import fatqat as fq

model_document, calibration_document = fq.emulator.generate_transmon_grid_documents(
    shape=(2, 2),
    frequency_groups_ghz=(5.0, 5.2),
)
```

Both values are ordinary JSON-compatible dictionaries. You can edit them or
save them with the standard `json` module.

| Argument | Meaning |
| --- | --- |
| `shape` | Grid dimensions as `(rows, columns)`. The grid must contain at least two transmons. |
| `frequency_groups_ghz` | The two idle-frequency centers in GHz. The generator assigns them in a checkerboard pattern. |
| `frequency_std_ghz` | Standard deviation of the normally distributed frequency variation in GHz. The default is `0.010`. |
| `anharmonicity_ghz` | Anharmonicity used for every transmon. The default is `-0.22`. |
| `seed` | Seed for reproducible frequency values. The default is `0`. |

The generator warns when the requested spread may make the two frequency
groups overlap. It also warns if the generated values overlap or reverse the
expected ordering on a neighboring pair. These warnings do not stop generation
or change the values.

The calibration contains fixed analytic starting values. The helper does not
run numerical calibration or optimization, start a simulation, or calculate
fidelity or leakage. See [Start with a transmon grid](../guide/transmon-emulation.md#start-with-a-transmon-grid)
for the complete workflow.

### Units


The runtime model exposes the pulse time coordinate:

**`model.time_unit` (`"ns"`)**

The coordinate for `PulseDefinition.duration` and for each
`PulseControl` waveform and `start_offset`.

Model and calibration documents store ordinary frequencies in GHz. Convert
document values to angular pulse rates with
[`fatqat.emulator.superconducting.angular_rate_from_ghz`][fatqat.emulator.superconducting.angular_rate_from_ghz].
Each control selector exposes its accepted waveform unit through
`selector.coefficient_unit`; the built-in drive, detuning, and exchange
selectors use `"rad/ns"`.

Transmon drive samples are the full complex Rabi rate \(\Omega(t)\). The
Hamiltonian uses \(\operatorname{Re}\Omega/2\) on the X quadrature and
\(\operatorname{Im}\Omega/2\) on the Y quadrature. Detuning and exchange
samples are their direct Hamiltonian coefficients.

`model.basis_order` is `("0", "1", "2")`. Use it to interpret flattened
physical results and derive the local dimension as `len(model.basis_order)`.

The `model.control` namespace selects the Hamiltonian mechanism, while
`model.frame(...)` selects a virtual-drive phase. The control-selector methods
are also available by name through the `model.available_controls` mapping.
Each mapping entry describes a supported control kind, not every fully bound
channel instance. Selectors expose `scope`, required `operands`,
`coefficient_domain`, and `coefficient_unit` for lightweight inspection.
Calling a selector returns a channel address. When you run the program, the
emulator checks resource names, declared pairs, waveform type, and values.

```python
drive = model.control.drive("q0")
detuning = model.control.detuning("q1")
exchange = model.control.exchange("q0", "q1")

assert model.available_controls["drive"] is model.control.drive

for name, selector in model.available_controls.items():
    print(name, selector.scope, selector.operands,
          selector.coefficient_domain, selector.coefficient_unit)
```

### Calibration recipes


The built-in calibration schema contains `rx_ry`, `iswap`, and per-edge
`cz` recipes. [`RZ`][fatqat.operations.RZ] is virtual and has no
calibration recipe.

The public scalar unit accessors `recipe_time_unit`,
`recipe_frequency_unit`, and `recipe_dimensionless_unit` describe the
stored recipe quantities. They are distinct from the model's pulse
coordinate `time_unit` and each control selector's `coefficient_unit`.

## Pulse implementation maps


A [`PulseImplementationMap`][fatqat.emulator.PulseImplementationMap] realizes ordinary gates.
The transmon constructor names this capability `gate_implementation_map`;
direct controls bypass it. The
[`default_transmon_gate_implementation_map`][fatqat.emulator.default_transmon_gate_implementation_map]
builder returns a new map containing the built-in `RX`, `RY`, `RZ`, `iSwap`,
and `CZ` rules for one model and calibration.

A map built by `default_transmon_gate_implementation_map` is tied to the
model's subsystem labels, anharmonicities, coupling topology, and selected CZ
endpoints. Build a new map after changing any of these values. Model identity,
revision, idle frequencies, declaration order, and edge IDs do not require a
new map. This compatibility check does not apply to maps written by users.

See [Gate realization](pulse-control/gate-realization.md) for accepted rule forms and errors.

## Direct controls


The same model channels can be used without a gate-realization rule.
Drive and detuning resolve one declared transmon; exchange resolves two
transmons and their declared coupling. Drive accepts a full complex Rabi-rate
envelope for the two quadratures, while detuning and exchange require real
values. Pulse times use the model units described above. The current transmon
model does not add amplitude or duration limits beyond requiring finite values.

See [PulseOperation](pulse-control/pulse-operation.md),
[PulseControl](pulse-control/pulse-control.md), and
[SampledWaveform](pulse-control/sampled-waveform.md) for construction and timing.
`iSwap` is a gate whose built-in realization uses exchange;
`iSwap` is not a channel name.

## Lindblad noise


Pass supported declarations through `noise=`. The transmon emulator provides
the collapse-operator realizations documented in
[Pulse emulators](noise/backend-support.md#noise-emulator-support), including
[`AmplitudeDamping`][fatqat.noise.AmplitudeDamping],
[`PhaseDamping`][fatqat.noise.PhaseDamping], and
[`ThermalRelaxation`][fatqat.noise.ThermalRelaxation], plus rate-form
[`Depolarizing`][fatqat.noise.Depolarizing]. Qutrit amplitude damping requires two
adjacent-level rates. Depolarization acts on the full three-level space and can
populate `|2>`. Rates use inverse nanoseconds, while `t1`, `t2`, and
`t_phi` use nanoseconds. The emulator accepts both background declarations and
declarations scoped to ordinary operations. Finite probability forms, `Loss`,
and nonlocal declarations are rejected.

Probability-form channels are not converted to rates. In particular,
[`PauliChannel`][fatqat.noise.PauliChannel] remains Simulator-only. See
[Continuous-time noise](pulse-control/index.md#pulse-probability-noise). Readout confusion is classical and must use a
binary `2 x 2` matrix; construction and `validate_noise_model()` reject
larger matrices before a measured run.

Call [`validate_noise_model`][fatqat.emulator.TransmonEmulator.validate_noise_model] before
running a program to validate its noise model. Program-specific selectors are
checked at run time.

## Neutral-atom pulse emulator


The two-level atom backend also accepts an optional gate implementation map.
It has its own continuous-noise implementations. `Atom2LevelEmulator` has an
empty built-in gate map and global direct controls; user-supplied maps can add
gate rules. See [Neutral-atom emulator](atom-emulators.md) for its API and
[Choose and run a neutral-atom workflow](../guide/neutral-atom-emulation.md)
for the complete workflow.

## API


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
