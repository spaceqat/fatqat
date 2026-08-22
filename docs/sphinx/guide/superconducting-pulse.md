# Superconducting transmon emulation

`fq.emulator.TransmonEmulator` realizes a small native superconducting gate set
as calibrated controls on fixed three-level transmons. It is a physical
simulator, separate from the two-level IBM- and Google-style fake backends.
The public API accepts ordinary data and NumPy results: QuTiP and qutip-qip
objects remain implementation details.

Gate-authored programs and direct controls are separate capabilities. This
system supports both: gates use a gate implementation map, while direct
`PulseOperation` values already contain model-authored structural controls and
bypass it. The private bound target validates those addresses during shared
preparation.

This guide explains the workflow. See {doc}`../api/pulse-emulator` for exact
constructor and method signatures, model/calibration values, pulse-rule
contracts, and generated reference documentation.

## Create a backend

The common workflow needs only the hardware model. The model declares ordered
transmon IDs, frequencies, anharmonicities, and coupling edges; the emulator
compiles a nominal package calibration internally.

Each declared frequency is the nominal 0-to-1 transition frequency and
defines that transmon's implicit resonant rotating-frame carrier. The current
solver uses `Delta_i = 0`, so changing a frequency alone does not numerically
change the simulated dynamics. A future frame-explicit model can consume it
under a new model version.

```python
import fatqat as fq

model_document = fq.emulator.load_model_document("transmon.reference")
print(model_document["model"])
print(model_document["units"])
print(model_document["parameters"])
print(model_document.get("references", []))
model = fq.emulator.TransmonModel.from_document(model_document)
assert tuple(model.available_controls) == ("drive", "detuning", "exchange")
backend = fq.emulator.TransmonEmulator(model)
```

The packaged Transmon snapshot is explicitly synthetic and is not a device
calibration. Copy and edit the returned dictionary when authoring a model, and
change its identity and revision whenever physical values change. Add
`references` only when useful citations exist.

For an explicit calibration, keep it as a separate complete JSON-compatible
document, compile a map, and pass only that map to the emulator:

```python
calibration = fq.emulator.TransmonCalibration(calibration_document)
gate_map = fq.emulator.default_transmon_gate_implementation_map(
    model=model,
    calibration=calibration,
)
backend = fq.emulator.TransmonEmulator(
    model,
    gate_implementation_map=gate_map,
)
```

The calibration document has its own durable identity but no model field.
The source model contributes pulse-design facts such as anharmonicity and
structural controls when the map is built.

A compiled map may be reused on a distinct richer target model when every
structural label and edge referenced by its definitions still exists. That
reuses the coarse-source pulse design; it does not silently redesign DRAG from
the richer target's anharmonicity. Rebuild the map with the richer source
model when redesign is what you intend.

The model document uses structured format identity
`{"id": "sc.transmon_exchange", "version": 1}`, ordered
`system.subsystems`, and `system.control_edges`. Frequency and
anharmonicity units are top-level quantity kinds. Calibration recipe fields
use unsuffixed persisted names such as `duration`, `ramp_duration`, and
`detuning`; the Python runtime exposes unit-explicit conversion properties.

By default, the program's qubits bind to model subsystems in declaration
order. Pass ``resource_layout=ResourceLayout({...})`` to ``run()`` or
``propagator()`` to bind program refs to selected transmon IDs instead. The
backend—not the user—allocates numerical tensor axes over the complete model.
Additional, unaddressed transmons therefore remain part of drift, noise,
physical evolution, and final-state output.

## Native operations and results

The backend accepts the calibrated physical gates `RX`, `RY`, `iSwap`, and
ordered `CZ`, plus `RZ`, which realizes as an exact virtual frame rotation
with no physical control and no calibration degree of freedom. `iSwap` and
`CZ` require a declared model coupling. Both orders of each declared edge are
compiled; an ordered CZ override applies only to its exact operand tuple, with
the calibration's default recipe used otherwise.

The built-in `CZ` realization calculates its nominal virtual frame correction
by integrating the generated detuning waveform. This keeps the correction
consistent with arbitrary gate and ramp durations without storing a duplicate
phase in calibration. It is a first-version model correction rather than a
hardware-calibrated phase: device-specific phase calibration can further
improve gate quality in a future calibration layer.

```python
program = fq.Program(2, 1)
program.add(fq.ops.RX(0.4), 0)
program.add(fq.ops.iSwap, (0, 1))
program.measure(0, 0)

result = backend.run(
    program,
    shots=100,
    simulation_config={"seed": 7, "schedule_mode": "ASAP"},
    result_config={"counts": True},
).result()
print(result.get_counts())
```

Request `result_config={"final_state": True}` for a NumPy density matrix.
It is the full physical qutrit state, including leakage and every transmon in
the selected model. For `m = len(model.subsystems)`, its shape is
`(3**m, 3**m)`, not `(2**n, 2**n)` for the program's qubits. A run containing
measurement may export this sampled posterior only with `shots=1`.
``result.metadata["state_axes"]`` lists those full-model factors in backend
allocation order using transmon IDs. Only addressed transmons also carry a
program ``RegisterRef``; unaddressed transmons carry ``None``. No private
numerical axis is exposed.

The accepted result request keys are `counts` and `final_state`. By default,
counts are requested when the program contains measurement, while a final
state is requested only for a program without measurement.

## Direct drive, detuning, and exchange controls

Model factories also support model-neutral direct operations. Complex drive
samples encode quadratures; detuning and exchange use their model-defined
rates. Each structural control address identifies its subsystem or edge, so
the operation has no ordinary program targets:

```python
duration = 20.0
controls = (
    fq.emulator.PulseControl(
        model.control.drive("q0"),
        fq.waveforms.SampledWaveform((0.0, duration), (0.02, 0.02j)),
    ),
    fq.emulator.PulseControl(
        model.control.exchange("q0", "q1"),
        fq.waveforms.SampledWaveform((0.0, duration), (0.01, 0.01)),
    ),
)
direct_program = fq.Program(2)
direct_program.add(fq.ops.PulseOperation(duration, controls))
```

Direct blocks can coexist with calibrated gates. `iSwap` remains a gate whose
built-in realization drives the `exchange` control; `iSwap` is not a channel
name. Structural control addresses can be reused with compatible model
instances.

## Coherent propagators

Use `backend.propagator(program)` to obtain the complete coherent program
propagator as a complex NumPy array in the full model Hilbert space. The
method includes virtual frame updates by default, so a terminal virtual `RZ`
and the nominal `CZ` frame correction appear in the returned operation even
though neither adds a physical control pulse. Intermediate frame updates
always rotate later phase-sensitive controls.

```python
unitary = backend.propagator(program)
raw_dynamics = backend.propagator(program, apply_final_frame=False)
```

`apply_final_frame=False` omits only the terminal frame transformation; it
does not disable frame updates that affect later controls. As with final
states, a model containing `m` transmons returns a `(3**m, 3**m)` array.
The array uses the same per-subsystem near-resonant rotating-frame convention
reported by executed results in
`result.metadata["solver"]["frame_convention"]`, rather than a
laboratory-frame Hamiltonian. Virtual-Z frame updates also use
`diag(1, exp(i*angle), ...)`;
on the computational subspace this differs from the conventional `RZ` gate
matrix by a global phase. Compare propagators phase-invariantly when checking
them against ideal circuit matrices.

Measurement, reset, and classical conditions are rejected because they do not
have a single unitary propagator. Bound collapse terms are rejected when the
program has nonzero elapsed pulse evolution. Rate-based noise has no effect
on a frame-only, zero-duration program because no time elapses.
`schedule_mode` may be `"ASAP"` or `"ALAP"`.

## Custom gate realizations

Each native operation resolves to its physical pulse recipe through a
`gate_implementation_map=`. Build a fresh standard map from a model and
calibration, then replace one gate's realization to change *how* a gate is
physically executed - the waveform shape, which control channels are
driven, which model resources are claimed - without editing calibration
data or subclassing `TransmonEmulator`:

The map object is still named `PulseImplementationMap`; the constructor
keyword is gate-specific because the map is not involved in direct control.

```python
def custom_cz(operation, *, device_operands):
    first, second = device_operands
    ...
    return fq.emulator.PulseDefinition(
        duration=...,
        controls=(...,),
    )

calibration = fq.emulator.TransmonCalibration(calibration_document)
implementations = fq.emulator.default_transmon_gate_implementation_map(
    model=model,
    calibration=calibration,
)
implementations.remove(fq.ops.CZ)
implementations.add(fq.ops.CZ, custom_cz)
backend = fq.emulator.TransmonEmulator(
    model, gate_implementation_map=implementations
)
```

Changing a calibrated *number* (a gate duration, a DRAG coefficient, a
per-edge detuning) is a calibration-document change. Changing the physical
*mechanism* a gate uses is an implementation-map change; the calibration
document itself is never a place to select executable behavior. Supply a
complete separate custom document rather than patching the package default.
The package default is a nominal simulation baseline, not a hardware-fidelity
guarantee. See
[Advanced user topics](advanced.md) for the matrix-family version of this
same pattern, and {doc}`../api/experimental` for the full rule contract,
registration modes, and error semantics.

## Dynamic execution and timing

Measurement collapses the physical qutrit. Its reported classical bit maps
physical outcomes as `0 -> 0`, `1 -> 1`, and `2 -> 1`; any readout confusion
then resamples that reported bit. Reset prepares the selected transmon's
physical `|0>` state. Execution is serial per shot, so a later classical
guard sees that persisted reported (and possibly confused) value, while later
pulses retain the accumulated virtual-frame state.

`schedule_mode="ASAP"` is the default. The private scheduler places pulse
blocks as early as their program dependencies and claimed resources permit.
`"ALAP"` instead places them as late as possible within the same ASAP
makespan. Both modes preserve dependency order and resource exclusivity; they
are simulation controls, not a public hardware schedule artifact.

Pulse execution is serial in v0.1, so the emulator takes no parallelism
settings at all — the matrix backend's `parallel_mode`, `max_workers`, and
`numba_parallel` are rejected rather than silently ignored. Its matrix-only
`fusion` setting is rejected for the same reason. Seeded runs are reproducible.

## Noise support

Use {py:meth}`~fatqat.NoiseModel.add` without `operation=` to apply local
background qutrit noise, including idle time. A background declaration must
name exactly one target:

```python
noise.add(fq.noise.ThermalRelaxation(t1=60.0, t2=80.0), targets="q0")
noise.add(
    fq.noise.AmplitudeDamping(rate=(0.001, 0.002)),
    targets="q1",
)
```

Classical {py:class}`~fatqat.noise.ReadoutConfusion` is also supported and is
intrinsically measurement-bound.

Provide `operation=...` to scope a generator to matching pulse blocks:

```python
noise.add(
    fq.noise.PhaseDamping(rate=0.002),
    operation=fq.ops.X,
    targets="q0",
)
```

Pulse emulators use authored generator/time forms directly. They reject the
built-in finite `p` forms and do not convert them with the realized block
duration. The block duration controls how long the resolved generator evolves,
not how the descriptor is interpreted. Operation-bound collapse operators are
active only over their gate's placed interval; a conditionally disabled gate
contributes neither control nor attached noise. Background noise remains active
through elapsed scheduled time, including a disabled block's reserved window.

Background and operation-specific registrations compose rather than replace
one another. The default map accepts `ThermalRelaxation(t1, t2)` in either
scope. Other descriptors require an explicit local generator rule in the
supplied {py:class}`~fatqat.noise.LindbladImplementationMap`. Coherent ZZ is
not in the default map. See {doc}`noise` for scope conflicts and backend
parameter boundaries.
