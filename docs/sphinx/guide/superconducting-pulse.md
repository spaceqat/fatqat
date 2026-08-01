# Superconducting pulse simulation

`fq.backends.PulseBackend` realizes a small native superconducting gate set
as calibrated controls on fixed three-level transmons. It is a physical
simulator, separate from the two-level IBM- and Google-style fake backends.
The public API accepts ordinary data and NumPy results: QuTiP and qutip-qip
objects remain implementation details.

## Create a backend

Keep the hardware model and its calibration as separate JSON-compatible
documents. The model declares ordered transmon IDs, frequencies,
anharmonicities, and coupling edges. The calibration names the exact model
identity and supplies gate durations, DRAG settings, and per-edge CZ detuning
values.

```python
import fatqat as fq

model = fq.backends.load_physics_model(model_document)
calibration = fq.backends.load_calibration_spec(calibration_document, model)
backend = fq.backends.PulseBackend(model, calibration)
```

The program's qubits bind to model subsystems in declaration order. A model
may contain additional, unused transmons; they remain part of physical
evolution and final-state output.

## Native operations and results

The backend accepts the calibrated physical gates `RX`, `RY`, `iSwap`, and
oriented `CZ`, plus `RZ`, which realizes as an exact virtual frame rotation
with no physical control and no calibration degree of freedom. `iSwap` and
`CZ` require a declared model coupling. For `CZ`, program target order must
match the calibration's detuning orientation.

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
    simulation_config={"seed": 7, "placement_mode": "ASAP"},
    result_config={"counts": True},
).result()
print(result.get_counts())
```

Request `result_config={"final_state": True}` for a NumPy density matrix.
It is the full physical qutrit state, including leakage and every transmon in
the selected model. For `m = len(model.subsystems)`, its shape is
`(3**m, 3**m)`, not `(2**n, 2**n)` for the program's qubits. A run containing
measurement may export this sampled posterior only with `shots=1`.

The accepted result request keys are `counts` and `final_state`. By default,
counts are requested when the program contains measurement, while a final
state is requested only for a program without measurement.

## Custom gate realizations

Each native operation resolves to its physical pulse recipe through a
`pulse_implementation_map=`, defaulting to
`fq.backends.default_superconducting_pulse_implementation_map()`. Copy that
default map and replace one gate's realization to change *how* a gate is
physically executed - the waveform shape, which control channels are
driven, which model resources are claimed - without editing calibration
data or subclassing `PulseBackend`:

```python
def custom_cz(operation, *, targets, model, calibration):
    ...
    return fq.backends.PulseDefinition(
        duration=...,
        controls=(...,),
        resource_claims=(...,),
    )

implementations = fq.backends.default_superconducting_pulse_implementation_map()
implementations.add(fq.ops.CZ, custom_cz)
backend = fq.backends.PulseBackend(
    model, calibration, pulse_implementation_map=implementations
)
```

Changing a calibrated *number* (a gate duration, a DRAG coefficient, a
per-edge detuning) is a calibration-document change. Changing the physical
*mechanism* a gate uses is an implementation-map change; the calibration
document itself is never a place to select executable behavior. See
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

`placement_mode="ASAP"` is the default. The private scheduler places pulse
blocks as early as their program dependencies and claimed resources permit.
`"ALAP"` instead places them as late as possible within the same ASAP
makespan. Both modes preserve dependency order and resource exclusivity; they
are simulation controls, not a public hardware schedule artifact.

Pulse execution is serial in v0.1 (`parallel_mode="auto"` normalizes to
serial). Seeded runs are reproducible under that policy.

## Noise support

Use `NoiseModel.add_channel` without `operation=` to apply always-on qutrit
noise, including idle time:

```python
noise.add_channel(fq.noise.ThermalRelaxation(t1=..., t2=...))
noise.add_channel(
    fq.noise.AmplitudeDamping(rate=(..., ...)), targets=("q0",)
)
```

Classical readout confusion is also supported.

Provide `operation=...` to scope the same channel descriptors to matching
pulse blocks. The two primitive damping
descriptors, {py:class}`~fatqat.noise.AmplitudeDamping` and
{py:class}`~fatqat.noise.PhaseDamping` (see {doc}`noise`), in either `p` or
`rate` mode - a `p`-mode instance is converted to a rate using the realized
gate's own duration, in nanoseconds (this model's declared time unit); a
`rate`-mode instance is used as-is. The resulting collapse operators are
active only over that gate's own placed time interval: idle time and other
concurrent, disjoint gates are unaffected, and a conditionally disabled gate
contributes neither its controls nor its attached noise. This composes with
always-on noise rather than replacing it - both scopes share the same
registration and collapse-operator implementation. Every other
channel type (e.g. `Depolarizing`) is still rejected, and coherent ZZ is
explicitly deferred in v0.1.
