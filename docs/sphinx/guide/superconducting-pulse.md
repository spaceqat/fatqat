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
and phase corrections.

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

The backend accepts calibrated `RX`, `RY`, virtual `RZ`, `iSwap`, and oriented
`CZ`. `iSwap` and `CZ` require a declared model coupling. For `CZ`, program
target order must match the calibration's detuning orientation.

```python
program = fq.Program(2, 1)
program.add(fq.ops.RX(0.4), 0)
program.add(fq.ops.iSwap, (0, 1))
program.add_measurement(0, 0)

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

Use `NoiseModel.add_continuous_noise` with
`fq.noise.ThermalRelaxation(T1_ns, T2_ns)` to apply qutrit T1/T2 evolution
continuously, including idle time. Classical readout confusion is also
supported. Gate-keyed Kraus channels are not supported by this backend, and
coherent ZZ is explicitly deferred in v0.1.
