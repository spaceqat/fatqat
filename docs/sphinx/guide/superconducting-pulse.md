# Superconducting transmon emulation

`fq.emulator.TransmonEmulator` runs gates and sampled controls on a fixed
three-level transmon model. Use it when pulse timing, leakage, coupling, or a
calibrated gate realization matters. For ideal matrix simulation, use
{py:class}`~fatqat.simulator.Simulator` instead.

See {doc}`../api/pulse-emulator` for the complete emulator reference and
{doc}`Pulse control <../api/pulse-control/index>` for direct-control and custom
gate-realization APIs.

## Create the emulator

Load a model document and construct the backend. The packaged snapshot is a
simulation reference, not a hardware calibration.

```{doctest}
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> model = fq.emulator.TransmonModel.from_document(
...     fq.emulator.load_model_document("transmon.reference")
... )
>>> model.subsystem_ids
('q0', 'q1')
>>> backend = fq.emulator.TransmonEmulator(model)
```

Program qubits bind to `model.subsystem_ids` in declaration order. Pass a
{py:class}`~fatqat.ResourceLayout` to `run()` when a program should use a
different ordering. Every transmon in the model remains part of the physical
state, including transmons not addressed by the program.

By default, the emulator uses the packaged gate calibration. To use a custom
calibration document, build a gate map explicitly:

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

The calibration document supplies pulse recipe values; the model supplies the
transmon parameters, channels, frames, and coupling graph. Rebuild the map when
changing either input should change the resulting pulses.

## Run calibrated gates

The built-in map supports `RX`, `RY`, `RZ`, `iSwap`, and `CZ`. `iSwap` and `CZ`
require a coupling declared by the model. `RZ` is a zero-duration frame update.

```{doctest}
>>> program = fq.Program(2, 1)
>>> program.add(ops.RX(0.4), 0)
>>> program.add(ops.iSwap, (0, 1))
>>> program.measure(0, 0)
>>> result = backend.run(
...     program,
...     shots=20,
...     simulation_config={"seed": 7},
... ).result()
>>> sum(result.get_counts().values())
20
```

Every run starts with all model transmons in physical `|0>`. Without
measurement, the default result is the full qutrit density matrix. For `m`
model transmons its shape is `(3**m, 3**m)`, so it includes leakage and any
unaddressed transmons. A measured run can return one sampled posterior density
matrix when `shots=1` and `final_state` is requested.

## Add direct controls

Direct controls use channels returned by `model.control`. Drive samples may be
complex; detuning and exchange samples must be real. Times are in nanoseconds
and values are angular rates in `rad/ns`.

Unlike an ordinary gate, a `PulseOperation` takes no logical targets when it is
added to a program. Its channels already identify the physical resources:

```{doctest}
>>> duration = 20.0
>>> waveform = fq.emulator.SampledWaveform(
...     (0.0, 10.0, duration),
...     (0.0, 0.02, 0.0),
... )
>>> channel = model.control.drive("q0")
>>> control = fq.emulator.PulseControl(channel, waveform, start_offset=0.0)
>>> operation = ops.PulseOperation(duration=duration, controls=(control,))
>>> direct_program = fq.Program(1)
>>> direct_program.add(operation)
>>> backend.run(direct_program).result().get_density_matrix().shape
(9, 9)
```

Direct blocks and calibrated gates may appear in the same program. A
`ResourceLayout` does not remap direct-control channels.

## Propagators and timing

Use `backend.propagator(program)` for a coherent full-model operator.
Measurement, reset, conditions, and nonzero-duration Lindblad evolution are
not compatible with a single propagator.

The default `apply_final_frame=True` includes remaining virtual-frame updates.
Set it to `False` to omit only that terminal frame transformation; frame
updates that rotate later controls still apply. Compare the result with ideal
qubit gates up to global phase because the emulator uses a rotating-frame
convention.

`schedule_mode="ASAP"` places operations as early as dependencies and resource
conflicts allow. `"ALAP"` places them as late as possible without lengthening
the program. Conditions are supported by `run()`: when a condition is false,
the gate controls and gate-scoped noise are skipped, but elapsed time, drift,
and background noise remain.

## Add noise

The default transmon Lindblad map supports rate-form amplitude damping, phase
damping, and thermal relaxation. Rates use inverse nanoseconds; relaxation
times use nanoseconds. Background noise acts throughout elapsed time:

```python
noise = fq.NoiseModel()
noise.add(
    fq.noise.ThermalRelaxation(t1=60_000.0, t2=80_000.0),
    targets="q0",
)
backend = fq.emulator.TransmonEmulator(model, noise=noise)
```

Probability-form channels are not converted to continuous rates. Other
continuous generators, including `Depolarizing(rate=...)`, require a supplied
{py:class}`~fatqat.noise.LindbladImplementationMap` that registers them. See
{ref}`noise-emulator-support` for the support table and
{ref}`pulse-probability-noise` for why pulse emulators require rates.

To change how an ordinary gate is realized, use
{doc}`../api/pulse-control/gate-realization` rather than putting executable
behavior in a calibration document.
