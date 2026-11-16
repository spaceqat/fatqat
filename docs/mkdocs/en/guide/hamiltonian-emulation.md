# Follow a Program into physical dynamics

A FatQat emulator follows the physical state during timed controls. It accepts
a [`Program`][fatqat.Program], builds a physical schedule, and integrates a
time-dependent Hamiltonian (and, when requested, Lindblad evolution). Use it
to study pulse duration, coupling, leakage, or noise acting continuously over
elapsed time. Circuit simulators can include noise channels without resolving
this time evolution.

Two authoring paths meet at that same schedule:

```text
Program gate ----> calibration ----\
                                  +----> physical schedule
PulseOperation --> controls ------/               |
                                                   v
                                    Hamiltonian/Lindblad evolution
```

The model supplies the physical levels, units, drift, channels, and coupling.
A gate implementation map turns an ordinary gate into calibrated controls.
A direct pulse supplies those controls explicitly. The following pages apply
this shared workflow to transmons and atoms.

## Run a calibrated gate

Use a packaged physical model as a reproducible baseline, then construct its
emulator. Here a one-qubit Program is realized on a model containing two
three-level transmons:

```pycon
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> model = fq.emulator.TransmonModel.from_document(
...     fq.emulator.load_model_document("transmon.reference")
... )
>>> backend = fq.emulator.TransmonEmulator(model)
>>> calibrated_program = fq.Program(1)
>>> calibrated_program.add(ops.RX(np.pi / 2), 0)
>>> calibrated_result = backend.run(calibrated_program).result()
>>> calibrated_result.get_statevector().shape
(9, 1)
```

`RX` remains an ordinary Program operation. At execution time, the emulator's
gate map obtains a pulse recipe from its calibration and binds that recipe to
the model's drive channel. The physical model has two qutrits, so the returned
state, a QuTiP ket `Qobj`, covers all $3^2$ basis states, including the second
transmon. The Program
still declares one two-level qubit. Each transmon's third physical level lets
the result show leakage out of its qubit states, `|0>` and `|1>`.

Packaged models and calibrations are reference snapshots. Supply a validated
model document and implementation map when the physical system or calibration
changes; do not interpret the packaged data as a live-device calibration.

## Add a pulse directly

A direct pulse has three pieces: waveform samples, the physical channel they
drive, and the duration of the control block. The code below builds them in
that order as a [`SampledWaveform`][fatqat.emulator.SampledWaveform], a
[`PulseControl`][fatqat.emulator.PulseControl], and finally a
[`PulseOperation`][fatqat.operations.PulseOperation]:

```pycon
>>> duration = 20.0
>>> waveform = fq.emulator.SampledWaveform(
...     (0.0, 10.0, duration),
...     (0.0, 0.02, 0.0),
... )
>>> control = fq.emulator.PulseControl(
...     model.control.drive("q0"),
...     waveform,
... )
>>> pulse = ops.PulseOperation(duration=duration, controls=(control,))
>>> direct_program = fq.Program(1)
>>> direct_program.add(pulse)
>>> direct_result = backend.run(direct_program).result()
>>> direct_result.get_statevector().shape
(9, 1)
```

Notice that `direct_program.add(pulse)` has no logical target. The channel
already names physical transmon `q0`, and a
[`ResourceLayout`][fatqat.ResourceLayout] does not remap that address. The model and
channel define the time and value units; the API reference records the exact
domains for each emulator.

## Read a pulse as continuous evolution

The three waveform samples above specify a drive over the full 20-unit
interval. The emulator interpolates between them, combines the drive with
drift and coupling terms, and integrates the physical state. The state covers
every level in the physical model throughout the pulse, including levels that
the Program did not declare.

The next chapter turns this mechanism into a concrete transmon experiment: it
compares a calibrated rotation with a direct drive and makes the resulting
leakage visible.

## Understand scheduling

All controls inside one `PulseOperation` share its interval; `start_offset`
can move an individual waveform within that interval. Between operations, the
schedule preserves source order for blocks that use the same physical resource
and may overlap independent blocks. Choose `"ASAP"` or
`"ALAP"` through `execution_config` when that placement matters.

Drift and background continuous noise evolve throughout elapsed time. On an
emulator that supports classical conditions, a false condition can skip a
control block without deleting its duration, so the model still evolves during
the interval. This lets you study the effect of elapsed time even when no
control is active.

The constructor's `method` selects the mathematical representation and Result
accessor: `statevector` (the default), `density_matrix`, or `unitary`. It does
not select an internal differential-equation solver.

Continue with [Transmon emulation](transmon-emulation.md) or
[Neutral-atom emulation](neutral-atom-emulation.md). For pulse object,
schedule, simulation-method, and model contracts, use the
[pulse-control API](../api/pulse-control/index.md) and
[emulator API](../api/emulators/index.md).
