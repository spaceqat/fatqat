# Noise

FATQAT models noise separately from the program. A
{py:class}`~fatqat.NoiseModel` says which physical and classical errors to use
and where they apply. The backend decides which ones it supports.

This makes comparisons straightforward: keep one program and run it ideally,
with a reference device profile, or with several noise models.

```python
import numpy as np
import fatqat as fq
import fatqat.operations as ops

program = fq.Program(2, 2)
program.add(ops.H, 0)
program.add(ops.CX, (0, 1))
program.measure_all()

noise = fq.NoiseModel()
noise.add(fq.noise.Depolarizing(p=0.05), operation=ops.CX)
noise.add(
    fq.noise.ReadoutConfusion(
        np.array([[0.98, 0.04], [0.02, 0.96]])
    )
)

backend = fq.simulator.Simulator(method="density_matrix", noise=noise)
counts = backend.run(
    program,
    shots=1000,
    simulation_config={"seed": 7},
).result().get_counts()
```

## Noise on operations

Pass `operation` to associate noise with every matching operation. A class and
an instance select the same operation type; parameters on an instance do not
narrow the match.

```python
noise = fq.NoiseModel()

# Joint depolarization for every CX.
noise.add(fq.noise.Depolarizing(p=0.02), operation=ops.CX)

# Phase damping for every H on logical qubit 0.
q0 = program.quantum_registers[0][0]
noise.add(
    fq.noise.PhaseDamping(p=0.01),
    operation=ops.H,
    targets=q0,
)
```

Operation `targets` are exact and ordered. For a two-target operation,
`(q0, q1)` matches only that target order; it does not match the same operation
on `(q1, q0)`. A scalar selects a one-target operation. A tuple names the
complete operation, not just the operands that receive the noise.

Use `target_positions` to put a local source on selected operands of a
multi-target operation:

```python
noise = fq.NoiseModel()
noise.add(
    fq.noise.AmplitudeDamping(p=0.002),
    operation=ops.CZ,
    target_positions=0,
)
noise.add(
    fq.noise.AmplitudeDamping(p=0.003),
    operation=ops.CZ,
    target_positions=1,
)
```

Positions follow the operation's target order. Targets may be program
{py:class}`~fatqat.RegisterRef` values or device labels from a
{py:class}`~fatqat.ResourceLayout`. See {doc}`../api/noise/model` for the
complete selector, composition, and validation rules.

## Background noise

Omitting `operation` means background evolution, not “after every gate.”
Background noise must be local and target exactly one logical reference or
device label.

This example uses the reference transmon labels `"q0"` and `"q1"`:

```python
pulse_noise = fq.NoiseModel()
pulse_noise.add(
    fq.noise.ThermalRelaxation(t1=60_000.0, t2=80_000.0),
    targets="q0",
)
pulse_noise.add(
    fq.noise.PhaseDamping(t_phi=500_000.0),
    targets="q1",
)
```

Background Lindblad operators remain active throughout elapsed pulse time,
including idle intervals. Add one rule per site when the same source exists at
several sites.

Rates use the inverse of the backend model's `time_unit`; `t1`, `t2`, and
`t_phi` use that time unit directly. Check the model rather than inferring the
unit from a value's magnitude.

Calls to `add()` accumulate. Different noise types can act on the same
operation, and background noise can coexist with operation noise. FATQAT
rejects overlapping rules of the same type.

## Choose the backend form

Simulators apply probability-form channels after operations. Pulse emulators
apply rate- or time-based noise over elapsed time. FATQAT does not guess a gate
duration to convert between the two. Built-in coverage is listed in
{ref}`noise-simulator-support` and {ref}`noise-emulator-support` under the
combined {ref}`noise-backend-support` reference.

When you know the duration, convert explicitly:

```python
relaxation = fq.noise.ThermalRelaxation(t1=60e-6, t2=80e-6)
damping, dephasing = relaxation.as_channels(duration=2e-6)

matrix_noise = fq.NoiseModel()
matrix_noise.add(damping, operation=ops.H)
matrix_noise.add(dephasing, operation=ops.H)
```

{py:meth}`~fatqat.noise.ThermalRelaxation.as_channels` returns two channels
that reproduce qubit `T1`/`T2` evolution over that duration. The damping
types also provide explicit probability/rate conversions; their API pages
describe the multilevel limits.

## Readout errors

{py:class}`~fatqat.noise.ReadoutConfusion` models classical reporting errors,
not quantum evolution. Each column of its matrix is a probability distribution
under the convention

$$
C[\mathrm{reported},\mathrm{true}]
= P(\mathrm{reported}\mid\mathrm{true}).
$$

```python
confusion = fq.noise.ReadoutConfusion(
    np.array([[0.98, 0.05], [0.02, 0.95]])
)

readout_noise = fq.NoiseModel()
readout_noise.add(confusion)                 # every measured subsystem
# Or pass one logical reference or device label:
# readout_noise.add(confusion, targets=q0)
```

Physical collapse follows the true outcome. Only the digit written to the
classical register is resampled, so feedforward and counts see the reported
digit while later quantum evolution starts from the true post-measurement
state. The matrix size must match the backend's reported digit dimension; a
three-level pulse model can still have binary reporting.

## Carrier loss and occupancy

{py:class}`~fatqat.noise.Loss` is not amplitude damping. On an
occupancy-aware simulator, a hit removes a present carrier. The site stays
empty until `Put`; gates that need the carrier are skipped, and measurement
reports erasure digit `2`. The built-in backend and method restrictions are
listed in {ref}`noise-simulator-support`.

```python
loss_noise = fq.NoiseModel()
loss_noise.add(
    fq.noise.Loss(p=0.01),
    operation=ops.Put,  # Model unsuccessful loading after Put.
)
```

Loss is sampled independently for each selected carrier after a matching
operation. An empty-site erasure bypasses readout confusion because no
physical digit was measured.

## Validate backend support

Use
{py:meth}`~fatqat.simulator.Simulator.validate_noise_model` for a simulator or
{py:meth}`~fatqat.emulator.TransmonEmulator.validate_noise_model` for a pulse
emulator to validate a model before running a program. Each method returns
`None` for an accepted model or raises
{py:exc}`~fatqat.errors.BackendValidationError`, listing every model-level
problem it finds. Checks that need a program or layout happen at run time.

If you are implementing a backend or custom noise type, see
{doc}`../api/noise/custom-implementations` for the extension API. Ordinary
noise-model users do not need an implementation map.
