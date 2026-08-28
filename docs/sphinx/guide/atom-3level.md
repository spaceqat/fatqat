# Three-level atom emulation

{py:class}`~fatqat.emulator.Atom3LevelEmulator` runs calibrated gates and
selected-site controls in the physical basis `|0>, |1>, |r>`. Use it when
Rydberg leakage, atom spacing, or interactions during a pulse are important.
For global controls in a two-level model, use [the two-level
emulator](atom-2level.md).

See [Neutral-atom emulation](neutral-atoms.md) for a comparison and
{doc}`../api/atom-emulators` for the complete API reference.

## Create the emulator

The model supplies the atomic levels, units, and signed `C6` coefficient. The
arrangement supplies fixed site coordinates.

```{doctest}
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> model = fq.emulator.Atom3LevelModel.from_document(
...     fq.emulator.load_model_document("atom3level.reference")
... )
>>> arrangement = fq.emulator.AtomArrangement.chain(2, spacing=6.0)
>>> backend = fq.emulator.Atom3LevelEmulator(
...     model,
...     arrangement=arrangement,
... )
```

The program must declare one dimension-two quantum resource per arrangement
site. Declaration order binds resources to the arrangement's row-major site
order. Every run starts with each atom in physical `|0>`.

The default backend uses the packaged reference calibration. To use another
complete calibration document, build and pass a gate map:

```python
calibration = fq.emulator.Atom3LevelCalibration(calibration_document)
gate_map = fq.emulator.default_atom_3level_gate_implementation_map(
    model=model,
    calibration=calibration,
)
backend = fq.emulator.Atom3LevelEmulator(
    model,
    arrangement=arrangement,
    gate_implementation_map=gate_map,
)
```

The resulting rules use that model's channels and frames. Geometry and `C6`
still determine physical evolution, but they do not retune the fixed pulse
recipes.

## Run calibrated gates

The built-in map supports `RX`, `RY`, `RZ`, and `CZ`. `RZ` is a zero-duration
frame update; `CZ` drives the requested sites with the calibrated Rydberg
pulse.

```{doctest}
>>> program = fq.Program(2)
>>> program.add(ops.CZ, (0, 1))
>>> result = backend.run(program).result()
>>> result.get_density_matrix().shape
(9, 9)
```

The density matrix covers the complete two-qutrit state, not only the
four-dimensional computational subspace. In general, `N` sites produce a
`(3**N, 3**N)` matrix.

Measurement reports physical `|0>` as `0` and both `|1>` and `|r>` as `1`,
then applies any readout confusion. This does not remove Rydberg population
from the physical state. A measured run can return one sampled posterior
density matrix only when `shots=1`.

## Add direct controls

Use `model.control.raman(site)` for the `|0> <-> |1>` transition and
`model.control.rydberg(site)` for the `|1> <-> |r>` transition. Both channels
accept complex angular-rate samples in `rad/us`.

```{doctest}
>>> waveform = fq.emulator.SampledWaveform(
...     (0.0, 0.25, 0.5),
...     (0.0, 0.2, 0.0),
... )
>>> channel = model.control.raman(0)
>>> control = fq.emulator.PulseControl(channel, waveform, start_offset=0.0)
>>> operation = ops.PulseOperation(duration=0.5, controls=(control,))
>>> direct_program = fq.Program(2)
>>> direct_program.add(operation)
>>> backend.run(direct_program).result().get_density_matrix().shape
(9, 9)
```

Direct controls and calibrated gates can coexist. A channel can also be reused
with another three-level atom model when its site exists there.

## Physics and timing

The Rydberg interaction is

```{math}
\sum_{i<j} \frac{C_6}{R_{ij}^6} n_i^r n_j^r.
```

Every site pair contributes while Rydberg population is present, including
pairs that are not the targets of the same `CZ`. Changing the spacing changes
this interaction and may change gate fidelity; the packaged CZ pulse is not
automatically retuned.

Use `backend.propagator(program)` for a coherent `(3**N, 3**N)` operator.
Measurement, reset, conditions, and nonzero-duration Lindblad evolution are
rejected. Terminal virtual-frame corrections are included by default.

`run()` supports `"ASAP"` and `"ALAP"` scheduling and classical conditions.
A false condition skips the operation's controls while elapsed time, drift,
and background noise continue.

## Add noise

The default three-level Lindblad map is empty. Pass a
{py:class}`~fatqat.noise.LindbladImplementationMap` to enable compatible local
rate- or time-form declarations. Qutrit amplitude damping needs two adjacent
transition rates. Rates use inverse microseconds and relaxation times use
microseconds.

Binary {py:class}`~fatqat.noise.ReadoutConfusion` remains available without a
Lindblad map. Probability-form channels are not converted to rates. See
{ref}`noise-emulator-support` for supported forms and
{ref}`pulse-probability-noise` for why pulse emulators require rates.
