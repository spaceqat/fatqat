# Two-level atom emulation

{py:class}`~fatqat.emulator.Atom2LevelEmulator` integrates directly authored
global controls in a two-level Rydberg model. Its drive and detuning channels
act on every site. See {doc}`../api/pulse-control/index` for the shared pulse
authoring API.

This is an analog Hamiltonian-control workflow: “analog” describes how the
program is authored, not a different two-level physics system.

The direct-control API accepts sampled waveforms. Function and symbolic
waveforms are not supported.

## Run global controls

The packaged model is a reference two-level reduction of the Rb87 53S
`|1> <-> |r>` profile, not a device calibration.

```{doctest}
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> model_document = fq.emulator.load_model_document("atom2level.reference")
>>> model = fq.emulator.Atom2LevelModel.from_document(model_document)
>>> arrangement = fq.emulator.AtomArrangement.chain(num_sites=2, spacing=6.0)
>>> backend = fq.emulator.Atom2LevelEmulator(model, arrangement=arrangement)
>>> drive_waveform = fq.emulator.SampledWaveform(
...     (0.0, 0.4, 1.0),
...     (0.0, 0.5j, 0.25 - 0.1j),
... )
>>> detuning_waveform = fq.emulator.SampledWaveform(
...     (0.0, 0.5, 1.0),
...     (-0.2, 0.1, 0.0),
... )
>>> controls = (
...     fq.emulator.PulseControl(model.control.drive(), drive_waveform),
...     fq.emulator.PulseControl(model.control.detuning(), detuning_waveform),
... )
>>> program = fq.Program(arrangement.num_sites)
>>> operation = ops.PulseOperation(1.0, controls)
>>> program.add(operation)
>>> result = backend.run(program).result()
>>> result.get_statevector().shape
(4,)
```

The complex drive value is the physical complex envelope: its magnitude and
argument encode amplitude and phase together. Detuning samples must be real.
Both controls use the model's `rad/us` unit and act on every arrangement site.

For complex drive envelope $u(t)$ and real detuning $\Delta(t)$, the coherent
Hamiltonian convention is

```{math}
H(t) = \frac{1}{2}\left[
u(t)\sum_i \sigma_i^+ + u(t)^*\sum_i \sigma_i^-
\right]
- \Delta(t)\sum_i n_i
+ \sum_{(i,j)\in E}\frac{C_6}{R_{ij}^6}n_i n_j.
```

Here $E$ is the static interaction-pair set derived from coordinates. The
default keeps every unordered pair; a finite `interaction_cutoff` truncates
the set by Euclidean distance.
Thus $u(t)=A(t)e^{i\phi(t)}$: a positive phase multiplies the raising-operator
term. For example, a `0.5j` sample has phase $+\pi/2$ under this convention.

Drive and detuning may run concurrently in one operation. See
{doc}`../api/pulse-control/pulse-operation` for the rules shared by all pulse
emulators.

## Interpolation and limits

See {doc}`../api/pulse-control/sampled-waveform` for interpolation behavior.
This model applies detuning and drive limits to the interpolated curve,
including extrema between samples.

## Geometry, interaction, and results

Program resources bind to the arrangement's row-major sites. The arrangement
defines fixed geometry, and the model contributes the signed interaction shown
above. Every run starts with all sites in ``|g>``.

The default `interaction_cutoff=None` selects every unordered pair. Set
`interaction_cutoff=arrangement.spacing` for only rectangular horizontal and
vertical nearest pairs, or `interaction_cutoff=0.0` for no pair interaction.
Any finite value is a hard, inclusive cutoff in the model distance unit,
currently micrometres. The cutoff removes Hamiltonian terms and must not be
interpreted as a blockade radius. Direct controls act on every declared site.

An unmeasured ideal run returns a full `(2**N,)` statevector. Coherent
measurement-free programs also support `backend.propagator(program)`. A
terminal measurement suffix returns counts by default. The built-in gate map
is empty, so ordinary gates reject by default; a user-supplied
`gate_implementation_map` can add them. Reset, conditions, per-site direct
controls, mid-circuit measurement, and a pulse after measurement are not
supported.

The built-in Atom2 forms are listed under {ref}`noise-emulator-support`.
Supported background declarations act on one site at a time. Rates use inverse
microseconds and relaxation times use microseconds; enumerate sites explicitly
to apply the same noise at several sites. Finite probabilities are not
converted to rates. Binary readout confusion is a separate classical report
channel and changes only the reported digit after physical collapse.

With Lindblad noise and no measurement, the backend returns an exact density
matrix; with terminal measurement it runs seeded trajectories. Readout
confusion by itself does not change the statevector or density-matrix result
type. See {doc}`../api/atom-emulators` for supported results and noise.

A typical model can combine background relaxation with readout confusion:

```{doctest}
>>> noise = fq.NoiseModel()
>>> noise.add(fq.noise.ThermalRelaxation(t1=1000.0, t2=800.0), targets=0)
>>> noise.add(fq.noise.ReadoutConfusion([[0.99, 0.02], [0.01, 0.98]]))
>>> noisy_backend = fq.emulator.Atom2LevelEmulator(
...     model, arrangement=arrangement, noise=noise
... )
```
