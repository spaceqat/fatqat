# Two-level atom emulation

{py:class}`~fatqat.emulator.Atom2LevelEmulator` integrates directly authored
global controls in a two-level Rydberg model. The public path has four distinct
layers:

This is an analog Hamiltonian-control workflow, but “analog” describes how the
program is authored—not the identity of the two-level physics system.

1. {py:class}`~fatqat.emulator.SampledWaveform` stores immutable samples.
2. The model returns a structural global drive or detuning address.
3. {py:class}`~fatqat.emulator.PulseControl` binds one waveform to one address.
4. {py:class}`~fatqat.operations.PulseOperation` groups concurrent bindings in
   one duration; lowering and scheduling remain private backend work.

Function and symbolic waveforms remain future work.

## Complete direct-control example

The model document stays geometry-free. Load the packaged reference explicitly
and inspect its identity, units, parameters, and references before choosing it.
The packaged model is an effective two-level reduction of the Rb87 53S
`|1> <-> |r>` profile used by the three-level emulator, not a universal Rb
constant or a device calibration.

```{doctest}
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> model_document = fq.emulator.load_model_document("atom2level.reference")
>>> model_document["model"]
{'id': 'rb87-53s-two-level-reference', 'revision': '2026-08-22'}
>>> model_document["units"]["distance"], model_document["parameters"]["c6"]
('um', 180955.73684677208)
>>> model_document["references"]
['doi:10.1038/s41586-023-06481-y']
>>> model = fq.emulator.Atom2LevelModel.from_document(model_document)
>>> tuple(model.available_controls)
('drive', 'detuning')
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
>>> program.add(ops.PulseOperation(1.0, controls))
>>> result = backend.run(program).result()
>>> result.get_statevector().shape
(4,)
```

Evered et al. report a 2 µm separation with
$V_\mathrm{Ryd}/2\pi \approx 450$ MHz. The stored effective profile uses the
agreed positive-sign convention and derives
$C_6/\hbar = 2\pi \times 450 \times 2^6 =
180955.73684677208$ rad/µs·µm⁶. The experimental input is approximate; the
digits identify the reproducible derived snapshot rather than additional
measurement precision.

The complex drive value is the physical complex envelope: its magnitude and
argument encode amplitude and phase together. Detuning samples must be real.
Both controls use the model's `rad/us` unit and act on every arrangement site.
The operation has zero ordinary targets because its addresses already describe
what is driven.

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

Each control may use a nonzero `start_offset`. Its offset plus waveform
duration must fit within the enclosing operation duration. Several operations
may be added sequentially, and drive plus detuning may be concurrent in one
operation. Two bindings for the same address in one operation are rejected;
sum their samples explicitly first.

## Interpolation and limits

Sampled controls use `fatqat.spline.v1`: cubic interpolation with effective
degree `min(3, sample_count - 1)`. Execution delegates sampled-array
coefficients to QuTiP, which uses SciPy's matching spline construction. The
shared SciPy helper is retained for exact limit validation. Detuning limits
apply to exact real spline extrema. The drive-amplitude limit applies to exact
extrema of the complex magnitude, including stationary points between samples.

## Geometry, interaction, and results

The arrangement binds program resources to declared row-major sites. It
defines geometry, not dynamic atom occupancy. The model contributes the signed
static interaction shown in the Hamiltonian above.

The default `interaction_cutoff=None` selects every unordered pair. Set
`interaction_cutoff=arrangement.spacing` for only rectangular horizontal and
vertical nearest pairs, or `interaction_cutoff=0.0` for no pair interaction.
Any finite value is a hard, inclusive numerical cutoff in the model distance
unit, currently micrometres. A fixed eight-machine-epsilon allowance at the
boundary compensates only for floating-point coordinate construction; it is
not a configurable physical tolerance. The cutoff removes Hamiltonian terms
and must not be interpreted as a blockade radius. Direct blocks claim and
target every declared site.

An unmeasured ideal run returns a full `(2**N,)` statevector. Coherent
measurement-free programs also support `backend.propagator(program)`. A
terminal measurement suffix returns counts by default. The built-in gate map
is empty, so ordinary gates reject by default; a user-supplied
`gate_implementation_map` can add them through the shared pulse-rule path.
Reset, conditions, local direct targets, mid-circuit measurement, and a pulse
after measurement remain outside the current two-level contract.

The implicit Lindblad map supports target-local background rate-form amplitude
damping, phase damping, thermal relaxation, and depolarization. Rates use
inverse microseconds and `ThermalRelaxation` times use microseconds. Enumerate
sites explicitly when the same generator is present on several. Binary
readout confusion is a separate classical report channel: it changes only the
reported digit after physical collapse and is not represented by a Lindblad
operator. Finite probabilities are not converted to rates. `Loss` and
`IncoherentTransitions` remain unsupported. With Lindblad noise and no
measurement the backend returns an exact density matrix; with terminal
measurement it runs seeded trajectories. Readout confusion by itself does not
change the statevector or density-matrix execution representation. See
{doc}`../api/atom-emulators` for the exact result and noise contracts.

All five built-in Atom2 noise declarations can be inspected together:

```{doctest}
>>> import numpy as np
>>> noise = fq.NoiseModel()
>>> noise.add(fq.noise.AmplitudeDamping(rate=0.001), targets=0)
>>> noise.add(fq.noise.PhaseDamping(rate=0.002), targets=0)
>>> noise.add(fq.noise.ThermalRelaxation(t1=1000.0, t2=800.0), targets=0)
>>> noise.add(fq.noise.Depolarizing(rate=0.0005), targets=0)
>>> noise.add(fq.noise.ReadoutConfusion(np.array([[0.99, 0.02], [0.01, 0.98]])))
>>> report = backend.check_noise_support(noise)
>>> report.supported
True
```
