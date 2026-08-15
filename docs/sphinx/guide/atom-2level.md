# Two-level atom emulation

{py:class}`~fatqat.emulator.Atom2LevelEmulator` integrates directly authored
global controls in a two-level Rydberg model. The public path has four distinct
layers:

This is an analog Hamiltonian-control workflow, but “analog” describes how the
program is authored—not the identity of the two-level physics system.

1. {py:class}`~fatqat.waveforms.SampledWaveform` stores immutable samples.
2. The model returns a structural global drive or detuning address.
3. {py:class}`~fatqat.emulator.PulseControl` binds one waveform to one address.
4. {py:class}`~fatqat.operations.PulseOperation` groups concurrent bindings in
   one duration; lowering and scheduling remain private backend work.

Function and symbolic waveforms remain future work.

## Complete direct-control example

The model document stays geometry-free. The `rydberg_global` key below is the
persisted v1 model-schema spelling; it is not an authoring channel name.

```{doctest}
>>> import fatqat as fq
>>> model_document = {
...     "format": {"id": "atom.rb87_rydberg_2level", "version": 1},
...     "model": {"id": "rb87-70s-analog-reference", "revision": "2026-08-07"},
...     "system": {
...         "species": "Rb87",
...         "basis": {"g": "5S1/2,F=2,mF=2", "r": "70S1/2,mJ=1/2"},
...         "transitions": {"rydberg": {"from": "g", "to": "r"}},
...     },
...     "units": {
...         "distance": "um", "time": "us",
...         "angular_frequency": "rad/us", "c6": "rad/us*um^6",
...     },
...     "parameters": {
...         "c6": 1.0,
...         "channel_limits": {
...             "rydberg_global": {
...                 "max_amplitude": None,
...                 "min_detuning": None, "max_detuning": None,
...                 "min_duration": None, "max_duration": None,
...             },
...         },
...     },
... }
>>> model = fq.emulator.Atom2LevelModel(model_document)
>>> arrangement = fq.AtomArrangement.rectangular(rows=1, cols=2, spacing=2.0)
>>> backend = fq.emulator.Atom2LevelEmulator(model, arrangement=arrangement)
>>> drive_waveform = fq.waveforms.SampledWaveform(
...     (0.0, 0.4, 1.0),
...     (0.0, 0.5j, 0.25 - 0.1j),
... )
>>> detuning_waveform = fq.waveforms.SampledWaveform(
...     (0.0, 0.5, 1.0),
...     (-0.2, 0.1, 0.0),
... )
>>> controls = (
...     fq.emulator.PulseControl(model.drive_control(), drive_waveform),
...     fq.emulator.PulseControl(model.detuning_control(), detuning_waveform),
... )
>>> program = fq.Program(2)
>>> program.add(fq.ops.PulseOperation(1.0, controls))
>>> result = backend.run(program).result()
>>> result.get_statevector().shape
(4,)
```

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

Here $E$ is the static edge set selected by the interaction policy: rectangular
nearest neighbors by default, or every unordered pair under `full_pair`.
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

The arrangement binds program resources to occupied row-major sites. The
model contributes the signed static interaction shown in the Hamiltonian
above.

The default interaction policy selects rectangular nearest-neighbor edges;
{py:meth}`~fatqat.emulator.GridInteractionPolicy.full_pair` selects every
pair. Direct blocks claim and target every allocated site.

An unmeasured ideal run returns a full `(2**N,)` statevector. Coherent
measurement-free programs also support `backend.propagator(program)`. A
terminal measurement suffix returns counts by default. The built-in gate map
is empty, so ordinary gates reject by default; a user-supplied
`gate_implementation_map` can add them through the shared pulse-rule path.
Reset, conditions, local direct targets, mid-circuit measurement, and a pulse
after measurement remain outside the current two-level contract.

The implicit Lindblad map supports always-on rate-form amplitude and phase
damping. A supplied replacement map can enable registered operation-scoped
probability/rate and always-on rate descriptors under the family's selector
and two-level operator-shape rules. Amplitude damping requires exactly one
adjacent-transition value in either mode. With noise and no measurement the backend
returns an exact density matrix; with terminal measurement it runs seeded
trajectories. See {doc}`../api/atom-emulators` for the exact result and noise
contracts.
