# Three-level atom emulation

{py:class}`fq.emulator.Atom3LevelEmulator <fatqat.emulator.Atom3LevelEmulator>` translates
a small native gate set into calibrated controls and integrates the physical
Rb87 three-level model `|0>, |1>, |r>`. Use it when gate calibration, coherent
Rydberg leakage, atom spacing, or parallel-gate crosstalk is part of the
question. For global direct control in a two-level model, use
[the two-level emulator](atom-2level.md); for fast ideal qubit gates on a
constrained grid, use {py:class}`~fatqat.simulator.AtomGridSimulator`.

See [Neutral-atom emulation](neutral-atoms.md) for a side-by-side comparison
and {doc}`../api/atom-emulators` for exact signatures.

## At a glance

- The program may mix calibrated `RX`, `RY`, `RZ`, and `CZ` gates with direct
  per-site Raman/Rydberg pulse values.
- The backend requires a physics model and rectangular arrangement; its
  nominal gate map is compiled internally.
- Program qubits are dimension two, but the physical state is dimension three
  per atom and is returned as a density matrix.
- Every occupied pair contributes a signed `C6/R^6` interaction whenever
  Rydberg population is present.
- Binary readout confusion is built in. The Lindblad default is empty; a
  supplied map enables registered qutrit channel descriptors.

## Build the backend

The physics model and calibration are separate, strict JSON-compatible
documents. The model owns species, levels, units, and `C6`; the portable
calibration owns only gate-control recipe data. Geometry belongs to neither
document.

This complete two-site example defines both documents inline so it can be
copied and executed without package-private fixtures:

```{doctest}
>>> import fatqat as fq
>>> model_document = {
...     "format": {"id": "atom.rb87_rydberg_3level", "version": 1},
...     "model": {"id": "rb87-53s-reference", "revision": "2026-08-05"},
...     "system": {
...         "species": "Rb87",
...         "basis": {
...             "0": "5S1/2,F=1,mF=0",
...             "1": "5S1/2,F=2,mF=0",
...             "r": "53S1/2,mJ=1/2",
...         },
...         "transitions": {"rydberg": {"from": "1", "to": "r"}},
...     },
...     "units": {
...         "mass": "u", "distance": "um", "time": "us",
...         "angular_frequency": "rad/us", "c6": "rad/us*um^6",
...     },
...     "parameters": {"mass": 86.9091805, "c6": 180955.73684677208},
... }
>>> calibration_document = {
...     "format": {
...         "id": "atom.rb87_rydberg_3level_fixed_pulse", "version": 1,
...     },
...     "calibration": {"id": "rb87_53s_lukin_2023_v1", "revision": "2026-08-05"},
...     "units": {
...         "angular_frequency": "rad/us", "angle": "rad",
...         "cycles": "cycle", "dimensionless": "1",
...     },
...     "recipes": {
...         "rx_ry": {"omega_01": 6.283185307179586},
...         "cz": {
...             "omega_1r": 28.902652413026097,
...             "phase_amplitude": 0.704973391304749,
...             "phase_rate_ratio": 1.0431,
...             "phase_offset": -0.7318,
...             "linear_phase_rate_ratio": 0.0,
...             "duration_area": 1.215,
...             "local_z_correction": 2.099085629,
...         },
...     },
... }
>>> model = fq.emulator.Atom3LevelModel(model_document)
>>> arrangement = fq.AtomArrangement.rectangular(rows=1, cols=2, spacing=2.0)
>>> backend = fq.emulator.Atom3LevelEmulator(
...     model, arrangement=arrangement
... )
```

For explicit calibration, compile it before backend construction:

```{doctest}
>>> calibration = fq.emulator.Atom3LevelCalibration(calibration_document)
>>> gate_map = fq.emulator.default_atom_3level_gate_implementation_map(
...     model=model, calibration=calibration
... )
>>> explicit_backend = fq.emulator.Atom3LevelEmulator(
...     model, arrangement=arrangement, gate_implementation_map=gate_map
... )
```

The v1 builder requires a source-model argument as its future pulse-design
seam but deliberately does not read or retain C6 or geometry. Consequently a
map compiled from a coarse source model can run on a distinct target model and
arrangement; target C6 and geometry still govern physical evolution. Rebuild
the map with a richer source model only when a later recipe actually uses its
additional design facts.

## Write a gate-authored atom program

The program must declare exactly one dimension-two quantum resource per
arrangement site. Declaration order binds resources to the arrangement's
row-major coordinates.

| Program feature | Three-level behavior |
|---|---|
| `RX(theta)`, `RY(theta)` | calibrated Raman rotation in `span{\|0>, \|1>}` |
| `RZ(theta)` | zero-duration virtual frame update |
| `CZ(a, b)` | fixed calibrated phase-modulated Rydberg pulse on the ordered pair |
| measurement | samples physical levels, then writes a binary classical bit |
| `Reset` | prepares physical `\|0>` |
| classical condition | supported by the shared pulse execution path |
| `Barrier` | structural no-op |
| direct `PulseOperation` | per-site Raman or Rydberg control addresses from the model |
| `LoadAtoms`; ordinary gates without a registered rule | rejected |

Here is the minimal calibrated-CZ program and deterministic final-state
request:

```{doctest}
>>> program = fq.Program(2)
>>> program.add(fq.ops.CZ, (0, 1))
>>> result = backend.run(
...     program,
...     shots=1,
...     result_config={"counts": False, "final_state": True},
... ).result()
>>> result.get_density_matrix().shape
(9, 9)
```

The `(9, 9)` shape is the complete two-qutrit state, not a projected
four-dimensional computational-subspace state. In general, `N` arrangement sites produce a
`(3**N, 3**N)` density matrix. QuTiP objects never cross the public boundary.

## Direct Raman and Rydberg controls

Use model factories for selected-site control addresses. Complex samples
encode the two quadratures of the selected transition; the address, rather
than ordinary program targets, identifies the site:

```{doctest}
>>> direct = fq.ops.PulseOperation(
...     0.5,
...     (
...         fq.emulator.PulseControl(
...             model.raman_control(0),
...             fq.waveforms.SampledWaveform((0.0, 0.5), (0.2, 0.2j)),
...         ),
...         fq.emulator.PulseControl(
...             model.rydberg_control(1),
...             fq.waveforms.SampledWaveform((0.0, 0.5), (0.1, 0.1)),
...         ),
...     ),
... )
>>> direct_program = fq.Program(2)
>>> direct_program.add(direct)
>>> backend.run(direct_program).result().get_density_matrix().shape
(9, 9)
```

Direct and calibrated operations can coexist in one program. Raman and
Rydberg control addresses are structural by site ordinal, so the same authored
operation can be reused with another compatible arrangement containing that
site.

## Run configuration and results

`simulation_config` accepts only:

- `seed`: an integer seed for physical measurement and readout sampling;
- `schedule_mode`: `"ASAP"` by default or `"ALAP"`.

`result_config` accepts `counts` and `final_state`. Measurement makes counts
the default; without measurement, the full physical density matrix is the
default. Counts require a positive integer `shots`. A run that both samples a
physical measurement and returns its posterior density matrix requires
`shots == 1`, because several sampled posterior states do not define one final
state.

Physical levels are reported as `|0> -> 0`, `|1> -> 1`, and `|r> -> 1` before
classical readout confusion. The `|r>` mapping is only a reported bit: Rydberg
population remains coherent qutrit leakage and does not mean the atom was
physically lost. Classical conditions consume the reported classical value,
including any configured readout confusion.

## Coherent propagators

`backend.propagator(program)` returns the complete coherent `(3**N, 3**N)`
operator. It accepts gate programs without measurement, reset, or classical
conditions. The default `apply_final_frame=True` includes terminal virtual
frame corrections; set it to `False` to inspect the raw physical evolution
before the remaining frame ledger is composed.

Binary readout confusion is inert for a propagator because no measurement
boundary exists.

## Physics and geometry

The physical model uses `|0>` and `|1>` as the computational subspace and
drives `|1> <-> |r>` for the Rydberg gate. Its interaction drift is

```{math}
\sum_{i<j} \frac{C_6}{R_{ij}^6} n_i^r n_j^r.
```

The signed interaction includes every occupied pair, not only the targets of
a requested `CZ`. A parallel layer of disjoint CZ gates therefore still has
cross-pair interaction while Rydberg population is present. This is physical
crosstalk, not an unintended all-to-all effective gate.

The built-in phase-modulated CZ recipe is fixed. Changing spacing changes
`R_ij`, the interaction, and potentially the fidelity, but the backend does
not optimize, retune, reject, or warn about that pulse.

## Noise and current boundary

Binary `2 x 2` classical readout-confusion matrices are supported directly.
The default Lindblad implementation map is empty, so physical channel
descriptors reject unless the user supplies a map. A supplied map can enable
registered `3 x 3` local qutrit collapse operators from authored generator
declarations. `operation=...` limits a generator to matching blocks; omitting
it declares target-local background noise. Finite probabilities are rejected
rather than converted with a realized duration. Qutrit amplitude damping needs
two adjacent-transition rates. Use `backend.validate_noise(noise_model)` to
inspect the effective instance capability without executing a program.

Physical atom loss, occupancy changes, Rydberg `T1`, quasi-static `T2_star`,
intermediate-state scattering, adjacent Rydberg levels, and position
fluctuation are not modeled in the current three-level system. `T2_star` is a
measured aggregate reserved for a later stochastic model, not a synonym for
one modeled Doppler term.

The three-level emulator accepts a public `PulseImplementationMap`. Standard
maps come from `default_atom_3level_gate_implementation_map(model=...,
calibration=...)`; custom operand-aware rules receive plain site ordinals in
`device_operands` and return claim-free `PulseDefinition` values using public
`frame()`, `raman_control()`, and `rydberg_control()` structural addresses.
Direct control continues to bypass gate-map lookup. Package defaults are
nominal simulation baselines rather than hardware-fidelity guarantees; use a
complete separate calibration document for custom values instead of patching
the packaged JSON.
