# Neutral-atom emulation

fatqat provides two pulse-resolved neutral-atom emulators. They share the
ordinary `Program -> run() -> Job -> Result` workflow, rectangular atom
arrangements, and geometry-derived Rydberg interactions. They differ in the
physical model and control surface:

- {py:class}`~fatqat.emulator.Atom3LevelEmulator` integrates the physical
  `|0>, |1>, |r>` model. It accepts calibrated gates and selected-site direct
  Raman/Rydberg controls.
- {py:class}`~fatqat.emulator.Atom2LevelEmulator` integrates the physical
  `|g>, |r>` model. It accepts global direct drive/detuning controls.

Gate-authored and direct-control programming are independent paths. The
three-level system has built-in gate recipes. The two-level system has an empty
built-in gate map but accepts user-supplied gate rules through the same path.

Neither class is a mode of {py:class}`~fatqat.simulator.AtomArraySimulator`.
That simulator is a fast, constrained gate-level target: it applies finite
qubit matrices and enforces a dynamic connectivity graph. The emulators integrate
time-dependent physical Hamiltonians and retain their model's complete
Hilbert space.

The gate-level AtomArray prototype keeps a fixed private carrier slot and
Hilbert-space dimension for every declared resource during a run.
``Pair`` and ``Unpair`` edit a separate connectivity graph that decides
two-qubit-gate legality, leaving the quantum state and tensor-axis order
unchanged. Loss and ``Put`` update separate occupancy metadata on those fixed
slots. This is an AtomArray-specific invariant: connectivity and occupancy are
tracked outside the quantum state, with no coordinates or transport API.

## Choose an emulator

| Capability | `Atom3LevelEmulator` | `Atom2LevelEmulator` |
|---|---|---|
| Physical basis | `\|0>, \|1>, \|r>` | `\|g>, \|r>` |
| Built-in gate-authored program | `RX`, `RY`, `RZ`, and `CZ` | none; custom rules supported |
| Direct-control program | selected-site Raman/Rydberg `PulseOperation` values | global drive/detuning `PulseOperation` values |
| Gate implementation map | optional replacement map with built-in default | optional replacement map with empty default |
| Lindblad implementation map | optional replacement map; empty default | optional replacement map; built-in two-level damping default |
| Backend inputs | physics model, arrangement, optional maps | physics model, arrangement, interaction policy, optional maps |
| Interaction graph | signed `C6/R^6` over every occupied pair | nearest-neighbor by default; explicit full-pair option |
| Ideal final state | full-qutrit density matrix | full two-level statevector |
| Measurement during a program | measurement, reset, and classical conditions use the shared pulse engine | terminal measurement suffix only |
| Default noise behavior | binary readout confusion; no Lindblad descriptors | target-local background rate-form amplitude or phase damping |
| Typical use | calibrated gates, coherent leakage, or selected-site drives | global Rydberg dynamics and shaped drive/detuning controls |

Choose the three-level emulator when calibration, coherent Rydberg leakage,
or selected-site controls are part of the experiment. Choose the two-level
emulator for global drive/detuning dynamics. If only gate connectivity and
ideal qubit behavior matter, use
{py:class}`~fatqat.simulator.AtomArraySimulator` instead.

## Shared construction model

Both emulators keep three kinds of information separate:

1. A versioned physics-model document defines species, levels, units, and the
   signed `C6` coefficient.
2. An {py:class}`~fatqat.AtomArrangement` defines occupied rectangular site
   coordinates in row-major order.
3. The `Program` declares one dimension-two quantum resource per arrangement
   site. Resources bind to sites in declaration order.

The three-level emulator compiles a nominal gate map internally or accepts a
supplied replacement. Calibration is an input to that map builder, not
emulator state. Its direct Raman/Rydberg controls bypass gate realization. The
two-level emulator has no calibration and uses an empty built-in gate map; a
supplied map can add ordinary gates. Its model supplies global drive and
detuning addresses, and direct programs supply their sampled waveforms.

```python
arrangement = fq.AtomArrangement.rectangular(
    rows=2,
    cols=3,
    spacing=6.0,  # um for the current atom models
)
```

The arrangement is immutable and initially fully occupied. Arbitrary
coordinates, transport, loading, and refill are outside the current emulator
contracts.

## Shared run and result workflow

Both classes return an eager {py:class}`~fatqat.Job`:

```python
result = backend.run(
    program,
    shots=100,
    simulation_config={"seed": 7, "schedule_mode": "ASAP"},
    result_config={"counts": True, "final_state": False},
).result()
```

The result request follows the same defaults as the other pulse emulator:
measurement requests counts by default, while an unmeasured program requests
the backend's natural final-state representation. A sampled posterior final
state can be returned only for `shots == 1`. The concrete state artifact is
backend- and execution-mode-specific, so use `result.available_data` or the
appropriate accessor rather than assuming every emulator returns a
statevector.

Both classes also provide `propagator()` for coherent, measurement-free
programs. Its matrix covers the full physical Hilbert space: `(3**N, 3**N)`
for the three-level emulator and `(2**N, 2**N)` for the two-level emulator.

Continue with [Three-level atom emulation](atom-3level.md) or
[Two-level atom emulation](atom-2level.md). Exact constructors,
configuration keys, result types, and capability objects are collected in
the {doc}`neutral-atom API reference <../api/atom-emulators>`.
