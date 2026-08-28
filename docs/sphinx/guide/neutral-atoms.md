# Neutral-atom emulation

fatqat provides two pulse-resolved neutral-atom emulators. They share the
ordinary `Program -> run() -> Job -> Result` workflow, rectangular atom
arrangements, and geometry-derived Rydberg interactions. They differ in their
physical model and available controls:

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
qubit matrices and enforces a changing connectivity graph. The emulators
integrate time-dependent physical Hamiltonians and retain their model's
complete Hilbert space.

## Choose an emulator

| Capability | `Atom3LevelEmulator` | `Atom2LevelEmulator` |
|---|---|---|
| Physical basis | `\|0>, \|1>, \|r>` | `\|g>, \|r>` |
| Built-in gate-authored program | `RX`, `RY`, `RZ`, and `CZ` | none; custom rules supported |
| Direct-control program | selected-site Raman/Rydberg `PulseOperation` values | global drive/detuning `PulseOperation` values |
| Gate implementation map | optional replacement map with built-in default | optional replacement map with empty default |
| Lindblad implementation map | optional replacement map; empty default | optional replacement map; built-in two-level damping default |
| Backend inputs | physics model, arrangement, optional maps | physics model, arrangement, optional interaction cutoff and maps |
| Interaction pairs | signed `C6/R^6` over every declared-site pair | coordinate-derived, all pairs by default; optional distance cutoff |
| Ideal final state | full-qutrit density matrix | full two-level statevector |
| Measurement during a program | measurement, reset, and classical conditions are supported | terminal measurement suffix only |
| Default noise behavior | binary readout confusion; no Lindblad descriptors | rate-form amplitude/phase damping, thermal relaxation, depolarization, and binary readout confusion |
| Typical use | calibrated gates, coherent leakage, or selected-site drives | global Rydberg dynamics and shaped drive/detuning controls |

Choose the three-level emulator when calibration, coherent Rydberg leakage,
or selected-site controls are part of the experiment. Choose the two-level
emulator for global drive/detuning dynamics. If only gate connectivity and
ideal qubit behavior matter, use
{py:class}`~fatqat.simulator.AtomArraySimulator` instead.

## Set up the model and sites

Load the emulator's physics model, create an
{py:class}`~fatqat.emulator.AtomArrangement`, and declare one dimension-two
program resource per site. Program resources bind to the arrangement's
row-major site order.

The three-level emulator supplies calibrated gates and selected-site
Raman/Rydberg channels. The two-level emulator supplies global drive and
detuning channels and has no built-in gates.

```python
arrangement = fq.emulator.AtomArrangement.rectangular(
    rows=2,
    cols=3,
    spacing=6.0,  # um for the current atom models
)
```

The arrangement describes fixed geometry. Arbitrary coordinates, transport,
loading, loss, and refill are not part of these pulse emulators.

`arrangement.num_sites` is the exact number of declared coordinates and equals
`len(arrangement)`; a pulse program must have exactly that many resources.
This differs from `AtomArraySimulator(num_sites=6)`, where `num_sites` is a
maximum capacity and smaller programs are valid. `AtomArraySimulator()` is
unbounded.

## Run a program

Both classes return an eager {py:class}`~fatqat.Job`:

```python
result = backend.run(
    program,
    shots=100,
    simulation_config={"seed": 7, "schedule_mode": "ASAP"},
    result_config={"counts": True, "final_state": False},
).result()
```

Measurement requests counts by default, while an unmeasured program requests
the backend's natural final state. A sampled posterior state can be returned
only for `shots == 1`. Use `result.available_data` when code must handle both
statevectors and density matrices.

Both classes also provide `propagator()` for coherent, measurement-free
programs. Its matrix covers the full physical Hilbert space: `(3**N, 3**N)`
for the three-level emulator and `(2**N, 2**N)` for the two-level emulator.

Continue with [Three-level atom emulation](atom-3level.md) or
[Two-level atom emulation](atom-2level.md). Exact constructors,
configuration keys, result types, and capability objects are collected in
the {doc}`neutral-atom API reference <../api/atom-emulators>`.
