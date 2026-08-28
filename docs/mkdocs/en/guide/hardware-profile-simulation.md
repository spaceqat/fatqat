# Test a Program against a hardware profile

A hardware-profile simulator checks whether a Program can run on a selected
device shape as written. Like the general
[`Simulator`][fatqat.simulator.Simulator], it evolves discrete gates at circuit
level; it also enforces a native operation set, placement, connectivity,
capacity, and, for atom arrays, occupancy.

This makes profiles useful before a physical Hamiltonian model is needed. It
also sets an important boundary: a profile validates your choices; it does not
make those choices for you.

## Start from logical behavior

First establish what the Program means with the general-purpose simulator. A
Bell Program uses convenient logical gates and has no device placement yet:

```pycon
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> bell = fq.Program(2, 2)
>>> bell.add(ops.H, 0)
>>> bell.add(ops.CX, (0, 1))
>>> bell.measure_all()
>>> counts = fq.simulator.Simulator(runtime="numpy").run(
...     bell,
...     shots=16,
...     simulation_config={"seed": 7},
... ).result().get_counts()
>>> sum(counts.values())
16
>>> set(counts) <= {"00", "11"}
True
```

Now ask a Google-style superconducting profile about the same Program. The
profile can report whether it supports `H`, so you do not need to copy its
gate table into application code:

```pycon
>>> profile = fq.simulator.SCQubitGoogleSimulator(
...     grid_size=(2, 3),
...     runtime="numpy",
... )
>>> profile.implementation_map.supports(ops.H)
False
```

Submitting `bell` would therefore raise
[`UnsupportedOperationError`][fatqat.errors.UnsupportedOperationError]. FatQat does not silently
decompose `H` or `CX` into this profile's native operations.

## Make placement explicit

Native operations are only half the question. On this 2 x 3 profile, integer
device labels are arranged row by row:

```text
0 --- 1 --- 2
|     |     |
3 --- 4 --- 5
```

The next Program is native, but placing its two qubits at `0` and `4` asks for
a diagonal `CZ` that the grid does not provide:

```pycon
>>> qubits = fq.QuantumRegister(2, name="q")
>>> native = fq.Program([qubits])
>>> native.add(ops.RX(np.pi), qubits[0])
>>> native.add(ops.RX(np.pi), qubits[1])
>>> native.add(ops.CZ, (qubits[0], qubits[1]))
>>> bad_layout = fq.ResourceLayout({qubits[0]: 0, qubits[1]: 4})
>>> try:
...     profile.run(native, resource_layout=bad_layout)
... except fq.errors.UnsupportedOperationError as error:
...     print(error)
CZGate is not supported on device operands (0, 4)
```

Move the second program qubit to the neighbouring device label `1`; the
Program itself does not need to change:

```pycon
>>> layout = fq.ResourceLayout({qubits[0]: 0, qubits[1]: 1})
>>> state = profile.run(native, resource_layout=layout).result().get_statevector()
>>> state.shape
(4,)
>>> int(np.argmax(np.abs(state) ** 2))
3
```

This confirms that the gate set and placement are valid. Fidelity, timing, and
pulse dynamics require a physical emulator.

## Add reference noise deliberately

The superconducting profiles are ideal unless a noise model is passed. Their
packaged models are useful comparison baselines, not current hardware
characterizations:

```python
profile_type = fq.simulator.SCQubitGoogleSimulator
noisy_profile = profile_type(
    grid_size=(2, 3),
    runtime="numpy",
    noise=profile_type.default_noise_model(),
)

measured_native = fq.Program(2, 2)
measured_native.add(ops.RX(np.pi), 0)
measured_native.add(ops.RX(np.pi), 1)
measured_native.add(ops.CZ, (0, 1))
measured_native.measure_all()
noisy_counts = noisy_profile.run(
    measured_native,
    shots=100,
    simulation_config={"seed": 7},
).result().get_counts()
```

Keeping noise opt-in makes the comparison legible: first verify target
compatibility, then decide whether the reference error model answers your
question. `AtomArraySimulator` has no packaged reference noise model; pass a
[`NoiseModel`][fatqat.NoiseModel] of your own when loading, loss, or other
effects belong in the experiment.

[`SCQubitIBMSimulator`][fatqat.simulator.SCQubitIBMSimulator] follows the same workflow
with a different native gate family. Inspect the selected profile's
implementation map, then make the Program and layout choices that it requires.

## Track atom occupancy and pairing

The atom-array profile asks a different hardware question. It has no fixed
geometry. Instead, `Put` establishes occupancy and `Pair`/`Unpair` change the
connectivity on which `CZ` is legal:

![Two occupied atoms begin separated, move together when Pair applies with depolarizing noise, remain paired for CZ, and separate again when Unpair applies with depolarizing noise.](../assets/generated/guide/atom-pairing-lifecycle.png)

??? example "Reproduce this figure"

    ```python
    import matplotlib.pyplot as plt

    stage_centers = (0.8, 2.8, 4.8, 6.8)
    separations = (1.00, 0.42, 0.42, 1.00)
    stage_labels = ("occupied", "paired", "CZ eligible", "unpaired")

    fig, ax = plt.subplots(figsize=(7.6, 2.35))

    def draw_atoms(center, separation, *, connected=False, gate=False, noisy=False):
        positions = (center - separation / 2.0, center + separation / 2.0)
        if noisy:
            ax.scatter(
                positions,
                (0.0, 0.0),
                s=1050,
                color="C1",
                alpha=0.13,
                edgecolor="none",
                zorder=1,
            )
        if connected:
            ax.plot(
                positions,
                (0.0, 0.0),
                color="C2" if not gate else "C4",
                linewidth=4.0,
                solid_capstyle="round",
                zorder=2,
            )
        ax.scatter(
            positions,
            (0.0, 0.0),
            s=610,
            facecolor="C0",
            edgecolor="white",
            linewidth=2.2,
            zorder=3,
        )
        for atom, position in enumerate(positions):
            ax.text(
                position,
                0.0,
                str(atom),
                color="white",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                zorder=4,
            )
        if gate:
            ax.text(
                center,
                0.0,
                "CZ",
                color="white",
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.28",
                    "facecolor": "C4",
                    "edgecolor": "white",
                    "linewidth": 1.2,
                },
                zorder=5,
            )

    draw_atoms(stage_centers[0], separations[0])
    draw_atoms(stage_centers[1], separations[1], connected=True, noisy=True)
    draw_atoms(stage_centers[2], separations[2], connected=True, gate=True)
    draw_atoms(stage_centers[3], separations[3], noisy=True)

    transitions = (
        (stage_centers[0], stage_centers[1], "Pair", True),
        (stage_centers[1], stage_centers[2], "CZ", False),
        (stage_centers[2], stage_centers[3], "Unpair", True),
    )
    for start, end, label, noisy in transitions:
        ax.annotate(
            "",
            xy=(end - 0.62, 0.72),
            xytext=(start + 0.62, 0.72),
            arrowprops={"arrowstyle": "->", "color": "0.40", "linewidth": 1.6},
        )
        midpoint = (start + end) / 2.0
        ax.text(
            midpoint,
            0.88,
            label,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="C0" if label != "CZ" else "C4",
        )
        if noisy:
            ax.text(
                midpoint,
                0.57,
                "+ depolarizing",
                ha="center",
                va="top",
                fontsize=8.5,
                color="C1",
            )

    for center, label in zip(stage_centers, stage_labels):
        ax.text(center, -0.66, label, ha="center", va="top", fontsize=9.5)

    ax.set_title("Occupancy stays present while pairing changes", fontsize=12, pad=4)
    ax.set(xlim=(0.0, 7.6), ylim=(-0.96, 1.24))
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    ```

The changing distance is a picture of the pairing intent, not a simulated
trajectory. `AtomArraySimulator` records no coordinates or movement duration;
`Pair` declares that the two occupied sites may execute native `CZ`, and
`Unpair` removes that eligibility.

```pycon
>>> atoms = fq.Program(2, 2)
>>> atoms.add(ops.Put, (0, 1))
>>> atoms.add(ops.Pair, (0, 1))
>>> atoms.add(ops.RX(np.pi), 0)
>>> atoms.add(ops.CZ, (0, 1))
>>> atoms.add(ops.Unpair, (0, 1))
>>> atoms.measure_all()
>>> atom_counts = fq.simulator.AtomArraySimulator(num_sites=2).run(
...     atoms,
...     shots=8,
...     simulation_config={"seed": 7},
... ).result().get_counts()
>>> atom_counts
{'01': 8}
```

Pairing is ideal unless you attach a noise assumption. For example, apply a
small depolarizing channel independently to each atom whenever either movement
instruction occurs:

```pycon
>>> movement_noise = fq.NoiseModel()
>>> for movement in (ops.Pair, ops.Unpair):
...     for target_position in (0, 1):
...         movement_noise.add(
...             fq.noise.Depolarizing(p=0.02),
...             operation=movement,
...             target_positions=target_position,
...         )
>>> noisy_atom_backend = fq.simulator.AtomArraySimulator(
...     num_sites=2,
...     noise=movement_noise,
... )
>>> noisy_counts = noisy_atom_backend.run(
...     atoms,
...     shots=100,
...     simulation_config={"seed": 7},
... ).result().get_counts()
>>> sum(noisy_counts.values())
100
```

This channel perturbs the quantum state during `Pair` and `Unpair`; it does not
remove either atom. Use [`Loss`][fatqat.noise.Loss] instead when movement
should change occupancy.

Omitting `Pair` before `CZ` is a Program error; FatQat does not transport or
pair atoms automatically. A missing atom is different: an empty site reports
the erasure digit `2`, while supported gates simply have no atom to act on.

For native sets, capacities, and method support, use the
[hardware-profile API](../api/simulators/index.md). Continue to
[Hamiltonian emulation](hamiltonian-emulation.md) when pulse duration,
physical levels, drift, or continuous-time noise becomes relevant.
