# Emulate a superconducting system

A transmon is operated as a qubit but modeled with at least three levels.
[`TransmonEmulator`][fatqat.emulator.TransmonEmulator] keeps the third level so that
pulse-induced leakage remains visible, alongside timing and coupling effects.
It uses the shared [Hamiltonian-emulation workflow](hamiltonian-emulation.md).

## Load a reproducible baseline

The packaged document describes two coupled physical transmons. It is a
simulation baseline, not a live calibration from a named device:

```pycon
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> model_document = fq.emulator.load_model_document("transmon.reference")
>>> model_document["parameters"]["subsystems"]["q0"]["frequency"]
5.1
>>> model = fq.emulator.TransmonModel.from_document(model_document)
>>> model.subsystem_ids
('q0', 'q1')
>>> backend = fq.emulator.TransmonEmulator(model, method="density_matrix")
```

Program qubits bind to those subsystem IDs in declaration order unless a
[`ResourceLayout`][fatqat.ResourceLayout] says otherwise. Every model transmon remains
in the physical state even when the Program addresses only one of them.
Retain `model_document` when you need persisted frequencies, anharmonicities,
model identity, or coupling topology; the runtime model intentionally exposes
execution capabilities rather than normalized copies of those records.
This guide chooses the density-matrix method because it repeatedly inspects
populations. The common default is `method="statevector"`; use
`method="unitary"` when the complete coherent operator is the result you need.

## Generate a grid reference

For a rectangular nearest-neighbor grid, generate matching model and
calibration documents, parse both, and pass the resulting map explicitly to
the emulator:

```pycon
>>> reference = fq.emulator.generate_transmon_grid_reference(
...     shape=(2, 2),
...     frequency_groups_ghz=(5.0, 5.2),
...     seed=0,
... )
>>> grid_model = fq.emulator.TransmonModel.from_document(
...     reference.model_document
... )
>>> grid_calibration = fq.emulator.TransmonCalibration(
...     reference.calibration_document
... )
>>> grid_gate_map = fq.emulator.default_transmon_gate_implementation_map(
...     model=grid_model,
...     calibration=grid_calibration,
... )
>>> grid_backend = fq.emulator.TransmonEmulator(
...     grid_model,
...     gate_implementation_map=grid_gate_map,
... )
>>> grid_model.subsystem_ids
('q0', 'q1', 'q2', 'q3')
```

The generated calibration contains fixed analytic reference recipes; this
workflow does not run numerical pulse calibration. The documents are ordinary
JSON-compatible mappings, so standard-library JSON tools are sufficient when
you want to persist and read them later. The following is a literal file-I/O
example and is not executed by the documentation build:

```python
import json

with open("model.json", "w", encoding="utf-8") as stream:
    json.dump(reference.model_document, stream, indent=2)
with open("calibration.json", "w", encoding="utf-8") as stream:
    json.dump(reference.calibration_document, stream, indent=2)

with open("model.json", encoding="utf-8") as stream:
    saved_model = fq.emulator.TransmonModel.from_document(json.load(stream))
with open("calibration.json", encoding="utf-8") as stream:
    saved_calibration = fq.emulator.TransmonCalibration(json.load(stream))
```

For `N` transmons, the physical qutrit Hilbert-space dimension is `3**N`.

The generator warnings cover idle fabrication spread only. The current
effective rotating-frame/RWA model cannot evaluate laboratory-frequency
crossings during ramp, park, or return, including spectator crossings. When
such a check matters, keep each relevant pairwise frequency-separation sign
unchanged with a positive margin throughout the trajectory, using your own
carrier/flux mapping or JSON post-processing.

## Run the rotation as a calibrated gate

Reuse the guide's central rotation as an ordinary Program operation. The
emulator realizes it with its gate implementation map and reference
calibration:

```pycon
>>> rotation = fq.Program(1)
>>> rotation.add(ops.RX(np.pi / 2), 0)
>>> calibrated_rho = backend.run(rotation).result().get_density_matrix()
>>> calibrated_rho.shape
(9, 9)
```

The Program declares a logical qubit, but the result covers the complete
two-transmon qutrit space. To inspect `q0`, reshape the diagonal in FatQat's
little-endian physical-axis order and sum over `q1`:

```pycon
>>> calibrated_physical = np.real(np.diag(calibrated_rho)).reshape(
...     (3, 3), order="F"
... )
>>> calibrated_q0 = calibrated_physical.sum(axis=1)
>>> np.allclose(calibrated_q0.sum(), 1.0)
True
>>> bool(calibrated_q0[2] < 1e-6)
True
```

Here `calibrated_q0[2]` is population in physical level `|2>`. This is a
physical qutrit used to model transmon leakage; it is not the logical qutrit
authoring feature supported by the general simulator.

## Replace the gate with a direct drive

To see how pulse shape changes the physical behavior, replace the calibrated
`RX` with a drive on the same transmon. The pulse below is intentionally
hard-edged so its leakage is visible:

```pycon
>>> duration = 20.0
>>> drive = fq.emulator.SampledWaveform(
...     (0.0, duration),
...     (0.08, 0.08),
... )
>>> control = fq.emulator.PulseControl(model.control.drive("q0"), drive)
>>> direct = fq.Program(1)
>>> direct.add(ops.PulseOperation(duration, (control,)))
>>> direct_rho = backend.run(direct).result().get_density_matrix()
>>> direct_physical = np.real(np.diag(direct_rho)).reshape((3, 3), order="F")
>>> direct_q0 = direct_physical.sum(axis=1)
>>> q0_leakage = direct_q0[2]
>>> f"{100 * q0_leakage:.2f}%"
'0.64%'
```

`q0_leakage` is the probability that the driven transmon finishes in physical
level `|2>`. The plot keeps that small value on its own scale so it does not
disappear beside the computational-level populations.

![Computational-level populations for a calibrated rotation and a direct drive are shown beside a magnified comparison of their level-two leakage percentages.](../assets/generated/guide/transmon-emulation-1.png)

??? example "Reproduce this figure"

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    import fatqat as fq
    import fatqat.operations as ops

    model = fq.emulator.TransmonModel.from_document(
        fq.emulator.load_model_document("transmon.reference")
    )
    backend = fq.emulator.TransmonEmulator(model, method="density_matrix")

    calibrated = fq.Program(1)
    calibrated.add(ops.RX(np.pi / 2), 0)
    calibrated_rho = backend.run(calibrated).result().get_density_matrix()

    duration = 20.0
    waveform = fq.emulator.SampledWaveform(
        (0.0, duration),
        (0.08, 0.08),
    )
    control = fq.emulator.PulseControl(model.control.drive("q0"), waveform)
    direct = fq.Program(1)
    direct.add(ops.PulseOperation(duration, (control,)))
    direct_rho = backend.run(direct).result().get_density_matrix()

    def q0_populations(rho):
        physical = np.real(np.diag(rho)).reshape((3, 3), order="F")
        return physical.sum(axis=1)

    calibrated_q0 = q0_populations(calibrated_rho)
    direct_q0 = q0_populations(direct_rho)
    assert np.allclose(calibrated_q0.sum(), 1.0)
    assert np.allclose(direct_q0.sum(), 1.0)

    levels = np.arange(2)
    width = 0.36
    fig, (population_ax, leakage_ax) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.8),
        gridspec_kw={"width_ratios": (2.5, 1.0)},
    )
    population_ax.bar(
        levels - width / 2,
        calibrated_q0[:2],
        width,
        label="calibrated RX(pi/2)",
    )
    population_ax.bar(
        levels + width / 2,
        direct_q0[:2],
        width,
        label="direct drive",
    )
    population_ax.set(
        xticks=levels,
        xticklabels=("|0>", "|1>"),
        ylabel="population on q0",
        ylim=(0.0, 1.08),
    )
    population_ax.legend()
    leakage_bars = leakage_ax.bar(
        ("calibrated", "direct"),
        100 * np.array((calibrated_q0[2], direct_q0[2])),
        color=("C0", "C1"),
    )
    leakage_ax.bar_label(leakage_bars, fmt="%.2f%%", padding=3)
    leakage_ax.set(ylabel="|2> population (%)")
    leakage_ax.set_ylim(0.0, max(0.75, 120 * direct_q0[2]))
    leakage_ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    ```

## Know when more physics changes the answer

- **Coupling:** two-transmon gates are available only on coupling edges in the
  model, and an unaddressed neighbour is still part of the Hamiltonian.
- **Frames and timing:** calibrated frame changes and the placement of later
  controls can change phases even when computational populations look alike.
- **Continuous noise:** rate- or time-form Lindblad declarations act over
  elapsed physical time, rather than once at a circuit-operation boundary.

Use the calibrated path to study the supplied gate recipe. Use direct controls
when the waveform itself is the experiment; the two forms can also coexist in
one Program. The [transmon emulator API](../api/pulse-emulator.md) lists the
supported units, gates, noise forms, and execution methods.
