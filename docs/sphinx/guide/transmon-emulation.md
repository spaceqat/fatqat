# Emulate a superconducting system

A transmon is operated as a qubit but modeled with at least three levels.
{py:class}`~fatqat.emulator.TransmonEmulator` keeps the third level so that
pulse-induced leakage remains visible, alongside timing and coupling effects.
It uses the shared {doc}`Hamiltonian-emulation workflow
<hamiltonian-emulation>`.

## Load a reproducible baseline

The packaged document describes two coupled physical transmons. It is a
simulation baseline, not a live calibration from a named device:

```{doctest}
>>> import numpy as np
>>> import fatqat as fq
>>> import fatqat.operations as ops
>>> model = fq.emulator.TransmonModel.from_document(
...     fq.emulator.load_model_document("transmon.reference")
... )
>>> model.subsystem_ids
('q0', 'q1')
>>> backend = fq.emulator.TransmonEmulator(model)
```

Program qubits bind to those subsystem IDs in declaration order unless a
{py:class}`~fatqat.ResourceLayout` says otherwise. Every model transmon remains
in the physical state even when the Program addresses only one of them.

## Run the rotation as a calibrated gate

Reuse the guide's central rotation as an ordinary Program operation. The
emulator realizes it with its gate implementation map and reference
calibration:

```{doctest}
>>> rotation = fq.Program(1)
>>> rotation.add(ops.RX(np.pi / 2), 0)
>>> calibrated_rho = backend.run(rotation).result().get_density_matrix()
>>> calibrated_rho.shape
(9, 9)
```

The Program declares a logical qubit, but the result covers the complete
two-transmon qutrit space. To inspect `q0`, reshape the diagonal in FatQat's
little-endian physical-axis order and sum over `q1`:

```{doctest}
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

```{doctest}
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

```{eval-rst}
.. plot::
   :include-source: false
   :alt: Computational-level populations for a calibrated rotation and a direct drive are shown beside a magnified comparison of their level-two leakage percentages.

   import matplotlib.pyplot as plt
   import numpy as np
   import fatqat as fq
   import fatqat.operations as ops

   model = fq.emulator.TransmonModel.from_document(
       fq.emulator.load_model_document("transmon.reference")
   )
   backend = fq.emulator.TransmonEmulator(model)

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
one Program. The {doc}`transmon emulator API <../api/pulse-emulator>` lists the
supported units, gates, noise forms, and solver options.
