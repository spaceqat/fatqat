# User guide

Write the quantum computation once, as a {py:class}`~fatqat.Program`. Then
choose how closely you want FatQat to model the machine beneath it.

That choice gives you three useful views of the same program:

:::::{grid} 1
:gutter: 3
:class-container: guide-paths

::::{grid-item-card} Explore the algorithm
:link: simulation
:link-type: doc
:link-alt: Explore algorithm behavior with the general simulator
:class-card: guide-path-card
:class-body: guide-path-card-body

:::{container} guide-path-copy
Build and inspect an algorithm as a Program, then use the general-purpose
simulator for states, parameter sweeps, counts, and observables.
:::

```{eval-rst}
.. plot::
   :alt: A five-qubit variational ansatz applies RX and RY rotations, two staggered layers of parallel CZ gates, a barrier, and the next RX and RY rotations.
   :filename-prefix: guide-path-algorithm
   :include-source: false

   import matplotlib.pyplot as plt

   import fatqat as fq
   import fatqat.operations as ops

   ansatz = fq.Program(5)
   first_rx = [fq.Parameter(f"x0_{qubit}") for qubit in range(5)]
   first_ry = [fq.Parameter(f"y0_{qubit}") for qubit in range(5)]
   second_rx = [fq.Parameter(f"x1_{qubit}") for qubit in range(5)]
   second_ry = [fq.Parameter(f"y1_{qubit}") for qubit in range(5)]

   for qubit, angle in enumerate(first_rx):
       ansatz.add(ops.RX(angle), qubit)
   for qubit, angle in enumerate(first_ry):
       ansatz.add(ops.RY(angle), qubit)
   for pair in ((0, 1), (2, 3)):
       ansatz.add(ops.CZ, pair)
   for pair in ((1, 2), (3, 4)):
       ansatz.add(ops.CZ, pair)
   # Mark the boundary between variational layers.
   ansatz.add(ops.Barrier, tuple(range(5)))
   for qubit, angle in enumerate(second_rx):
       ansatz.add(ops.RX(angle), qubit)
   for qubit, angle in enumerate(second_ry):
       ansatz.add(ops.RY(angle), qubit)

   fig, ax = plt.subplots(figsize=(4.7, 2.4))
   ansatz.draw(ax=ax)
   ax.set_title("5-qubit VQA ansatz", fontsize=10.5, pad=4)
   fig.tight_layout(pad=0.25)
```
::::

::::{grid-item-card} Test hardware constraints
:link: hardware-profile-simulation
:link-type: doc
:link-alt: Test a Program against a hardware profile
:class-card: guide-path-card
:class-body: guide-path-card-body

:::{container} guide-path-copy
Keep gate-level execution, but add native operations, placement,
connectivity, occupancy, and a selected hardware profile.
:::

```{eval-rst}
.. plot::
   :alt: A three-by-four device topology highlights one native nearest-neighbor CZ in green and one unsupported diagonal CZ in red.
   :filename-prefix: guide-path-hardware
   :include-source: false

   import matplotlib.pyplot as plt
   from matplotlib.patches import FancyBboxPatch

   import fatqat as fq
   import fatqat.operations as ops

   profile = fq.simulator.SCQubitGoogleSimulator(
       grid_size=(3, 4),
       runtime="numpy",
   )
   gate_map = profile.implementation_map
   edges = {
       tuple(sorted(edge))
       for edge in gate_map.device_operands_for(ops.CZ)
   }
   native_pair = (9, 10)
   diagonal_pair = (1, 6)
   assert gate_map.supports(ops.CZ, device_operands=native_pair)
   assert not gate_map.supports(ops.CZ, device_operands=diagonal_pair)

   positions = {site: (site % 4, 2 - site // 4) for site in range(12)}
   fig, ax = plt.subplots(figsize=(4.2, 2.55))

   chip = FancyBboxPatch(
       (-0.38, -0.38),
       3.76,
       2.76,
       boxstyle="round,pad=0.08,rounding_size=0.18",
       facecolor="C0",
       edgecolor="0.72",
       linewidth=1.0,
       alpha=0.07,
       zorder=0,
   )
   ax.add_patch(chip)

   for left, right in edges:
       x = (positions[left][0], positions[right][0])
       y = (positions[left][1], positions[right][1])
       ax.plot(x, y, color="0.76", linewidth=1.6, zorder=1)

   for site, (x, y) in positions.items():
       ax.scatter(x, y, s=500, color="C0", alpha=0.10, edgecolor="none", zorder=2)
       ax.scatter(
           x,
           y,
           s=360,
           facecolor="white",
           edgecolor="0.42",
           linewidth=1.2,
           zorder=3,
       )
       ax.text(x, y, str(site), ha="center", va="center", fontsize=8.5, zorder=4)

   native_x = tuple(positions[site][0] for site in native_pair)
   native_y = tuple(positions[site][1] for site in native_pair)
   ax.plot(native_x, native_y, color="C2", linewidth=4.0, zorder=2)
   ax.scatter(
       native_x,
       native_y,
       s=610,
       facecolor="none",
       edgecolor="C2",
       linewidth=2.2,
       zorder=5,
   )
   ax.text(
       1.5,
       -0.30,
       "native CZ",
       color="C2",
       ha="center",
       va="center",
       fontsize=9.5,
       fontweight="bold",
   )

   rejected_x = tuple(positions[site][0] for site in diagonal_pair)
   rejected_y = tuple(positions[site][1] for site in diagonal_pair)
   ax.plot(
       rejected_x,
       rejected_y,
       color="C3",
       linestyle=(0, (3, 2)),
       linewidth=2.4,
       zorder=4,
   )
   ax.scatter(
       rejected_x,
       rejected_y,
       s=610,
       facecolor="none",
       edgecolor="C3",
       linewidth=2.2,
       zorder=5,
   )
   ax.text(
       1.5,
       1.5,
       "X",
       color="C3",
       ha="center",
       va="center",
       fontsize=10,
       fontweight="bold",
       bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
       zorder=6,
   )
   ax.text(1.78, 1.62, "not connected", color="C3", fontsize=9.5)

   ax.set_title("CZ placement on a 3 x 4 device", fontsize=11, pad=4)
   ax.set(xlim=(-0.48, 3.48), ylim=(-0.48, 2.48), aspect="equal")
   ax.axis("off")
   fig.tight_layout(pad=0.3)
```
::::

::::{grid-item-card} Follow the physics
:link: hamiltonian-emulation
:link-type: doc
:link-alt: Follow a Program into Hamiltonian-level dynamics
:class-card: guide-path-card
:class-body: guide-path-card-body

:::{container} guide-path-copy
Turn gates or direct controls into a timed schedule and integrate the physical
Hamiltonian, including leakage and continuous-time noise.
:::

```{eval-rst}
.. plot::
   :alt: A driven-atom spectroscopy heatmap shows Rydberg population versus pulse duration and detuning, with resonant Rabi oscillations and off-resonant chevrons.
   :filename-prefix: guide-path-physics
   :include-source: false

   import matplotlib.pyplot as plt
   import numpy as np

   import fatqat as fq
   import fatqat.operations as ops

   model = fq.emulator.Atom2LevelModel.from_document(
       fq.emulator.load_model_document("atom2level.reference")
   )
   arrangement = fq.emulator.AtomArrangement.rectangular(
       1,
       1,
       spacing=6.0,
   )
   backend = fq.emulator.Atom2LevelEmulator(
       model,
       arrangement=arrangement,
       method="unitary",
   )
   omega = 2.0 * np.pi
   durations = np.linspace(0.0, 1.5, 31)
   detunings = np.linspace(-3.0 * omega, 3.0 * omega, 25)
   step = durations[1] - durations[0]
   population = np.zeros((len(detunings), len(durations)))
   ground_state = np.array([1.0, 0.0], dtype=complex)

   for row, detuning in enumerate(detunings):
       drive = fq.emulator.PulseControl(
           model.control.drive(),
           fq.emulator.SampledWaveform((0.0, step), (omega, omega)),
       )
       offset = fq.emulator.PulseControl(
           model.control.detuning(),
           fq.emulator.SampledWaveform((0.0, step), (detuning, detuning)),
       )
       program = fq.Program(arrangement.num_sites)
       program.add(ops.PulseOperation(step, (drive, offset)))
       step_unitary = backend.run(program, shots=0).result().get_unitary()

       state = ground_state.copy()
       for column in range(1, len(durations)):
           state = step_unitary @ state
           population[row, column] = abs(state[1]) ** 2

   resonance = population[len(detunings) // 2]
   np.testing.assert_allclose(
       resonance,
       np.sin(omega * durations / 2.0) ** 2,
       atol=1e-7,
   )
   np.testing.assert_allclose(population, population[::-1], atol=1e-7)

   fig, ax = plt.subplots(figsize=(4.2, 2.7))
   image = ax.pcolormesh(
       durations,
       detunings / (2.0 * np.pi),
       population,
       shading="gouraud",
       cmap="magma",
       vmin=0.0,
       vmax=1.0,
   )
   ax.contour(
       durations,
       detunings / (2.0 * np.pi),
       population,
       levels=(0.25, 0.5, 0.75),
       colors="white",
       linewidths=0.45,
       alpha=0.42,
   )
   ax.axhline(0.0, color="white", linestyle=(0, (3, 2)), linewidth=1.0)
   ax.text(
       1.47,
       0.18,
       "resonance",
       color="white",
       ha="right",
       va="bottom",
       fontsize=8.5,
   )
   ax.set(
       xlabel=r"pulse duration ($\mu$s)",
       ylabel=r"detuning $\Delta / 2\pi$ (MHz)",
       xlim=(0.0, 1.5),
       ylim=(-3.0, 3.0),
       xticks=(0.0, 0.5, 1.0, 1.5),
       yticks=(-3.0, 0.0, 3.0),
   )
   ax.set_title("Driven-atom spectroscopy", fontsize=11, pad=4)
   ax.tick_params(labelsize=9.5)
   ax.xaxis.label.set_size(10.5)
   ax.yaxis.label.set_size(10.5)
   colorbar = fig.colorbar(image, ax=ax, pad=0.025, aspect=18)
   colorbar.set_label(r"$P_r$", rotation=0, labelpad=8, fontsize=10.5)
   colorbar.set_ticks((0.0, 0.5, 1.0))
   colorbar.ax.tick_params(labelsize=9)
   fig.tight_layout(pad=0.35)
```
::::

:::::

If you come from another quantum SDK, you may be looking for a `Circuit`
class. In FatQat, the object you want is `Program`.

Why the different name? A circuit usually describes the gate-level view of a
computation. A Program can describe that circuit, but it can also include
classical conditions, qubits and qudits with different local dimensions, and
direct physical pulse controls. Throughout this guide, *circuit* means the
gate-level view and `Program` means the complete description passed to a
backend. There is no separate `Circuit` class to learn.

Whichever path you choose, you author that one `Program`. The execution target
determines which instructions it supports and how much detail it returns.

Not sure where your question belongs? [Run one rotation through all three
levels](execution-models.md) and compare what each result contains.

## Begin with a working program

New to FatQat? [Build and run a Bell program](quickstart.md). It takes about
ten minutes and ends with a circuit drawing and a counts plot.

When you are ready to go beyond the first circuit, [write a richer
Program](program.md). That chapter introduces named registers, classical
control, reusable parameters, and mixed qubit–qutrit systems without changing
the authoring model.

:::{tip}
The guide teaches complete workflows and the reasoning behind them. Follow a
link to the [API reference](../api/index.rst) when you need a precise signature
or contract. The [tutorials](../tutorials/index.rst) are longer case studies
built on the same features.
:::

```{toctree}
:maxdepth: 1
:caption: Start with one Program
:hidden:

quickstart
program
execution-models
```

```{toctree}
:maxdepth: 1
:caption: Study algorithm behavior
:hidden:

simulation
interpret-results
ideal-and-noisy
performance
```

```{toctree}
:maxdepth: 1
:caption: Study hardware behavior
:hidden:

hardware-profile-simulation
hamiltonian-emulation
transmon-emulation
neutral-atom-emulation
```

```{toctree}
:maxdepth: 1
:caption: Connect another circuit workflow
:hidden:

interoperability
```

```{toctree}
:maxdepth: 1
:caption: Get unstuck
:hidden:

troubleshooting
```
