Noise
=====

FatQat keeps noise separate from :class:`~fatqat.Program`. Add the noise
sources you need to a :class:`~fatqat.NoiseModel`, then pass that model to a
compatible simulator or emulator with ``noise=...``. You can reuse the same
program for ideal and noisy runs.

See :doc:`../guide/ideal-and-noisy` for a controlled ideal-versus-noisy
comparison. This section is the reference for selectors, support, units, and
validation.

Noise types live in ``fatqat.noise``. ``NoiseModel`` is also available as
``fatqat.NoiseModel``.

Choose a noise type
-------------------

Probabilities describe one simulator channel application after a matched
operation. Rates and relaxation times describe local Lindblad operators that
act over emulator time. Backends do not convert between these forms.

.. list-table:: Built-in noise types
   :header-rows: 1
   :widths: 23 25 23 29

   * - Noise type
     - Accepted parameters
     - Applies to
     - User-visible effect
   * - :class:`~fatqat.noise.Depolarizing`
     - Exactly one of ``p`` or ``rate``
     - ``p``: the selected operands; ``rate``: one subsystem
     - Uniform mixing toward the maximally mixed state
   * - :class:`~fatqat.noise.PauliChannel`
     - Pauli-string probability mapping or pair sequence
     - One qubit per string character
     - A stochastic mixture of ``I``, ``X``, ``Y``, and ``Z`` strings
   * - :class:`~fatqat.noise.AmplitudeDamping`
     - Exactly one of ``p`` or ``rate``, with one value per adjacent transition
     - One subsystem
     - Ladder decay from level ``k`` to ``k - 1``
   * - :class:`~fatqat.noise.PhaseDamping`
     - Exactly one of ``p``, ``rate``, or ``t_phi``
     - One subsystem
     - Coherence decay without population transfer
   * - :class:`~fatqat.noise.ThermalRelaxation`
     - ``t1`` and ``t2``
     - One subsystem
     - Combined energy relaxation and residual pure dephasing
   * - :class:`~fatqat.noise.Loss`
     - Per-carrier probability ``p``
     - Every selected carrier in a matched operation
     - Persistent carrier removal on an occupancy-aware backend
   * - :class:`~fatqat.noise.ReadoutConfusion`
     - Column-stochastic report matrix
     - Each measured subsystem independently, or one selected subsystem
     - Classical resampling of the reported digit after physical collapse

See :doc:`noise/model` for operation and background noise, target selection,
composition, conflicts, and validation timing. Each noise-type page covers
its parameters, units, and mathematical definition.

Backend support
---------------

Support depends on the noise form, where it applies, and the backend. The
:ref:`noise-backend-support` tables show what works out of the box, what needs
a custom implementation map, and what a backend family cannot support.

Quick start
-----------

This model adds a joint channel after every ``CX`` and then applies binary
readout confusion to every measurement:

.. code-block:: python

   import numpy as np
   import fatqat as fq
   import fatqat.operations as ops

   noise = fq.NoiseModel()
   noise.add(fq.noise.Depolarizing(p=0.05), operation=ops.CX)
   noise.add(
       fq.noise.ReadoutConfusion(
           np.array([[0.98, 0.04], [0.02, 0.96]])
       )
   )

   backend = fq.simulator.Simulator(method="density_matrix", noise=noise)

For a pulse backend, express rates and relaxation times in the model's time
unit. For example, the reference transmon model uses device labels such as
``"q0"``. This relaxation noise stays active there throughout elapsed pulse
time:

.. code-block:: python

   pulse_noise = fq.NoiseModel()
   pulse_noise.add(
       fq.noise.ThermalRelaxation(t1=60_000.0, t2=80_000.0),
       targets="q0",
   )

The reference transmon model uses nanoseconds, while the neutral-atom models
use microseconds. Check the chosen model's ``time_unit`` instead of inferring a
unit from the size of a value: see
:attr:`~fatqat.emulator.superconducting.TransmonModel.time_unit`,
:attr:`~fatqat.emulator.Atom2LevelModel.time_unit`, and
:attr:`~fatqat.emulator.Atom3LevelModel.time_unit`.

API pages
---------

.. toctree::
   :maxdepth: 1

   noise/model
   noise/backend-support
   noise/depolarizing
   noise/pauli-channel
   noise/amplitude-damping
   noise/phase-damping
   noise/thermal-relaxation
   noise/loss
   noise/readout-confusion
   noise/custom-implementations
