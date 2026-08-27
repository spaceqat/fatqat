Noise (``fq.noise``)
====================

Noise is authored independently from a :py:class:`~fatqat.Program`. Put
physical declarations in :py:class:`~fatqat.NoiseModel`, then pass the model
to a simulator or emulator with ``noise=...``.

Quick start
-----------

Use :py:meth:`~fatqat.NoiseModel.add` for every supported source. The
declaration identifies the source; the remaining arguments identify its
activation scope.

.. code-block:: python

   import numpy as np
   import fatqat as fq
   import fatqat.operations as ops

   noise = fq.NoiseModel()

   # A finite channel at every X occurrence.
   noise.add(fq.noise.PhaseDamping(p=0.01), operation=ops.X)

   # A different physical source on CZ's first operand.
   noise.add(
       fq.noise.AmplitudeDamping(p=0.002),
       operation=ops.CZ,
       target_positions=0,
   )

   # Classical measurement-report confusion.
   noise.add(
       fq.noise.ReadoutConfusion(
           np.array([[0.98, 0.04], [0.02, 0.96]])
       )
   )

Matrix simulators accept supported finite channel declarations with an
``operation``. Pulse emulators accept supported local generator/time
declarations either on an operation window or as background noise on exactly
one target. FATQAT does not infer a generator from a finite probability or a
finite channel from a rate.

.. code-block:: python

   pulse_noise = fq.NoiseModel()
   pulse_noise.add(
       fq.noise.ThermalRelaxation(t1=60.0, t2=80.0),
       targets="q0",
   )
   pulse_noise.add(
       fq.noise.PhaseDamping(rate=0.002),
       operation=ops.X,
       targets="q0",
   )

The :doc:`../guide/noise` guide covers ordered target selectors,
``target_positions``, conflict rejection, explicit conversion utilities,
readout semantics, carrier loss, backend lifecycle capture, and custom
implementation maps.

Noise model
-----------

.. autoclass:: fatqat.NoiseModel
   :members: add
   :show-inheritance:

Finite and generator-capable channels
-------------------------------------

.. autoclass:: fatqat.noise.Channel
   :members:

.. autoclass:: fatqat.noise.Depolarizing
   :members:
   :show-inheritance:

.. autoclass:: fatqat.noise.PauliChannel
   :members:
   :show-inheritance:

.. autoclass:: fatqat.noise.AmplitudeDamping
   :members:
   :show-inheritance:

.. autoclass:: fatqat.noise.PhaseDamping
   :members:
   :show-inheritance:

.. autoclass:: fatqat.noise.ThermalRelaxation
   :members:
   :show-inheritance:

Specialized physical and classical sources
------------------------------------------

.. autoclass:: fatqat.noise.Loss
   :members:

.. autoclass:: fatqat.noise.ReadoutConfusion
   :members:

Backend implementation maps
---------------------------

Most users only choose declarations and a backend. Backend authors can extend
the exact-type implementation maps without changing :class:`NoiseModel`.
Finite-channel rules and pulse-generator rules are intentionally separate.

.. autoclass:: fatqat.noise.ChannelImplementationMap
   :members: add, supported_channels

.. autofunction:: fatqat.noise.default_channel_implementation_map

Pulse-generator map signatures and defaults are documented in
:doc:`pulse-emulator`.
