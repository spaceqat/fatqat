Noise (``fq.noise``)
====================

Channel-representable noise and classical readout error are configured
separately from a :py:class:`~fatqat.Program`. A :py:class:`~fatqat.NoiseModel` describes how a backend
alters a run; it is not an instruction you add to a program.

Build the program and the noise model independently, then pass the model as
``noise=...`` when constructing :py:class:`~fatqat.backends.SimulatorBackend`. The backend combines
them only while executing the program.

Create and attach noise
-----------------------

:py:class:`~fatqat.NoiseModel` is normally created as ``noise = fq.NoiseModel()``.

:py:meth:`add_channel <fatqat.NoiseModel.add_channel>`
(``channel, *, operation=None, targets=None, slots=None``) registers a quantum
channel. Omit ``operation`` for always-on noise, or provide an operation such
as ``operation=op.X`` to activate it only for matching occurrences. Omit
``targets`` for the corresponding default scope; use a qubit reference such
as ``targets=(program.quantum_registers[0][0],)`` to select one qubit.

:py:meth:`add_readout_error <fatqat.NoiseModel.add_readout_error>` (``confusion_matrix, *, target=None``) attaches
classical measurement error. The matrix uses
``C[reported, true] = P(reported | true)``.

Built-in channels
-----------------

- :py:class:`~fatqat.noise.Depolarizing` (``p``)
- :py:class:`~fatqat.noise.AmplitudeDamping` (``p=(p,)`` or ``rate=(rate,)``)
- :py:class:`~fatqat.noise.PhaseDamping` (``p`` or ``rate``)

``AmplitudeDamping`` and ``PhaseDamping`` accept exactly one of a finite
probability ``p`` or a continuous ``rate`` (the inverse of the target
backend's declared time unit). For operation-scoped noise, matrix backends
resolve ``p`` directly into Kraus operators and reject ``rate`` mode (no
duration is available); the superconducting pulse backend resolves either
mode into a collapse-operator rate using the realized operation duration.
Always-on damping requires ``rate`` mode and a time-aware pulse backend.

:py:class:`~fatqat.noise.ThermalRelaxation` (``t1``, ``t2``) describes the
same T1/T2 model for always-on pulse noise and offers
``as_channels(duration)`` to produce compatible finite qubit channels.

Always-on pulse noise
---------------------

On the superconducting pulse backend, register
:py:class:`~fatqat.noise.ThermalRelaxation` or rate-mode damping without an
``operation`` to act throughout pulse and idle evolution:

.. code-block:: python

   noise.add_channel(
       fq.noise.AmplitudeDamping(rate=(0.001, 0.002)),
       targets=(program.quantum_registers[0][0],),
   )

Providing ``operation=op.X`` instead scopes the same descriptor to each
matching placed pulse interval. Matrix-family backends reject always-on
entries because they have no continuous-time evolution model. The pulse
backend still rejects unsupported descriptors such as ``Depolarizing``.
Coherent ZZ is not available in v0.1.

The :doc:`../guide/noise` guide explains method choice, target selection,
and a readout-error matrix example. Physical device selectors and custom
channel implementation machinery are backend-author concerns and are not
part of the normal application API.

Detailed reference
------------------

.. autoclass:: fatqat.NoiseModel
   :members: add_channel, channels_for, always_on_channels_for,
      add_readout_error, readout_error_for, validate_for,
      has_readout_error, has_noise_for, channel_types, channel_registrations
   :show-inheritance:

.. autoclass:: fatqat.noise.Depolarizing
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
