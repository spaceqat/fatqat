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

:py:meth:`add_noise <fatqat.NoiseModel.add_noise>` (``operation, channel, *, targets=None``) attaches a
quantum channel after matching gate occurrences. Omit ``targets`` to affect
every occurrence; use a qubit reference such as
``targets=(program.qreg[0][0],)`` to affect one qubit.

:py:meth:`add_readout_error <fatqat.NoiseModel.add_readout_error>` (``confusion_matrix, *, target=None``) attaches
classical measurement error. The matrix uses
``C[reported, true] = P(reported | true)``.

Built-in channels
-----------------

- :py:class:`~fatqat.noise.Depolarizing` (``p``)
- :py:class:`~fatqat.noise.AmplitudeDamping` (``gammas=(gamma,)``)
- :py:class:`~fatqat.noise.PhaseDamping` (``p``)

:py:func:`~fatqat.noise.relaxation_channels` (``t1, t2, duration``) returns compatible
damping and dephasing channels for a qubit gate with a known duration.

The :doc:`../guide/noise` guide explains method choice, target selection,
and a readout-error matrix example. Physical device selectors and custom
channel implementation machinery are backend-author concerns and are not
part of the normal application API.

Detailed reference
------------------

.. autoclass:: fatqat.NoiseModel
   :members:
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

.. autofunction:: fatqat.noise.relaxation_channels
