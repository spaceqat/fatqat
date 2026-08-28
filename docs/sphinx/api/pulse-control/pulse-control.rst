PulseControl
============

.. currentmodule:: fatqat.emulator

:py:class:`PulseControl` assigns a waveform to one physical control channel.
Get the channel from a model's ``control`` selectors and use the result in a
:doc:`pulse-operation` or :doc:`gate-realization` definition. Do not construct
:py:class:`ControlChannel` directly.

Methods on ``model.control`` check their address arguments when they create a
channel. When the control is used, the emulator checks that its model family
and named resources are compatible and that the waveform meets the model's
limits. A channel can therefore be reused with another compatible model that
contains the same resource. The built-in methods and their units are listed
in :doc:`index`; see
:doc:`sampled-waveform` for interpolation outside the sample grid.

Reference
---------

.. autoclass:: fatqat.emulator.PulseControl
   :members:
   :no-inherited-members:

.. autoclass:: fatqat.emulator.ControlChannel
   :no-members:
   :no-inherited-members:
