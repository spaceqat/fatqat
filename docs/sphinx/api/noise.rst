Noise (``fq.noise``)
====================

Channel-representable noise and classical readout error. Most users only
need `NoiseModel` (also exported as ``fq.NoiseModel``) plus the catalog
descriptors; the rule/registry machinery at the end is for defining custom
channels (see the noise guide).

.. autoclass:: fatqat.NoiseModel
   :members:
   :show-inheritance:

Catalog
-------

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

Custom channels
---------------

.. autoclass:: fatqat.noise.Channel
   :members:
   :show-inheritance:

.. autoclass:: fatqat.noise.ChannelImplementation
   :members:
   :show-inheritance:

.. autoclass:: fatqat.noise.ChannelImplementationMap
   :members:
   :show-inheritance:

.. autofunction:: fatqat.noise.default_channel_implementation_map

Capability reporting
--------------------

.. autoclass:: fatqat.noise.NoiseSupportReport
   :members:
   :show-inheritance:
