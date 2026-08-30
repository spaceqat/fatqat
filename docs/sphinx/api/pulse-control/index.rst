Pulse control
=============

Read :doc:`../../guide/hamiltonian-emulation` first for the gate-calibration,
direct-control, and scheduling workflow. This section defines the exact pulse
objects and validation rules.

A direct pulse block is built from three values:

* :doc:`sampled-waveform` holds the samples for one signal.
* :doc:`pulse-control` assigns that signal to a model channel.
* :doc:`pulse-operation` groups one or more controls into a timed program
  instruction.

``PulseOperation`` is imported from ``fatqat.operations`` and is normally used
as ``ops.PulseOperation``. ``PulseControl`` and ``SampledWaveform`` are in
``fatqat.emulator``. Unlike an ordinary gate, a pulse operation is added
without targets because its channels already name the physical resources to
drive.

To implement an ordinary gate with pulses, register a
:py:class:`~fatqat.emulator.PulseImplementationMap` rule that returns a
:py:class:`~fatqat.emulator.PulseDefinition`. See :doc:`gate-realization`.

Validation
----------

FatQat checks sample structure and timing when you construct these values. It
rejects invalid offsets, non-positive block durations, repeated channels, and
controls that extend past the block. A model's channel factory checks its
address arguments when you create a channel.

When a control is used, the emulator checks model compatibility, resource
names, real or complex sample requirements, amplitude and duration limits,
and channel combinations that cannot run together.

Built-in support
----------------

Use the model's time unit for every duration, offset, and sample time. Sample
values use the channel's unit and may be restricted to real values.

.. list-table:: Direct pulse support
   :header-rows: 1
   :widths: 23 42 17 18

   * - Backend
     - Channels
     - Time / sample units
     - Conditions in ``run()``
   * - :py:class:`~fatqat.emulator.TransmonEmulator`
     - ``drive(id)`` accepts complex values; ``detuning(id)`` and
       ``exchange(first, second)`` require real values.
     - ``ns``, ``rad/ns``
     - Yes
   * - :py:class:`~fatqat.emulator.Atom2LevelEmulator`
     - Global ``drive()`` accepts complex values; global ``detuning()``
       requires real values.
     - ``us``, ``rad/us``
     - No
   * - :py:class:`~fatqat.simulator.Simulator`
     - Direct pulse operations are not supported.
     - --
     - --

The :doc:`transmon <../pulse-emulator>` and
:doc:`neutral-atom <../atom-emulators>` pages document model resources and
limits.

.. _pulse-probability-noise:

Continuous-time noise
---------------------

Pulse emulators evolve noise over time, so their family-owned Lindblad
realizations use rates or relaxation times. They do not infer a rate from a
finite probability.
In particular, :py:class:`~fatqat.noise.PauliChannel` remains a discrete
Simulator channel: its probabilities do not specify a duration or conversion
convention. Use
:py:class:`~fatqat.simulator.Simulator` for discrete Pauli noise, or a
rate-form declaration listed for the pulse-emulator family in
:ref:`noise-emulator-support`.

.. toctree::
   :maxdepth: 1

   pulse-operation
   pulse-control
   sampled-waveform
   gate-realization
