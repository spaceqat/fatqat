SampledWaveform
===============

.. currentmodule:: fatqat.emulator

:py:class:`SampledWaveform` describes a signal on a local time grid. Times use
the model's time unit, while the channel determines the sample unit and whether
values may be complex.

Interpolation
-------------

Built-in pulse emulators use not-a-knot spline interpolation. Two samples
give a linear curve, three give a quadratic curve, and four or more give a
cubic curve.

``Atom2LevelEmulator`` uses zero outside the sample interval.
``TransmonEmulator`` and ``Atom3LevelEmulator`` hold the nearest endpoint, so
use zero first and last samples when a control should be off outside its grid.
This behavior does not change the operation's duration.

A spline can exceed the supplied sample values between points. If a model has
amplitude limits, the emulator checks the interpolated curve as well as the
samples.

Reference
---------

.. autoclass:: fatqat.emulator.SampledWaveform
   :members: duration
   :no-inherited-members:
