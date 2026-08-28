Measurement and structural operations
======================================

.. currentmodule:: fatqat.operations

Measurement and reset
---------------------

Measurement writes computational-basis outcomes to classical storage. Reset
returns its targets to ``|0>`` without producing an output.

Create measurements with :py:meth:`fatqat.Program.measure` or
:py:meth:`fatqat.Program.measure_all`. A grouped measurement pairs quantum
targets and classical outputs by tuple position, and each pair must have the
same local dimension. Measurements are created through these methods, not
:py:meth:`~fatqat.Program.add`, and cannot carry its ``condition=`` argument.

Repeated targets and outputs are accepted, and pairs are processed in tuple
order. Repeating a target reports its already-collapsed outcome, with reporting
noise applied independently to each pair. Repeating a classical output means
the last write wins. :py:class:`~fatqat.noise.ReadoutConfusion` changes only
the reported digit, not the collapsed physical outcome.

.. autoclass:: fatqat.operations.Measurement
   :members:
   :no-inherited-members:

Add :py:data:`Reset` with :py:meth:`fatqat.Program.add`. It accepts one or more
distinct scalar targets and can carry a condition when the backend supports
feedforward. It rejects an empty target tuple, duplicate targets, and
:py:class:`~fatqat.RegisterView`. Reset is non-unitary. For an entangled
target, a statevector run samples one reset branch, while a density-matrix run
represents the resulting mixture directly. It has no attached-noise
realization, so using it as the ``operation=`` selector in
:py:meth:`fatqat.NoiseModel.add` raises :py:exc:`ValueError`.

.. autodata:: fatqat.operations.Reset
   :no-value:

Compiler barrier
----------------

:py:data:`Barrier` is a compiler and scheduling marker, not a state-changing
operation or noise boundary. Add it with :py:meth:`fatqat.Program.add` and one
or more distinct scalar targets. It rejects an empty target tuple, duplicate
targets, and :py:class:`~fatqat.RegisterView`.

Built-in simulators ignore barriers, including any condition recorded by
:py:meth:`~fatqat.Program.add`, so barriers do not change states or counts. A
barrier cannot be bound to noise: using it as the ``operation=`` selector in
:py:meth:`fatqat.NoiseModel.add` raises :py:exc:`ValueError`.

.. autodata:: fatqat.operations.Barrier
   :no-value:
