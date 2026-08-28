Measurement and structural operations
======================================

.. currentmodule:: fatqat.operations

Measurement and reset
---------------------

Measurement and reset are state-changing execution boundaries. Measurement
also writes a reported value to classical storage, while reset reprepares its
targets without producing a result.

Create measurements with :py:meth:`fatqat.Program.measure` or
:py:meth:`fatqat.Program.measure_all`. A grouped measurement pairs quantum
targets and classical outputs by tuple position, and each pair must have the
same local dimension. Measurements are separate instruction values rather than
:py:class:`Operation` subclasses, so they are not passed to
:py:meth:`~fatqat.Program.add` and cannot carry its ``condition=`` argument.

Repeated targets and outputs are accepted. Built-in backends process the pairs
in tuple order: a repeated target repeats its collapsed physical outcome, with
reporting noise resolved for each pair, and a repeated classical output keeps
the later pair's reported value. :py:class:`~fatqat.noise.ReadoutConfusion`
can change a reported digit without changing that collapsed physical outcome.

.. autoclass:: fatqat.operations.Measurement
   :members:
   :no-inherited-members:

Add :py:data:`Reset` with :py:meth:`fatqat.Program.add`. It accepts one or more
distinct scalar targets and can carry a condition when the backend supports
feedforward. It rejects an empty target tuple, duplicate targets, and
:py:class:`~fatqat.RegisterView`. Reset is non-unitary: a statevector backend
samples the reset branch when a target is entangled, while a density-matrix
backend applies the deterministic reset channel. It has no attached-noise
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

Built-in simulators treat a barrier as a no-op, so it does not change states or
counts. :py:meth:`~fatqat.Program.add` accepts a condition, but built-in
simulators do not evaluate it. A barrier cannot be bound to noise: using it as
the ``operation=`` selector in :py:meth:`fatqat.NoiseModel.add` raises
:py:exc:`ValueError`.

.. autodata:: fatqat.operations.Barrier
   :no-value:
