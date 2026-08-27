Measurement and structural operations
======================================

.. currentmodule:: fatqat.operations

Measurements
------------

Create measurements with :py:meth:`fatqat.Program.measure` or
:py:meth:`fatqat.Program.measure_all`. A grouped measurement pairs quantum
targets and classical outputs by tuple position, and each pair must have the
same local dimension. Measurements are separate instruction values rather than
``Operation`` subclasses, so they are not passed to ``Program.add`` and cannot
carry its ``condition=`` argument.

Repeated targets and outputs are accepted. Built-in backends process the pairs
in tuple order: a repeated target repeats its collapsed physical outcome, with
reporting noise resolved for each pair, and a repeated classical output keeps
the later pair's reported value.

.. autoclass:: fatqat.operations.Measurement
   :members:
   :no-inherited-members:

Reset and barrier
-----------------

.. list-table:: Common structural operations
   :header-rows: 1
   :widths: 15 18 27 20 20

   * - Value
     - Scalar targets
     - Effect
     - Conditions
     - Attached noise
   * - :py:data:`Reset`
     - One or more
     - Reprepares each present target in ``|0>``.
     - Supported through ``Program.add`` when the backend supports
       feedforward.
     - Not accepted.
   * - :py:data:`Barrier`
     - One or more
     - Preserved in the program and discarded by built-in simulators.
     - Accepted by the frontend, then discarded with the barrier by built-in
       lowering; it is never evaluated.
     - Not accepted.

Reset is non-unitary. A statevector backend samples the reset branch when the
target is entangled, while a density-matrix backend applies the deterministic
reset channel. Both values reject an empty target tuple, duplicate targets,
and ``RegisterView``.

.. autodata:: fatqat.operations.Reset
.. autodata:: fatqat.operations.Barrier

Atom lifecycle
--------------

:py:class:`~fatqat.simulator.AtomArraySimulator` is the only built-in backend
that implements the following values. Other matrix and pulse backends raise
:py:exc:`~fatqat.errors.UnsupportedOperationError`.

.. list-table:: Atom-array operations
   :header-rows: 1
   :widths: 14 17 30 19 20

   * - Value
     - Scalar targets
     - Effect
     - Conditions
     - Attached noise
   * - :py:data:`Put`
     - One or more
     - Loads ``|0>`` into each empty site; leaves occupied sites unchanged.
     - Supported.
     - ``Loss`` only, after each enabled occurrence.
   * - :py:data:`Pair`
     - Exactly two
     - Adds their undirected connectivity edge; repeated pairing is a no-op.
     - Rejected during lowering.
     - ``Loss`` or a supported finite channel.
   * - :py:data:`Unpair`
     - Exactly two
     - Removes their edge; removing an absent edge is a no-op.
     - Rejected during lowering.
     - ``Loss`` or a supported finite channel.

If a program contains ``Put``, every declared site starts empty for every shot.
Sites are populated only by their ``Put`` occurrences, and a later occurrence
can reload a lost atom. A ``Loss`` declaration attached to ``Put`` shares the
operation's condition and runs after every matching occurrence whose condition
passes, even when the site was already occupied and the ``Put`` itself did
nothing.

``Pair`` and ``Unpair`` change only the compiler-time connectivity graph; they
do not move state or add gate implementations. In the built-in atom-array
profile, CZ is native and requires a current pairing. A condition on either
connectivity instruction is accepted by the generic ``Program.add`` frontend
but raises :py:exc:`~fatqat.errors.BackendValidationError` when that backend
lowers the program.

.. autodata:: fatqat.operations.Put
.. autodata:: fatqat.operations.Pair
.. autodata:: fatqat.operations.Unpair
