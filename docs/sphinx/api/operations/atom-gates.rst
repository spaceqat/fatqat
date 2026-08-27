Atom gates
==========

.. currentmodule:: fatqat.operations

These atom-array lifecycle operations manage site occupancy and connectivity;
they are not unitary matrix gates. Add them with
:py:meth:`fatqat.Program.add`. :py:class:`~fatqat.simulator.AtomArraySimulator`
is the only built-in backend that implements them. Other matrix and pulse
backends raise :py:exc:`~fatqat.errors.UnsupportedOperationError`.

.. list-table:: Atom gates
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
