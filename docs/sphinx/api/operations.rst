Operations
==========

.. currentmodule:: fatqat.operations

Import ``fatqat.operations`` as ``ops``, then add operations to a program with
:py:meth:`fatqat.Program.add`:

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as ops

   program = fq.Program(2)
   program.add(ops.H, 0)            # ready-to-use operation
   program.add(ops.RX(0.2), 1)      # parameterized operation
   program.add(ops.CX, (0, 1))      # ordered targets

Parameter-free operations such as ``ops.H`` and ``ops.Reset`` are ready to use
without parentheses. Construct parameterized gates and
:py:class:`PulseOperation` values before adding them. Create measurements with
:py:meth:`~fatqat.Program.measure` or :py:meth:`~fatqat.Program.measure_all`.

Reference pages
---------------

.. list-table:: Operation families
   :header-rows: 1
   :widths: 28 72

   * - Page
     - Contents
   * - :doc:`operations/qubit-gates`
     - Fixed and parameterized qubit gates, exact target order, matrices, and
       constructor reference.
   * - :doc:`operations/qudit-gates`
     - Qudit gates, level constraints, and basis actions.
   * - :doc:`operations/structural`
     - Measurement and reset behavior, and compiler barriers.
   * - :doc:`operations/atom-gates`
     - Atom-array occupancy, pairing, and attached-noise constraints.
   * - :doc:`pulse-control/pulse-operation`
     - Channel-addressed ``PulseOperation``—still imported from
       ``fatqat.operations``—with its timing and backend support.

.. toctree::
   :maxdepth: 1

   operations/qubit-gates
   operations/qudit-gates
   operations/structural
   operations/atom-gates

Construction
------------

For target-based operations, :py:meth:`~fatqat.Program.add` resolves target
references, checks arity, and rejects repeated scalar targets. The selected
backend checks operation and device support when you submit the program; an
unsupported family raises
:py:exc:`~fatqat.errors.UnsupportedOperationError`. A direct
:py:class:`PulseOperation` follows the channel-addressing rules on
:doc:`pulse-control/pulse-operation` and is added without targets.

Most targets are a scalar :py:class:`~fatqat.RegisterRef` or an integer. Use an
integer when the program has one quantum register; with multiple registers,
index the register you want. Controlled gates use control-first order, and the
first local operand is the most-significant digit in the matrices and basis
actions on the family pages.

:py:class:`RX`, :py:class:`RY`, and :py:class:`RZ` accept one
:py:class:`~fatqat.RegisterView`; :py:data:`CX` and :py:data:`CZ` accept two
compatible views and pair their members in order. See :doc:`registers` for the
view compatibility rules and :doc:`../guide/program` for the ordinary
construction workflow.

Operation base
--------------

Subclassing :py:class:`Operation` defines a new program-level value; it does
not register a matrix or pulse realization. See :doc:`implementation` for the
custom matrix contract and :doc:`pulse-control/gate-realization` for pulse
realizations.

.. autoclass:: fatqat.operations.Operation
   :members: name, num_subsystems, min_targets, accepts_views, validate_targets
   :show-inheritance:
