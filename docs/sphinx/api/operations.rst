Operations
==========

.. currentmodule:: fatqat.operations

Import this namespace as ``ops`` and append operation values with
:py:meth:`fatqat.Program.add`:

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as ops

   program = fq.Program(2)
   program.add(ops.H, 0)            # parameter-free singleton
   program.add(ops.RX(0.2), 1)      # constructed parameterized value
   program.add(ops.CX, (0, 1))      # ordered multi-target value

Fixed gates and parameter-free structural operations are immutable singleton
values and must not be called. Parameterized gates and
:py:class:`PulseOperation` are classes that construct immutable values. These
values can be reused across instructions and programs. Create measurements
through :py:class:`~fatqat.Program` rather than adding them directly.

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
     - Dimension-derived gates, level constraints, and basis actions.
   * - :doc:`operations/structural`
     - Measurement and reset state transitions, and compiler barrier
       semantics.
   * - :doc:`operations/atom-gates`
     - Neutral-atom occupancy, pairing, and attached-noise constraints.
   * - :doc:`operations/direct-control`
     - Channel-addressed :py:class:`PulseOperation` values, validation, and
       model binding.

.. toctree::
   :maxdepth: 1

   operations/qubit-gates
   operations/qudit-gates
   operations/structural
   operations/atom-gates
   operations/direct-control

Construction
------------

For ordinary operations, :py:meth:`~fatqat.Program.add` resolves target
references, enforces target count, and rejects repeated scalar targets. It
does not decide whether the selected backend implements an operation or
whether a device supports the requested targets. An unsupported family raises
:py:exc:`~fatqat.errors.UnsupportedOperationError` when the backend prepares
the program. A direct :py:class:`PulseOperation` instead follows the
channel-addressed contract on :doc:`operations/direct-control`.

Most operations require scalar exact built-in ``int`` or
:py:class:`~fatqat.RegisterRef` targets. A bare integer is valid only when the
program has exactly one quantum register; booleans, NumPy integers, and integer
subclasses are rejected. Controlled gates use control-first order, and the
first local operand is the most-significant digit in the matrices and basis
actions on the family pages.

:py:class:`RX`, :py:class:`RY`, and :py:class:`RZ` accept one
:py:class:`~fatqat.RegisterView`; :py:data:`CX` and :py:data:`CZ` accept two
compatible views and pair their members in order. See :doc:`registers` for the
view compatibility rules and :doc:`../guide/gates` for the ordinary
construction workflow.

Operation base
--------------

Subclassing :py:class:`Operation` defines a new program-level value; it does
not register a matrix or pulse realization. See :doc:`../guide/advanced` for
the custom matrix workflow.

.. autoclass:: fatqat.operations.Operation
   :members: name, num_subsystems, min_targets, accepts_views, validate_targets
   :show-inheritance:
