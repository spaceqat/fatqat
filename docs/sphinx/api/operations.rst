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

Names exported through ``fatqat.operations.__all__`` are the supported public
surface. Fixed gates and parameter-free structural operations are immutable
singleton values and must not be called. Parameterized gates and
``PulseOperation`` are classes that construct values with frozen fields; their
constructor arguments and any retained-container caveats are shown on the
family pages. ``Measurement`` records are created by ``Program`` rather than
added directly. ``Program.add`` retains an operation value rather than copying
it, so values built with the documented immutable inputs can safely be reused
across instructions and programs.

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
     - Channel-addressed ``PulseOperation`` values and the boundary between
       construction and model-owned validation and binding.

.. toctree::
   :maxdepth: 1

   operations/qubit-gates
   operations/qudit-gates
   operations/structural
   operations/atom-gates
   operations/direct-control

Construction
------------

For ordinary operations, ``Program.add`` checks that an operation value was
supplied, resolves target references, enforces target count, and rejects
repeated scalar targets. For a scalar instruction it also calls the
operation's target validator. For a view instruction, it checks the
grouped-view shape immediately; each expanded scalar member or pair reaches
the target validator during built-in backend preparation. A direct
``PulseOperation`` instead uses the channel-addressed contract on
:doc:`operations/direct-control`. The frontend does not decide whether the
selected backend implements an operation. An unsupported family raises
:py:exc:`~fatqat.errors.UnsupportedOperationError` when the backend prepares
the program.

Most operations require scalar exact built-in ``int`` or
:py:class:`~fatqat.RegisterRef` targets. A bare integer is valid only when the
program has exactly one quantum register; booleans, NumPy integers, and integer
subclasses are rejected. Controlled gates use control-first order, and the
first local operand is the most-significant digit in the matrices and basis
actions on the family pages.

The built-in :py:class:`RX`, :py:class:`RY`, and :py:class:`RZ` operations and
:py:data:`CX` and :py:data:`CZ` values accept
:py:class:`~fatqat.RegisterView` targets:

* A rotation on one view expands to one operation per selected member.
* ``CX`` or ``CZ`` on two views pairs their members in view order.
* A two-view pair must use the same selector kind and equal cardinality. Two
  views of the same register must not overlap. Mixing a scalar with a view is
  invalid.

Those view constraints are checked when the instruction is added. Device
topology and operation support remain backend checks. A custom ``Operation``
subclass can override ``accepts_views`` to opt into the shared unary or
two-target expansion path. See :doc:`../guide/gates` for the ordinary
construction workflow and :doc:`registers` for target and view types.

Operation base
--------------

Subclassing ``Operation`` defines a new frontend value; it does not register a
matrix or pulse realization. See :doc:`../guide/advanced` for the custom matrix
workflow.

.. autoclass:: fatqat.operations.Operation
   :members: name, num_subsystems, min_targets, accepts_views, validate_targets
   :show-inheritance:
