Program
=======

:py:class:`~fatqat.Program` is the user-facing container for a quantum program. Build a
program, add operations and measurements in order, then pass it to a
backend.

Operation examples use ``import fatqat.operations as op``.

Create a program
----------------

:py:class:`~fatqat.Program` (``quantum_registers, classical_registers=0``)

- Pass integers for the common case: ``fq.Program(2, 2)`` creates two
  quantum slots and two classical slots.
- Pass lists of :py:class:`~fatqat.QuantumRegister` and :py:class:`~fatqat.ClassicalRegister` objects when
  you need named, multiple, grid, or higher-dimensional registers.

Add an operation
----------------

:py:meth:`~fatqat.Program.add` (``op, targets, *, condition=None``)

- ``op`` is a value or constructed gate from ``fatqat.operations``.
- ``targets`` is one integer or register reference for a one-target gate,
  or a tuple such as ``(0, 1)`` for a multi-target gate.
- ``condition=(clbit, value)`` applies an operation only when a previous
  measurement wrote the requested classical value.

Fixed gates are values: ``program.add(op.H, 0)``. Parametric gates are
constructed first: ``program.add(op.RX(0.2), 0)``.

Add a measurement
-----------------

:py:meth:`~fatqat.Program.measure` (``targets, outputs``)

Measure one quantum target into one classical output, or use matching
tuples for grouped measurement:

.. code-block:: python

   program.measure(0, 0)
   program.measure((0, 1), (0, 1))

:py:meth:`~fatqat.Program.measure_all` measures every quantum slot into every classical
slot in declaration order. It requires matching quantum and classical slot
counts.

Draw a circuit
--------------

:py:meth:`~fatqat.Program.draw` renders the program as a matplotlib figure by
default. Pass ``"text"`` to return a terminal diagram string instead.

:py:meth:`~fatqat.Program.copy` returns an independent copy when you want to branch a
program before adding more instructions.
:py:meth:`~fatqat.Program.assign_parameters` returns a new program with the
selected identity-based parameter objects replaced. See
:doc:`../guide/parameters-and-sweeps` for partial binding and vector examples.
For examples, see :doc:`../guide/concepts` and
:doc:`../guide/measurement-and-conditions`.

Detailed reference
------------------

.. autoclass:: fatqat.Program
   :members: operations, add, measure, measure_all, draw, copy, assign_parameters
   :show-inheritance:

.. autoclass:: fatqat.Measurement
   :members:
   :show-inheritance:

.. autoclass:: fatqat.Parameter
   :members:
   :show-inheritance:

.. autoclass:: fatqat.ParameterVector
   :members:
   :show-inheritance:
