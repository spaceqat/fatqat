Registers
=========

Registers name the program qubits and classical slots used by a
:py:class:`~fatqat.Program`. The simple ``Program(quantum_count, classical_count)`` form is
enough for most programs.

Quantum and classical registers
--------------------------------

- :py:class:`~fatqat.QuantumRegister` (``size, name=None, dim=2``) creates quantum slots.
- :py:class:`~fatqat.ClassicalRegister` (``size, name=None, dim=2``) creates classical slots.
- ``dim=2`` means qubits or classical bits. Use a larger dimension for
  qudits and their matching classical digits.

Index a register to obtain a :py:class:`~fatqat.RegisterRef`. References make targets
unambiguous in programs with several registers:

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as ops

   left = fq.QuantumRegister(2, name="left")
   right = fq.QuantumRegister(2, name="right")
   program = fq.Program([left, right])
   program.add(ops.H, program.quantum_registers[1][0])

Grid registers
--------------

:py:class:`~fatqat.GridRegister` (``rows, cols, name=None, dim=2``) is an optional abstract
rectangular quantum register. Its helpers return targets for the supported
grid-aware gates:

- :py:meth:`~fatqat.GridRegister.all` selects every member in row-major order.
- :py:meth:`~fatqat.GridRegister.row` selects one row.
- :py:meth:`~fatqat.GridRegister.column` selects one column.
- :py:meth:`~fatqat.GridRegister.block` selects a half-open rectangular block, for
  example ``grid.block((0, 2), (1, 3))``.

Use the returned value directly in :py:meth:`~fatqat.Program.add`; applications
do not construct a register view themselves. See :doc:`../guide/gates` for a
paired-row example.

Resource layouts
----------------

A :py:class:`~fatqat.ResourceLayout` optionally maps every program quantum
reference to an opaque device operand for one run. Most applications use the
backend's default mapping. Construct an explicit layout only when choosing a
specific placement supported by that backend, then pass it through
``run(..., resource_layout=layout)``.

Detailed reference
------------------

.. autoclass:: fatqat.Register
   :members:
   :show-inheritance:

.. autoclass:: fatqat.QuantumRegister
   :members:
   :show-inheritance:

.. autoclass:: fatqat.ClassicalRegister
   :members:
   :show-inheritance:

.. autoclass:: fatqat.RegisterRef
   :members:
   :show-inheritance:

.. autoclass:: fatqat.GridRegister
   :members:
   :show-inheritance:

.. autoclass:: fatqat.RegisterView
   :members:
   :show-inheritance:

.. autoclass:: fatqat.ResourceLayout
   :members:
   :show-inheritance:
