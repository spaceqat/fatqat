Registers
=========

Registers give quantum and classical program slots stable identities. With
``Program(quantum_count, classical_count)``, each positive count creates one
default register named ``"q"`` or ``"c"``; zero creates no register of that
kind. Construct registers explicitly when you need other names, multiple
registers, grids, metadata, or local dimensions greater than two.

Register types
--------------

.. list-table:: Register choices
   :header-rows: 1
   :widths: 24 48 28

   * - Type
     - Program role
     - Size
   * - :class:`~fatqat.Register`
     - Common base class; not itself accepted as a quantum or classical
       register by :class:`~fatqat.Program`
     - Explicit positive ``size``
   * - :class:`~fatqat.QuantumRegister`
     - Quantum operation and measurement targets
     - Explicit positive ``size``
   * - :class:`~fatqat.ClassicalRegister`
     - Measurement outputs and condition values
     - Explicit positive ``size``
   * - :class:`~fatqat.GridRegister`
     - Quantum targets with rectangular selection helpers
     - Derived as ``rows * cols``

Names are labels and need not be unique. Keep and index the same register
objects that you pass to :class:`~fatqat.Program`; a newly constructed register
with the same fields is not interchangeable. ``metadata`` is a mutable,
string-keyed mapping for application data.

Indexing with ``register[index]`` creates an immutable
:class:`~fatqat.RegisterRef`. In a program with multiple registers, pass the
explicit ref rather than an ambiguous integer:

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as ops

   left = fq.QuantumRegister(2, name="left")
   right = fq.QuantumRegister(2, name="right")
   program = fq.Program([left, right])
   program.add(ops.H, right[0])

``dim=2`` creates qubits or classical bits. A larger value creates qudits or
d-ary classical digits; the quantum and classical dimensions of each
measurement pair must match. Register construction accepts every integer
dimension of at least two, but individual operations and backends may support
only some dimensions. See :doc:`../guide/program` for a mixed qubit-qutrit
example.

.. autoclass:: fatqat.Register
   :members:
   :special-members: __getitem__
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

Grid selections
---------------

A :class:`~fatqat.GridRegister` arranges logical targets in row-major order; it
does not assign physical coordinates. The flat index of ``(row, col)`` is
``row * cols + col``, and its selection helpers return
:class:`~fatqat.RegisterView` objects rather than tuples of refs.

For a ``GridRegister(2, 3)``, the helpers select these flat indices:

.. list-table:: Grid selection order
   :header-rows: 1
   :widths: 38 62

   * - Expression
     - Selected indices, in order
   * - ``grid.all()``
     - ``(0, 1, 2, 3, 4, 5)``
   * - ``grid.row(1)``
     - ``(3, 4, 5)``
   * - ``grid.column(1)``
     - ``(1, 4)``
   * - ``grid.block((0, 2), (1, 3))``
     - ``(1, 2, 4, 5)``

Pass views to :meth:`~fatqat.Program.add`. Every built-in unitary gate accepts
them. Unary gates act independently on each selected member; multi-target
gates zip corresponding members from one view per operand. All views must use
the same kind of grid selection and cardinality, and selections on the same
grid cannot overlap. Measurements and QASM export require scalar targets. The
backend validates physical placement and connectivity. See
:doc:`../guide/program` for the ordinary Program workflow and
:doc:`../guide/hardware-profile-simulation` for physical placement.

.. autoclass:: fatqat.GridRegister
   :members:
   :show-inheritance:

.. py:class:: fatqat.RegisterView
   :canonical: fatqat.registers.RegisterView

   Immutable, hashable target returned by the grid selection helpers. Its
   :py:attr:`~fatqat.RegisterView.register` attribute identifies the selected
   grid. Obtain views from the grid helpers; direct construction is
   unsupported.

   .. py:attribute:: register
      :canonical: fatqat.registers.RegisterView.register
      :type: fatqat.GridRegister

      Grid register containing the selected members.

Resource layouts
----------------

A :class:`~fatqat.ResourceLayout` associates scalar quantum
:class:`~fatqat.RegisterRef` operands with device labels. Most applications can
use the backend's default layout; pass ``resource_layout=`` when you need a
specific supported placement. Each backend defines the labels it accepts and
checks coverage, uniqueness, dimensions, placement, and connectivity when the
program runs. :class:`~fatqat.operations.PulseOperation` channels address the
emulator model directly and do not use this layout.

.. autoclass:: fatqat.ResourceLayout
   :members:
   :show-inheritance:

.. py:type:: fatqat.DeviceOperand

   Backend-defined opaque, hashable label for one device resource.
