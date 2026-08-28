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

Register fields are frozen and use object identity for equality and hashing.
Names are display labels, need not be unique, and do not make two separately
constructed registers interchangeable. The stored ``metadata`` dictionary
defaults to empty and is the deliberate mutability exception: FATQAT reserves
no keys and shallow-copies its string-keyed entries, so later top-level changes
to the input are not observed while nested values remain shared. A
:class:`~fatqat.Program` given an explicit register list or tuple retains those
register objects while copying the outer collection.

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
only some dimensions. See :doc:`../guide/advanced` for a qutrit example.

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

A :class:`~fatqat.GridRegister` is a row-major, backend-neutral quantum
register. Its shape describes program targets, not physical coordinates or a
placement guarantee. Flat indexing uses ``row * cols + col``. The selection
helpers return a :class:`~fatqat.RegisterView`, not a tuple of refs.

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

Pass views to :meth:`~fatqat.Program.add`. The built-in view-capable operations
are :class:`~fatqat.operations.RX`, :class:`~fatqat.operations.RY`,
:class:`~fatqat.operations.RZ`, :data:`~fatqat.operations.CX`, and
:data:`~fatqat.operations.CZ`. A unary operation is applied independently to
every selected member. For :data:`~fatqat.operations.CX` or
:data:`~fatqat.operations.CZ`, FATQAT pairs the first and second views in their
documented order. Both views must use the same selector kind and cardinality.
Views over one register must not overlap; views over different grid registers
still require equal cardinality.

Views are operation target expressions. Measurements require scalar targets,
and QASM export does not currently support a program containing a view.
Device-specific placement and connectivity are validated by the selected
backend. See :doc:`../guide/gates` for a paired-row example.

.. autoclass:: fatqat.GridRegister
   :members:
   :show-inheritance:

.. py:class:: fatqat.RegisterView
   :canonical: fatqat.registers.RegisterView

   Immutable, hashable target value returned by the grid selection helpers
   above. Its :py:attr:`~fatqat.RegisterView.register` attribute identifies the
   owning grid, and equality combines that register's identity with the
   selection. Construct views with the grid helpers; the selector
   representation is not a public construction contract.

   .. py:attribute:: register
      :canonical: fatqat.registers.RegisterView.register
      :type: fatqat.GridRegister

      Grid register that owns the selected members.

Resource layouts
----------------

A :class:`~fatqat.ResourceLayout` is a read-only lookup from scalar quantum
refs to opaque, hashable :obj:`~fatqat.DeviceOperand` labels. Most applications
should let the backend create its default layout. Supply one through
``resource_layout=`` only to request a placement supported by that backend.
Labels identify public device resources; they are not private simulator
tensor-axis indices.

A layout maps the logical :class:`~fatqat.RegisterRef` operands of ordinary
operations. Direct :class:`~fatqat.operations.PulseOperation` channels bind
against the emulator's physical model and are not remapped by the layout.
Backend calls do not mutate a layout, so it can be reused with the same
register objects and a compatible backend. Use immutable labels whose equality
and hashes remain stable for that lifetime.

The constructor and backend perform different validation:

.. list-table:: Layout validation
   :header-rows: 1
   :widths: 25 75

   * - Stage
     - Contract
   * - Construction
     - Shallow-copies the mapping and requires hashable labels.
   * - Backend use
     - Validates complete program coverage, distinct labels, accepted label
       types, subsystem dimensions, placement, and connectivity as required by
       the selected backend.

.. autoclass:: fatqat.ResourceLayout
   :members:
   :show-inheritance:

.. py:type:: fatqat.DeviceOperand

   Backend-defined opaque, hashable label for one device resource.
