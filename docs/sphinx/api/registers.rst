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

Register fields are frozen and use object identity for equality and hashing,
with one deliberate shallow-immutability exception: the stored ``metadata``
dictionary remains mutable. Names are display labels, need not be unique, and
do not make two separately constructed registers interchangeable. Use ``str``
or ``None`` for a name; its type is not checked at runtime. A
:class:`~fatqat.Program` given an explicit register list or tuple retains those
register objects while copying the outer collection into a tuple.

Indexing with ``register[index]`` creates an immutable
:class:`~fatqat.RegisterRef`. Indices are zero-based; negative indices and
``bool`` values are rejected. In a program with multiple registers, pass the
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

Each register's ``metadata`` argument defaults to an empty mapping. Its
supported keys are strings, its values may have any type, and FATQAT assigns
no predefined meaning or per-key default. Construction shallow-copies the
top-level entries with ``dict()``; nested values remain shared. Runtime
conversion also retains non-string keys and accepts dictionary-compatible pair
iterables. An invalid input raises ``TypeError`` or ``ValueError``.

.. autoclass:: fatqat.Register
   :members:
   :special-members: __getitem__
   :show-inheritance:

.. autoclass:: fatqat.QuantumRegister
   :members:
   :special-members: __getitem__
   :show-inheritance:

.. autoclass:: fatqat.ClassicalRegister
   :members:
   :special-members: __getitem__
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
are ``RX``, ``RY``, ``RZ``, ``CX``, and ``CZ``; a custom ``Operation`` subclass
can also opt in. A unary operation is applied independently to every selected
member. For ``CX`` or ``CZ``, FATQAT zips the first and second views in their
documented order. Both targets must be views of the same selector kind and
cardinality. Two views over the same register must not overlap; views over
different grid registers still require equal cardinality.

Views are only operation target expressions. Measurements accept bare integers
or scalar quantum refs, never views. Noise selectors accept scalar quantum refs
or physical device labels, never views. QASM export does not currently support
a program that contains a view. Device-specific placement and connectivity are
validated by the selected backend. See :doc:`../guide/gates` for a paired-row
example.

.. autoclass:: fatqat.GridRegister
   :members:
   :special-members: __getitem__
   :show-inheritance:

.. autoclass:: fatqat.RegisterView
   :members:
   :show-inheritance:

Resource layouts
----------------

A :class:`~fatqat.ResourceLayout` is a read-only lookup object from scalar
quantum refs to opaque, hashable :obj:`~fatqat.DeviceOperand` labels for a
program-to-device binding. It does not implement the general ``Mapping``
interface and layout objects compare by identity. Most applications should let
the backend create its default layout. Supply one through ``resource_layout=``
only to request a specific placement supported by that backend. Labels
identify public device resources; they are not private simulator tensor-axis
indices. A layout maps the logical :class:`~fatqat.RegisterRef` operands of
ordinary operations. Direct :class:`~fatqat.operations.PulseOperation`
channels bind against the emulator's physical model and are not remapped by
the layout. Backend calls do not mutate a layout, so it can be reused with the
same register objects and a compatible backend. Label equality and hashes
must remain stable for that lifetime; immutable labels are strongly
preferred.

The constructor and backend perform different validation:

.. list-table:: Layout validation
   :header-rows: 1
   :widths: 25 75

   * - Stage
     - Contract
   * - Construction
     - Shallow-copies with ``dict()`` and requires hashable labels. The
       supported keys are scalar quantum refs, but key type and ownership are
       not checked; partial coverage, foreign keys, and repeated labels remain.
   * - Backend use
     - Current backends require complete program coverage and distinct,
       exclusive labels. Label type, accepted program dimensions, placement,
       and connectivity are backend-specific checks that can occur during
       binding, preparation, or lowering.

The supported ``labels`` argument is a mapping whose keys are quantum
``RegisterRef`` objects and whose values are hashable ``DeviceOperand`` values;
there are no universal label choices or defaults. Runtime construction accepts
any ``dict()``-compatible iterable and does not validate key types. Invalid
dictionary input raises ``TypeError`` or ``ValueError``. The input mapping can
be changed after construction without affecting the layout. Ref keys and label
objects themselves are not copied. Refs use their register's identity, so a ref
from a separately reconstructed lookalike register is absent. The public lookup
direction is ref to label:
:meth:`~fatqat.ResourceLayout.device_label` looks up one ref and
:meth:`~fatqat.ResourceLayout.device_labels_for` preserves the order of a ref
tuple. The :attr:`~fatqat.ResourceLayout.refs` and
:attr:`~fatqat.ResourceLayout.device_labels` properties return immutable
sets; repeated labels therefore collapse in ``device_labels`` even though the
constructor retains their per-ref mappings.

.. autoclass:: fatqat.ResourceLayout
   :members:
   :show-inheritance:

.. autodata:: fatqat.DeviceOperand
