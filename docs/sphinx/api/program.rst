Program
=======

:py:class:`~fatqat.Program` stores registers and an ordered sequence of
operations and measurements. Build the complete instruction sequence, then
pass the program to a backend. Program construction checks register ownership,
target shape, measurement pairing, and conditions; backend-specific operation
and device support is checked when the program is run.

Operation examples use ``import fatqat.operations as ops``.

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as ops

   bell = fq.Program(2, 2, metadata={"name": "bell"})
   bell.add(ops.H, 0)
   bell.add(ops.CX, (0, 1))
   bell.measure_all()

Registers
---------

The two register arguments accept these forms:

.. list-table:: Register inputs
   :header-rows: 1
   :widths: 24 36 40

   * - Form
     - Result
     - Constraints
   * - Non-negative integer ``n``
     - ``n > 0`` creates one dimension-2 register named ``"q"`` or ``"c"``;
       ``0`` creates no register.
     - The value must be an exact integer, not a boolean. A negative value is
       rejected.
   * - List or tuple of registers
     - Preserves the supplied register objects in a tuple.
     - Every item must have the matching quantum or classical register kind.
       Use this form for names, multiple registers, grids, or qudits.

The outer collection is copied; its register objects are not. Treat the
register tuple attributes as construction-time state: rebinding them is
unsupported.

``metadata`` is ``None`` (the default) or a mapping whose supported key type is
``str`` and whose values may have any type. There are no predefined keys,
per-key defaults, or FATQAT-defined effects: entries are retained only for the
application. The top-level entries are copied into the program's mutable
dictionary, while nested values remain shared. Runtime conversion does not
validate key types, and an invalid truthy dictionary input can raise
``TypeError`` or ``ValueError``.

Targets
-------

A bare integer is an index into the sole register of the relevant kind. It
must be an exact built-in ``int``; booleans, NumPy integers, and integer
subclasses are rejected. It is not a global index across several registers.
With multiple registers, pass an explicit :py:class:`~fatqat.RegisterRef`
from the program instead. With no register of that kind, no valid target
exists. References are checked by register identity, so a reference from a
separately constructed register is rejected even if its fields are identical.

.. list-table:: Target forms
   :header-rows: 1
   :widths: 20 35 45

   * - Form
     - Accepted by
     - Rule
   * - Integer
     - ``add()``, ``measure()``, and conditions
     - Requires exactly one register of the relevant kind and must be within
       its zero-based bounds.
   * - :py:class:`~fatqat.RegisterRef`
     - ``add()``, ``measure()``, and conditions
     - Must have the required register kind and belong to this program.
   * - :py:class:`~fatqat.RegisterView`
     - ``add()`` only
     - ``RX``, ``RY``, and ``RZ`` accept one view. ``CX`` and ``CZ`` accept two
       compatible views. Measurement does not accept views.

A :py:class:`~fatqat.operations.PulseOperation` uses a direct-control contract
instead of the target forms above. It does not take a separate ``targets``
argument. Add it with ``program.add(operation)``; each contained
:py:attr:`~fatqat.emulator.PulseControl.channel` identifies the physical
resource or resources that the selected emulator resolves during program
preparation. See :doc:`operations/direct-control` for binding and validation.

A tuple supplies operation operands in order; it is not variadic call syntax.
For controlled gates, controls precede targets. Paired views are applied
positionally and must use the same selector kind, have equal cardinality, and
not overlap when they refer to one grid. See :doc:`registers` for selector
ordering and :doc:`../guide/gates` for gate target order.

Conditions
----------

Pass ``condition=(slot, literal)`` to :py:meth:`~fatqat.Program.add`, or pass a
non-empty tuple or list of such pairs for logical AND. A bare slot uses the
same exact-built-in-``int`` rule as a target. Each literal is a Python ``int``
in ``0 <= literal < slot.dim``; booleans and integer subclasses are accepted
and normalized with ``int()``, while NumPy integer scalars are rejected. On an
execution backend that supports conditions, every term is compared with the
current classical value when the operation is reached.
Classical storage starts at zero and an earlier measurement can replace a
slot's value.

Construction validates the condition shape, slot ownership, and literal
range. It does not require the slot to have been measured and does not promise
that every backend or execution method supports feedforward.

Measurements
------------

:py:meth:`~fatqat.Program.measure` pairs quantum targets with classical
outputs positionally. Both sides must be non-empty, have the same length, and
have matching dimensions at every position. One grouped call appends one
grouped :py:class:`~fatqat.operations.Measurement` to the program.

Repeated targets and outputs are accepted. Built-in backends process the pairs
in order: a repeated target repeats its collapsed physical outcome, with
reporting noise resolved for each pair, and a repeated classical output keeps
the later pair's reported value.

:py:meth:`~fatqat.Program.measure_all` flattens all registers and their members
in declaration order and appends one grouped measurement. It requires equal,
non-zero quantum and classical counts and matching dimensions at every
position. Read :doc:`../guide/measurement-and-conditions` for mid-program
measurement and feedforward workflows.

Templates
---------

Parameters are immutable identity objects. Names are labels only: two
``Parameter("theta")`` objects are different binding keys. Reuse one object
when several gate fields should share a value.

.. list-table:: Binding forms
   :header-rows: 1
   :widths: 28 34 38

   * - Mapping key
     - Accepted value
     - Constraint
   * - :py:class:`~fatqat.Parameter`
     - Built-in integer or float, or NumPy integer or floating scalar
     - The same object must occur directly in an operation field.
   * - :py:class:`~fatqat.ParameterVector`
     - One-dimensional NumPy array, or a non-string, non-bytes, non-mapping
       iterable of accepted scalars
     - Consumed once in iteration order. The value length must match and every
       vector element must occur directly in an operation field. Bind
       individual elements for a partial vector.

:py:meth:`~fatqat.Program.assign_parameters` may bind any subset and always
returns a new program. Remaining parameters stay symbolic, but numeric
execution and export reject them. String keys, booleans, complex values, and
duplicate vector/element assignments are rejected. Parameter discovery is
structural and does not widen an operation field's declared type contract; an
invalid bound value can still fail during reconstruction or backend lowering.
Read
:doc:`../guide/parameters-and-sweeps` for sweep shapes and execution behavior.

:py:meth:`~fatqat.Program.copy` also returns a new mutable branch. Both methods
share register objects and reuse unchanged operation values from the source,
while owning independent internal instruction storage and a copied top-level
metadata dictionary. Nested metadata values remain shared. Calls to ``add()``,
``measure()``, and ``measure_all()`` instead mutate the current program and
return ``None``.

Draw
----

.. list-table:: Renderers
   :header-rows: 1
   :widths: 25 35 40

   * - ``renderer``
     - Return value
     - Notes
   * - ``"matplotlib"`` (default)
     - Matplotlib ``Figure``
     - Pass ``ax=`` to draw on an existing axis; other keyword arguments are
       forwarded to the renderer.
   * - ``"text"``
     - Terminal diagram string
     - The string is returned, not printed.
   * - Another QuTiP-QIP renderer name
     - Renderer-defined
     - The name and keyword arguments are forwarded unchanged.

Circuit drawings use one wire per slot but do not depict register dimension.
Unknown or custom operations appear as labeled boxes. A direct
``PulseOperation`` cannot be represented and raises
:py:class:`~fatqat.errors.UnsupportedOperationError`.

Container reference
-------------------

.. autoclass:: fatqat.Program
   :members: add, measure, measure_all, draw, copy, assign_parameters
   :show-inheritance:

Parameter values
----------------

.. autoclass:: fatqat.Parameter
   :members:
   :show-inheritance:

.. autoclass:: fatqat.ParameterVector
   :members:
   :show-inheritance:
