Program
=======

:py:class:`~fatqat.Program` is FATQAT's central object: a device-independent
representation of a quantum workload. It owns the quantum and classical
register declarations, user metadata, and ordered instruction stream of
operations, measurements, and classical conditions. A program records the
requested computation without selecting a device instruction set or promising
that every operation has an implementation on the eventual target.

Construction validates representation-level invariants such as register kind
and ownership, operation arity and generic target constraints, measurement
pairing, and condition shape. It does not decide whether a gate is native,
supported by a selected backend, compatible with a device topology, or
realizable for that device's subsystem dimensions. Those capability checks and
the concrete lowering belong to the selected compiler and backend during
program preparation or execution. The same well-formed program can therefore
be offered to different backends, which may accept or reject different parts
of its instruction stream.

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

The program copies the outer collection but retains its register objects.
Treat the register tuple attributes as construction-time state. ``metadata``
is a mutable, shallow-copied mapping for application-defined string keys;
FATQAT does not reserve or interpret any key. See :doc:`registers` for register
identity, dimensions, grids, and metadata ownership.

Targets
-------

A bare built-in integer is an index into the sole register of the relevant
kind, not a global index across several registers. With multiple registers,
pass an explicit :py:class:`~fatqat.RegisterRef` from the program. References
belong to their original register objects, so a ref from a separately
constructed register is not interchangeable with one from the program.

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
For controlled gates, controls precede targets. See :doc:`registers` for view
selection and pairing, and :doc:`../guide/gates` for gate target order.

Conditions
----------

Pass ``condition=(slot, literal)`` to :py:meth:`~fatqat.Program.add`, or pass a
non-empty tuple or list of such pairs for logical AND. A slot follows the same
integer-or-ref rules as other classical operands. Each literal is a Python
integer in ``0 <= literal < slot.dim``; booleans are also accepted. On a
backend that supports conditions, every term is compared with the current
classical value when the operation is reached. Classical storage starts at
zero, and an earlier measurement can replace a slot's value.

Construction validates the condition shape, slot ownership, and literal
range. It does not require the slot to have been measured and does not promise
that every backend or execution method supports feedforward.

Measurements
------------

:py:meth:`~fatqat.Program.measure` pairs quantum targets with classical
outputs positionally. Both sides must be non-empty, have the same length, and
have matching dimensions at every position. Repeated operands are processed
in pair order; see :doc:`operations/structural` for measurement semantics.

:py:meth:`~fatqat.Program.measure_all` flattens all registers and their members
in declaration order and appends one grouped measurement. It requires equal,
non-zero quantum and classical counts and matching dimensions at every
position. Read :doc:`../guide/measurement-and-conditions` for mid-program
measurement and feedforward workflows.

.. _program-templates:

Templates
---------

Parameters are immutable identity objects. Names are labels only: two
``Parameter("theta")`` objects are different binding keys. Reuse one object
when several operation arguments should share a value.

.. list-table:: Binding forms
   :header-rows: 1
   :widths: 28 34 38

   * - Mapping key
     - Accepted value
     - Constraint
   * - :py:class:`~fatqat.Parameter`
     - Built-in integer or float, or NumPy integer or floating scalar
     - The same object must be supplied directly to an operation parameter.
   * - :py:class:`~fatqat.ParameterVector`
     - One-dimensional NumPy array, or a non-string, non-bytes, non-mapping
       iterable of accepted scalars
     - Consumed once in iteration order. The value length must match and every
       vector element must be used directly as an operation parameter. Bind
       individual elements for a partial vector.

:py:meth:`~fatqat.Program.assign_parameters` may bind any subset and always
returns a new program. Remaining parameters stay symbolic, but numeric
execution and export reject them. String keys, booleans, complex values, and
duplicate vector/element assignments are rejected. Only parameters used
directly as operation arguments are bindable, and binding does not bypass an
operation's value constraints. Read
:doc:`../guide/parameters-and-sweeps` for nested-value limitations, partial
binding, sweep shapes, and execution behavior.

:py:meth:`~fatqat.Program.copy` also returns a new mutable branch. Both methods
retain the same register objects and copy the top-level metadata dictionary;
nested metadata values remain shared. Later instruction additions and
top-level metadata edits are independent. Calls to ``add()``, ``measure()``,
and ``measure_all()`` instead mutate the current program and return ``None``.

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

.. autoclass:: fatqat.Program(quantum_registers, classical_registers=0, *, metadata=None)
   :exclude-members: add, measure, measure_all, draw, copy, assign_parameters

.. automethod:: fatqat.Program.add(op, targets=(), *, condition=None)

.. automethod:: fatqat.Program.measure(targets, outputs)

.. automethod:: fatqat.Program.measure_all()

.. automethod:: fatqat.Program.draw(renderer="matplotlib", **kwargs)

.. automethod:: fatqat.Program.copy()

.. automethod:: fatqat.Program.assign_parameters(values)

Parameter values
----------------

.. autoclass:: fatqat.Parameter
   :members:
   :show-inheritance:

.. autoclass:: fatqat.ParameterVector
   :members:
   :show-inheritance:
