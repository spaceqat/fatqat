Program
=======

:py:class:`~fatqat.Program` records a quantum workload without tying it to a
device. Add operations and measurements in execution order, then choose a
backend when you run it.

FATQAT catches malformed targets, measurements, and conditions while you build
the program. The backend checks whether it supports the requested operations,
dimensions, placement, and feedforward.

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

See :doc:`registers` for explicit registers, dimensions, and grid selections.

Targets
-------

A bare integer indexes the sole register of the relevant kind; it is not a
global index across several registers. With multiple registers, index the
register you want and pass the resulting :py:class:`~fatqat.RegisterRef`.

.. list-table:: Target forms
   :header-rows: 1
   :widths: 20 35 45

   * - Form
     - Accepted by
     - Rule
   * - Integer
     - :py:meth:`~fatqat.Program.add`, :py:meth:`~fatqat.Program.measure`, and
       conditions
     - Requires exactly one register of the relevant kind and must be within
       its zero-based bounds.
   * - :py:class:`~fatqat.RegisterRef`
     - :py:meth:`~fatqat.Program.add`, :py:meth:`~fatqat.Program.measure`, and
       conditions
     - Must have the required register kind and come from a register in this
       program.
   * - :py:class:`~fatqat.RegisterView`
     - :py:meth:`~fatqat.Program.add` only
     - :py:class:`~fatqat.operations.RX`,
       :py:class:`~fatqat.operations.RY`, and
       :py:class:`~fatqat.operations.RZ` accept one view.
       :py:data:`~fatqat.operations.CX` and :py:data:`~fatqat.operations.CZ`
       accept two compatible views. Measurement does not accept views.

A :py:class:`~fatqat.operations.PulseOperation` does not use the target forms
above. Add it with ``program.add(operation)`` and no ``targets`` argument.
See :doc:`pulse-control/pulse-operation` for details.

For other operations, ``targets`` is one tuple in operand order. Controlled
gates list controls before targets. See :doc:`registers` for view selection and
pairing, and :doc:`../guide/program` for the ordinary construction workflow.

Conditions
----------

Pass ``condition=(slot, literal)`` to :py:meth:`~fatqat.Program.add`, or pass a
non-empty tuple or list of such pairs for logical AND. A slot follows the same
integer-or-ref rules as other classical operands. Each literal is a Python
integer in ``0 <= literal < slot.dim``; booleans are also accepted. Every term
is compared with the current classical value when the operation is reached.

FATQAT checks each condition when the operation is added. A condition may refer
to an unmeasured slot, whose initial value is zero; an earlier measurement
replaces that value. The backend decides whether it supports feedforward.

Measurements
------------

:py:meth:`~fatqat.Program.measure` pairs quantum targets with classical
outputs positionally. Both sides must be non-empty, have the same length, and
have matching dimensions at every position. Repeated operands are processed
in pair order; see :doc:`operations/structural` for measurement behavior.

:py:meth:`~fatqat.Program.measure_all` flattens all registers and their members
in declaration order and appends one grouped measurement. It requires equal,
non-zero quantum and classical counts and matching dimensions at every
position. Read :doc:`../guide/program` for a mid-program measurement and
feedforward workflow.

.. _program-templates:

Parameter binding
-----------------

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

:py:meth:`~fatqat.Program.assign_parameters` accepts an empty or partial
mapping and returns a new program. It binds only :py:class:`~fatqat.Parameter`
objects used directly as operation arguments. Unbound parameters remain
symbolic and are rejected by numeric execution and export. String keys,
positional assignments, boolean or complex values, and assigning both a vector
and one of its elements are rejected. Replacement values still undergo the
operation's normal validation. Read :doc:`../guide/program` for the authoring
workflow and :doc:`../guide/simulation` for a parameter sweep. The complete
binding and execution contracts are specified here and in :doc:`simulator`.

:py:meth:`~fatqat.Program.copy` and
:py:meth:`~fatqat.Program.assign_parameters` return new programs.
:py:meth:`~fatqat.Program.add`, :py:meth:`~fatqat.Program.measure`, and
:py:meth:`~fatqat.Program.measure_all` update the current program and return
``None``.

Draw
----

FatQat's circuit drawing is based on QuTiP-QIP's circuit drawing tools.
``Program.draw()`` translates the Program's instructions to a rendering
adapter before invoking the selected QuTiP-QIP renderer.

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
:py:class:`~fatqat.operations.PulseOperation` cannot be represented and raises
:py:exc:`~fatqat.errors.UnsupportedOperationError`.

Use :func:`fatqat.draw.to_qubit_circuit` only for low-level integration with
QuTiP-QIP's drawing tools. The returned circuit is a rendering adapter, not an
execution object.

.. autofunction:: fatqat.draw.to_qubit_circuit

Reference
---------

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
