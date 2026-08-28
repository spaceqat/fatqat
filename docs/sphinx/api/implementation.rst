Matrix implementations
======================

Use :py:class:`~fatqat.implementation.MatrixImplementationMap` to add custom
matrix operations to a simulator or choose gate rules by physical target. Pass
the map as ``implementation_map=`` when constructing the backend. Built-in
gates work without a custom map.

Choose a map
------------

:py:func:`~fatqat.implementation.default_matrix_implementation_map` returns a
fresh map containing FATQAT's built-in matrix gates. Editing it does not affect
another map or a later call. Construct
:py:class:`~fatqat.implementation.MatrixImplementationMap` directly when the
backend should support only rules you add.

Choose a rule
-------------

:py:meth:`MatrixImplementationMap.add
<fatqat.implementation.MatrixImplementationMap.add>` accepts an operation
instance or class and one of three rule forms:

.. list-table:: Matrix rule forms
   :header-rows: 1
   :widths: 24 38 38

   * - Form
     - Use it for
     - Value passed to ``add``
   * - Two-dimensional NumPy array
     - One constant matrix
     - The array; FATQAT copies it, so later changes do not affect the rule
   * - Callable
     - A matrix that depends on operation parameters or target dimensions
     - ``rule(op)`` by default. If the callable accepts a ``targets=`` keyword
       or ``**kwargs``, FATQAT calls ``rule(op, targets=targets)``.
   * - :py:class:`~fatqat.implementation.MatrixImplementation`
     - A configured or stateful rule object
     - Override ``__call__(op, *, targets)``.

``op`` is the applied operation value, so it contains parameters such as a
rotation angle. ``targets`` contains scalar program
:py:class:`~fatqat.RegisterRef` objects in operand order; a rule can inspect
``target.register.dim`` when its matrix depends on subsystem dimension.

Local basis order
-----------------

Matrix factors follow the target tuple passed to
:py:meth:`~fatqat.Program.add`. The first target is the most-significant local
factor and the last target changes fastest. For dimensions ``(d0, d1, ..., dk)``
and basis digits ``(b0, b1, ..., bk)``, the flat local index is
``b0 * d1 * ... * dk + b1 * d2 * ... * dk + ... + bk``.

For two qubits, targets ``(q0, q1)`` therefore use the local basis ``|00>``,
``|01>``, ``|10>``, ``|11>``. Controlled operations list controls before
targets. This local convention is independent of the full-system display and
result bit order.

Target-specific rules
---------------------

Pass ``device_operands=`` to ``add`` to register a rule only for that exact
ordered tuple of backend-defined physical labels. These labels are not program
register references. One operation family uses exactly one registration mode:

- A *uniform* rule, added without ``device_operands``, matches every physical
  tuple of the correct arity.
- *Device-specific* rules match only their explicitly registered tuples. You
  may add several tuples for the same operation.

Call :py:meth:`~fatqat.implementation.MatrixImplementationMap.remove` before
switching a family between these modes.

Validation timing
-----------------

``add`` rejects an invalid registration immediately with ``TypeError`` or
``ValueError``. Callables run when a backend prepares a program. A rule that
raises produces :py:class:`~fatqat.errors.MatrixImplementationError`, an array
with the wrong target shape produces
:py:class:`~fatqat.errors.BackendValidationError`, and a missing rule produces
:py:class:`~fatqat.errors.UnsupportedOperationError`. These errors occur before
``run`` returns a :py:class:`~fatqat.Job`.

Reference
---------

.. py:data:: fatqat.implementation.DeviceOperands

   Alias for an ordered tuple of hashable backend-defined physical labels.

.. autofunction:: fatqat.implementation.default_matrix_implementation_map

.. autoclass:: fatqat.implementation.MatrixImplementationMap
   :members:
   :show-inheritance:

.. autoclass:: fatqat.implementation.MatrixImplementation
   :members:
   :show-inheritance:

.. autoclass:: fatqat.implementation.FixedMatrix
   :members:
   :show-inheritance:
