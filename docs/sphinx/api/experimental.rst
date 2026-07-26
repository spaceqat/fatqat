Experimental and evolving API
=============================

The interfaces on this page are available for prototyping, tests, and
integration work, but their lifecycle semantics are not yet a stable
application contract. They are documented here so users can make informed
use of them without confusing them with the normal result workflow.

Job lifecycle
-------------

:py:meth:`~fatqat.backends.Backend.run` returns a :py:class:`~fatqat.Job`. The ordinary path remains
``job.result()``. The current constructor and helpers are:

- :py:class:`~fatqat.Job` (``status, result=None, error=None``)
- :py:meth:`~fatqat.Job.done` creates a completed job.
- :py:meth:`~fatqat.Job.failed` creates an error job.
- :py:meth:`~fatqat.Job.result` returns the result payload or re-raises the stored
  terminal error.

Current simulator jobs are terminal when returned. A completed job has
status ``"DONE"``; an error job has status ``"ERROR"``. Treat these status
strings and the current eager behavior as evolving: do not build polling,
queuing, or long-running orchestration around them yet.

Detailed Job reference
----------------------

.. autoclass:: fatqat.Job
   :members:
   :show-inheritance:

:py:attr:`~fatqat.Job.status` records the current state. :py:attr:`~fatqat.Job.error` holds the stored
exception for an ``"ERROR"`` job and is otherwise ``None``.

Direct :py:class:`~fatqat.Result` construction
-----------------------------------------------

The current constructor is:

:py:class:`~fatqat.Result` (``counts=None, statevector=None, available=frozenset(), metadata=None, classical_dims=(), density_matrix=None``)

Use it only when adapting an external execution path or creating focused
tests. Normal programs should receive a ``Result`` from ``job.result()`` so
the backend can populate metadata and the available-data contract
consistently.

For the stable result readers, see :doc:`result`. For the normal program
flow, start with :doc:`../guide/running-and-results`.

Implementation-map customization
--------------------------------

An implementation map tells a matrix-family backend how to resolve an
operation to the local matrix used at execution time. It is an experimental
extension point for custom operations, alternate matrix rules, and
device-specific backend behavior. Ordinary programs should use the backend
defaults.

A minimal map needs rules only for the operations used by its program. This
example supplies a matrix for ``H`` and then runs a one-operation program:

.. code-block:: python

   import numpy as np
   import fatqat as fq
   import fatqat.operations as op

   rules = fq.implementation.ImplementationMap()
   rules.add(
       op.H,
       np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
   )

   backend = fq.backends.SimulatorBackend(implementation_map=rules)
   program = fq.Program(1)
   program.add(op.H, 0)
   result = backend.run(
       program,
       shots=1,
       result_config={"counts": False, "statevector": True},
   ).result()

To retain the normal catalog and change just one rule, start from
:py:func:`~fatqat.implementation.default_matrix_implementation_map`. Use
``remove(op)`` before registering a replacement for an existing rule.

:py:meth:`~fatqat.implementation.ImplementationMap.add` (``op, implementation, *, device_operands=None``)
accepts an operation instance or class. ``implementation`` may be a
:py:class:`~fatqat.implementation.MatrixImplementation`, a NumPy array (wrapped as :py:class:`~fatqat.implementation.FixedMatrix`), or a
callable that returns a matrix for the applied operation. A callable may
accept ``op`` alone or ``op, targets``; use :py:class:`~fatqat.implementation.MatrixImplementation` and
override ``__call__`` when the rule is stateful or configured.

Use :py:meth:`~fatqat.implementation.ImplementationMap.supports` to check whether any rule exists and
:py:meth:`~fatqat.implementation.ImplementationMap.implementation_for` (``op, device_operands=...``) to inspect the selected
rule. :py:meth:`~fatqat.implementation.ImplementationMap.device_operands_for` lists a finite set of explicit
device-target keys, and :py:meth:`~fatqat.implementation.ImplementationMap.copy` creates an independent registry of
registrations.

A rule with no ``device_operands`` applies uniformly to that operation
family. A rule with ``device_operands=(...)`` is specific to a
backend-defined, ordered physical target tuple—not a program
``RegisterRef``. For one operation family, uniform and device-specific
registrations are mutually exclusive; remove the old rule before switching
modes. Invalid operation families or variable arity raise ``TypeError``;
non-square matrices and mixed registration modes raise ``ValueError``.

Detailed implementation-map reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: fatqat.implementation.default_matrix_implementation_map

.. autoclass:: fatqat.implementation.ImplementationMap
   :members:
   :show-inheritance:

.. autoclass:: fatqat.implementation.MatrixImplementation
   :members:
   :show-inheritance:

.. autoclass:: fatqat.implementation.FixedMatrix
   :members:
   :show-inheritance:
