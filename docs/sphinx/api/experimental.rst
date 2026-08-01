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

   rules = fq.implementation.MatrixImplementationMap()
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

:py:meth:`~fatqat.implementation.MatrixImplementationMap.add` (``op, implementation, *, device_operands=None``)
accepts an operation instance or class. ``implementation`` may be a
:py:class:`~fatqat.implementation.MatrixImplementation`, a NumPy array (wrapped as :py:class:`~fatqat.implementation.FixedMatrix`), or a
callable that returns a matrix for the applied operation. A callable may
accept ``op`` alone or ``op, targets``; use :py:class:`~fatqat.implementation.MatrixImplementation` and
override ``__call__`` when the rule is stateful or configured.

Use :py:meth:`~fatqat.implementation.MatrixImplementationMap.supports` to check whether any rule exists and
:py:meth:`~fatqat.implementation.MatrixImplementationMap.implementation_for` (``op, device_operands=...``) to inspect the selected
rule. :py:meth:`~fatqat.implementation.MatrixImplementationMap.device_operands_for` lists a finite set of explicit
device-target keys, and :py:meth:`~fatqat.implementation.MatrixImplementationMap.copy` creates an independent registry of
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

.. autoclass:: fatqat.implementation.MatrixImplementationMap
   :members:
   :show-inheritance:

.. autoclass:: fatqat.implementation.MatrixImplementation
   :members:
   :show-inheritance:

.. autoclass:: fatqat.implementation.FixedMatrix
   :members:
   :show-inheritance:

Pulse implementation-map customization
---------------------------------------

:py:class:`~fatqat.backends.PulseBackend` resolves each native operation
family (``RX``, ``RY``, ``RZ``, ``iSwap``, oriented ``CZ``) to a physical
pulse realization through a
:py:class:`~fatqat.backends.PulseImplementationMap` - the pulse family's
counterpart to the matrix implementation map above:

.. code-block:: text

   SimulatorBackend: Operation -> MatrixImplementationMap      -> matrix          -> ApplyMatrixStep
   PulseBackend:     Operation -> PulseImplementationMap -> PulseDefinition -> (lowered) pulse block

Replacing or adding a gate realization - for example a custom ``CZ`` -
never requires subclassing :py:class:`~fatqat.backends.PulseBackend` or
touching private emulator modules:

.. code-block:: python

   import fatqat as fq

   def custom_cz(operation, *, targets, model, calibration):
       first, second = (
           model.subsystem_ids[model.bind_resource(t)] for t in targets
       )
       # Build the physical realization from model-owned resources; a rule
       # may also read calibration.recipe(name) for calibrated numbers.
       return fq.backends.PulseDefinition(
           duration=...,
           controls=(...,),          # SampledControl values
           resource_claims=(...,),   # model.resource(...) / model.coupling(...)
           post_actions=(...,),      # optional PhaseShift / PhaseSwap
       )

   implementations = fq.backends.default_superconducting_pulse_implementation_map()
   implementations.add(fq.ops.CZ, custom_cz)

   backend = fq.backends.PulseBackend(
       model, calibration, pulse_implementation_map=implementations
   )

Calibration versus implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are two different kinds of change. Editing the calibration document
(gate durations, DRAG coefficients, and per-edge ``CZ`` detuning) changes the
*numbers* fed into the existing built-in physical
mechanism - still a Hann/DRAG drive for ``RX``/``RY``, still an atomic
detuning-plus-parked-exchange pulse for ``CZ``. Registering a pulse
implementation rule changes the *mechanism* itself: the waveform shape,
which physical control channels are driven, which resources are claimed, or
what frame corrections are applied. A calibration document is never a place
to select or smuggle in executable behavior - it stays plain, immutable
data, validated against exactly one model identity.

The built-in ``CZ`` rule derives its nominal virtual frame correction by
integrating its generated detuning waveform. This is a model-derived
first-version correction; later device-specific phase calibration can further
improve the realized gate quality.

Rule signature and reusable definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pulse implementation rule is a callable ``f(operation, *, targets, model,
calibration) -> PulseDefinition``. ``targets`` are the ordered
physical-model resource handles corresponding to the operation's ordered
program targets - never a program ``RegisterRef`` and never an engine
index. A rule may read immutable facts from ``model`` (subsystem
frequencies, anharmonicities, declared couplings) and from ``calibration``
(via ``calibration.recipe(name)``), and returns only the physical
realization: duration, sampled controls, the model resources/couplings it
claims, and any post-block frame actions.

A :py:class:`~fatqat.backends.PulseDefinition` is immutable and carries no
classical condition, resolved noise, engine index, or schedule position -
those are one lowered program occurrence's facts, attached by the backend
only after a rule is selected and invoked. This is also why one definition
is safely reusable: the same ``PulseDefinition`` instance may be returned
for a guarded and an unguarded occurrence of the same gate, and each
becomes an independently conditioned physical block; a rule's own
:py:class:`~fatqat.errors.BackendValidationError` (including
:py:class:`~fatqat.errors.UnsupportedOperationError`) propagates unchanged,
while any other failure, or a non-``PulseDefinition`` return, is reported as
``PulseImplementationError`` (an implementation/extension error, kept out of
the stable :doc:`exceptions` reference alongside its matrix-family
counterpart, ``MatrixImplementationError``).

Registration modes
~~~~~~~~~~~~~~~~~~~

:py:meth:`~fatqat.backends.PulseImplementationMap.add` (``op, implementation,
*, device_operands=None``) follows the same two-mode policy as
:py:meth:`~fatqat.implementation.MatrixImplementationMap.add`: an operation
family has either one unconstrained rule, applying to every legal target of
the correct arity, or a finite set of rules keyed by ordered
``device_operands`` - never both for the same family. Replacing the default
unconstrained ``CZ`` rule is a normal ``add(op.CZ, replacement)``. Building
a device-specific ``CZ`` table instead means calling ``remove(op.CZ)`` on
the existing unconstrained rule first, then registering every supported
ordered edge explicitly; there is no simultaneous
device-specific-override-plus-unconstrained-fallback mode.

Time coordinate
~~~~~~~~~~~~~~~~

``duration``, and every :py:class:`~fatqat.backends.SampledControl`'s
``tlist``/``start_offset``, use the owning model's native time coordinate
(``model.time_unit`` - nanoseconds, for the built-in superconducting
transmon model). The pulse-authoring types themselves are time-unit-neutral
and never assume ``ns``.

Backend copy semantics
~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~fatqat.backends.PulseBackend` copies a supplied
``pulse_implementation_map=`` immediately at construction, exactly like
``implementation_map=`` above: later mutations of the caller's map never
affect an already-constructed backend.

Detailed pulse implementation-map reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: fatqat.backends.default_superconducting_pulse_implementation_map

.. autoclass:: fatqat.backends.PulseImplementationMap
   :members:
   :show-inheritance:

.. autoclass:: fatqat.backends.PulseDefinition
   :members:

.. autoclass:: fatqat.backends.SampledControl
   :members:

.. autoclass:: fatqat.backends.PhaseShift
   :members:

.. autoclass:: fatqat.backends.PhaseSwap
   :members:
