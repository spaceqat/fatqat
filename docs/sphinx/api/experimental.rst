Experimental and evolving API
=============================

The interfaces on this page are available for prototyping, tests, and
integration work, but their lifecycle semantics are not yet a stable
application contract. They are documented here so users can make informed
use of them without confusing them with the normal result workflow.

Direct Job construction and status values
-----------------------------------------

The normal :py:class:`~fatqat.Job` surface is documented in :doc:`job`:
applications receive a job from execution and use its
:py:attr:`~fatqat.Job.status` and :py:meth:`~fatqat.Job.result` members.

Adapters and focused tests may directly construct the current eager terminal
representation as ``Job(status, result=None, error=None)``. A completed job has
status ``"DONE"``; an error job has status ``"ERROR"``. Direct construction,
these exact strings, and the eager-only lifecycle remain evolving. Do not build
polling, queuing, or long-running orchestration around them yet.

Direct :py:class:`~fatqat.Result` construction
-----------------------------------------------

The current constructor is :py:class:`~fatqat.Result`
(``counts=None, statevector=None, available=frozenset(), metadata=None,
classical_dims=(), density_matrix=None, unitary=None, superop=None,
data=None``).

Use it only when adapting an external execution path or creating focused
tests. An ordinary backend or Estimator run returns one ``Result`` from
``job.result()``; a parameter sweep returns an ordered ``list[Result]``. In
both cases the execution family populates each result's metadata and
available-data contract.

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
   import fatqat.operations as ops

   rules = fq.implementation.MatrixImplementationMap()
   rules.add(
       ops.H,
       np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
   )

   backend = fq.simulator.Simulator(implementation_map=rules)
   program = fq.Program(1)
   program.add(ops.H, 0)
   result = backend.run(
       program,
       shots=1,
       result_config={"counts": False, "final_state": True},
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

:py:class:`~fatqat.emulator.TransmonEmulator` resolves each native operation
family (``RX``, ``RY``, ``RZ``, ``iSwap``, oriented ``CZ``) to a physical
pulse realization through a
:py:class:`~fatqat.emulator.PulseImplementationMap` - the pulse family's
counterpart to the matrix implementation map above:

.. code-block:: text

   Simulator: Operation -> MatrixImplementationMap      -> matrix          -> ApplyMatrixStep
   TransmonEmulator:     Operation -> PulseImplementationMap -> PulseDefinition -> (lowered) pulse block

``PulseImplementationMap`` is the public rule-map value type; constructors use
the precise ``gate_implementation_map=`` keyword because direct
``PulseOperation`` controls bypass rule lookup. All three emulators expose the
same optional gate-map capability. The two-level atom family has an empty
built-in map, so ordinary gates require user-supplied rules.

Replacing or adding a gate realization - for example a custom ``CZ`` -
never requires subclassing :py:class:`~fatqat.emulator.TransmonEmulator` or
touching private emulator modules:

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as ops

   def custom_cz(operation, *, device_operands):
       first, second = device_operands
       # Model/calibration facts needed here were compiled into this closure.
       return fq.emulator.PulseDefinition(
           duration=...,
           controls=(...,),          # PulseControl values
           post_actions=(...,),      # optional PhaseShift / PhaseSwap
       )

   calibration = fq.emulator.TransmonCalibration(calibration_document)
   implementations = fq.emulator.default_transmon_gate_implementation_map(
       model=model, calibration=calibration
   )
   implementations.remove(ops.CZ)
   implementations.add(ops.CZ, custom_cz)

   backend = fq.emulator.TransmonEmulator(
       model, gate_implementation_map=implementations
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
data. A custom calibration is a complete separate document; package defaults
are nominal simulation baselines rather than hardware-fidelity guarantees.

The built-in ``CZ`` rule derives its nominal virtual frame correction by
integrating its generated detuning waveform. This is a model-derived
first-version correction; later device-specific phase calibration can further
improve the realized gate quality.

Rule signature and reusable definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A reusable pulse implementation rule is a callable
``f(operation, *, device_operands) -> PulseDefinition``.
``device_operands`` is the same exact ordered tuple used for map selection -
never a program ``RegisterRef`` and never an engine index. Model and
calibration facts are compiled into fixed definitions or callable closures
before the emulator receives the map. The returned definition contains only
duration, sampled controls, and optional post-block frame actions; target
binding derives private scheduling claims later.

A :py:class:`~fatqat.emulator.PulseDefinition` is immutable and carries no
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

:py:meth:`~fatqat.emulator.PulseImplementationMap.add` (``op, implementation,
*, device_operands=None``) follows the same two-mode policy as
:py:meth:`~fatqat.implementation.MatrixImplementationMap.add`: an operation
family has either one unconstrained operand-aware rule, applying across device
operands, or a finite set of rules keyed by ordered
``device_operands`` - never both for the same family. Replacing the default
CZ table with an unconstrained replacement means calling ``remove(ops.CZ)``
first. The same remove-first rule applies when changing standard unconstrained
RX to device-specific entries. Implementations may be direct fixed
``PulseDefinition`` values, operand-unaware callables with explicit
registration operands, or operand-aware reusable callables with an explicitly
named ``device_operands`` parameter.

Time coordinate
~~~~~~~~~~~~~~~~

``duration``, every :py:class:`~fatqat.emulator.PulseControl`'s
``waveform.times``, and its ``start_offset`` use the owning model's native time coordinate
(``model.time_unit`` - nanoseconds, for the built-in superconducting
transmon model). The pulse-authoring types themselves are time-unit-neutral
and never assume ``ns``.

Backend copy semantics
~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~fatqat.emulator.TransmonEmulator` and
:py:class:`~fatqat.emulator.Atom3LevelEmulator` copy a supplied
``gate_implementation_map=`` immediately at construction, exactly like
``implementation_map=`` above: later mutations of the caller's map never
affect an already-constructed backend.

Detailed pulse implementation-map reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The complete generated reference for the map, rule contract, returned model
accessors, and pulse-authoring values is in :doc:`pulse-emulator`.
