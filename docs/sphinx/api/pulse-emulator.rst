Superconducting pulse emulator
==============================

The superconducting pulse emulator is the physical-control counterpart to
:py:class:`~fatqat.backends.SimulatorBackend`. Both accept an ordinary
:py:class:`~fatqat.Program`, validate and lower it through an implementation
map, and return an eager :py:class:`~fatqat.Job`. They differ after lowering:
the matrix backend applies finite matrices or Kraus maps to program-level
subsystems, while :py:class:`~fatqat.backends.PulseBackend` schedules sampled
controls and integrates the full three-level transmon model.

All supported imports on this page come from ``fatqat.backends``. The classes
that schedule lowered blocks, manage shots, or adapt them to QuTiP live under
``fatqat.emulator`` but remain private implementation details. In particular,
applications do not construct ``PulseEngine``, ``PulseBlock``, or
``SCQutipAdapter``.

Backend lifecycle
-----------------

Construct a backend from a physics-model document and its matching calibration:

.. code-block:: python

   import json
   import fatqat as fq

   with open("model.json", encoding="utf-8") as stream:
       model = fq.backends.load_physics_model(json.load(stream))
   with open("calibration.json", encoding="utf-8") as stream:
       calibration = fq.backends.load_calibration_spec(json.load(stream), model)

   backend = fq.backends.PulseBackend(model, calibration)

The calibration must repeat the model's builder ID/version and model
ID/revision exactly. Program qubits bind to ``model.subsystem_ids`` in
declaration order. Unaddressed model transmons still participate in the full
physical state and therefore still contribute factors of three to result and
propagator dimensions.

``PulseBackend(...)`` accepts these optional extension inputs:

.. list-table:: Constructor options
   :header-rows: 1
   :widths: 28 72

   * - Argument
     - Contract
   * - ``noise``
     - A :py:class:`~fatqat.NoiseModel`, retained by reference. ``None`` uses
       an initially empty model.
   * - ``lindblad_implementation_map``
     - A :py:class:`~fatqat.noise.LindbladImplementationMap` mapping channel
       descriptors to local collapse operators. The backend copies it;
       ``None`` uses :py:func:`~fatqat.noise.default_lindblad_implementation_map`.
   * - ``pulse_implementation_map``
     - A :py:class:`~fatqat.backends.PulseImplementationMap` mapping operation
       families/ordered device operands to reusable pulse definitions. The
       backend copies it; ``None`` uses the built-in map.

Run configuration and results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:meth:`~fatqat.backends.PulseBackend.run` has the same eager job boundary as
:py:meth:`~fatqat.backends.SimulatorBackend.run`, but pulse-specific
configuration and state semantics:

.. list-table:: ``simulation_config`` keys
   :header-rows: 1
   :widths: 24 20 56

   * - Key
     - Default
     - Meaning
   * - ``seed``
     - ``None``
     - Integer seed for measurement and readout sampling.
   * - ``schedule_mode``
     - ``"ASAP"``
     - ``"ASAP"`` or ``"ALAP"`` lightweight placement within each continuous
       region. Both preserve dependencies and claimed-resource exclusion.
   * - ``parallel_mode``
     - ``"auto"``
     - ``"auto"`` normalizes to ``"serial"``. No other mode is supported in
       v0.1.
   * - ``max_workers``
     - ``None``
     - May be ``None`` or ``1`` in v0.1.

.. list-table:: ``result_config`` keys
   :header-rows: 1
   :widths: 24 28 48

   * - Key
     - Default
     - Meaning
   * - ``counts``
     - Whether measurement exists
     - Return sampled classical counts. This requires a positive integer
       ``shots``.
   * - ``final_state``
     - Whether measurement is absent
     - Return the terminal full-model physical density matrix. With
       measurement this is one sampled posterior and requires ``shots == 1``.

The density matrix has shape ``(3**m, 3**m)`` for all ``m`` model transmons.
Measurement first samples a physical level, maps ``0, 1, 2`` to ``0, 1, 1``,
then applies any classical readout-confusion matrix. Reset prepares physical
``|0>``.

Validation failures (invalid documents, model capacity/dimension, selectors,
configuration, unsupported operations, and pulse-rule failures) raise directly
from ``run()``. Failures after solver execution begins are represented by a
failed returned job; ``job.result()`` raises
:py:class:`~fatqat.errors.BackendExecutionError`.

Coherent propagators
~~~~~~~~~~~~~~~~~~~~

:py:meth:`~fatqat.backends.PulseBackend.propagator` returns a complex NumPy
array for the complete physical model. Measurement, reset, and classical
conditions are rejected because they do not define one coherent operator.
Nonzero evolution with bound collapse operators is also rejected. A
zero-duration, frame-only program remains coherent because no time elapses.

Intermediate virtual-frame updates always rotate later phase-sensitive
controls. ``apply_final_frame=True`` (the default) additionally composes the
remaining terminal frame ledger. ``False`` omits only this last transform.
The result uses per-subsystem near-resonant rotating frames and may differ from
a conventional qubit ``RZ`` by global phase; compare against ideal matrices
phase-invariantly.

Backend reference
~~~~~~~~~~~~~~~~~

.. autoclass:: fatqat.backends.PulseBackend
   :members: run, propagator, validate_noise

Physics model and calibration
-----------------------------

The public loaders accept JSON-compatible mappings and return immutable
objects. Documents are exact-schema, data-only envelopes: missing/unknown
keys, unsupported versions/builders, non-finite values, or executable Python
objects are rejected. Loading the same persisted document twice creates two
model instances with the same durable identity but distinct opaque handles;
handles must never be mixed between instances.

The built-in model contains fixed qutrit transmons and an arbitrary undirected
coupling graph. A coupling declares where controlled exchange operations may
be driven; it is not a residual always-on exchange Hamiltonian. Frequencies
define the implicit resonant carriers. The current model uses ``Delta_i = 0``,
so changing a frequency alone does not alter current numerical evolution.

Loader and builder reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: fatqat.backends.load_physics_model

.. autofunction:: fatqat.backends.load_calibration_spec

.. autoclass:: fatqat.backends.SCTransmonExchangeBuilder
   :members: build

Returned physics-model interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Application code normally obtains the model from
:py:func:`~fatqat.backends.load_physics_model` and does not import or construct
its concrete class. Its supported read interface is:

``model.key``
   Complete immutable builder/model snapshot identity.

``model.subsystems``
   Ordered immutable transmon records with ``id``, ``frequency_ghz``, and
   ``anharmonicity_ghz``.

``model.couplings``
   Immutable undirected edge records with ``id`` and two ``subsystem_ids``.

``model.annihilation``, ``model.creation``, ``model.number``
   Read-only local qutrit matrices. They are local operators, not full-model
   tensor expansions.

.. autoattribute:: fatqat.emulator.superconducting.PhysicsModel.time_unit

.. autoattribute:: fatqat.emulator.superconducting.PhysicsModel.subsystem_ids

.. autoattribute:: fatqat.emulator.superconducting.PhysicsModel.physical_dimension

The following accessors mint opaque handles for custom pulse definitions.
Never instantiate handle classes directly. ``resource`` claims a subsystem;
``coupling`` claims an edge for scheduling; the three control accessors choose
the Hamiltonian mechanism; ``frame`` selects a virtual-drive phase ledger.

.. automethod:: fatqat.emulator.superconducting.PhysicsModel.resource

.. automethod:: fatqat.emulator.superconducting.PhysicsModel.drive_control

.. automethod:: fatqat.emulator.superconducting.PhysicsModel.detuning_control

.. automethod:: fatqat.emulator.superconducting.PhysicsModel.exchange_control

.. automethod:: fatqat.emulator.superconducting.PhysicsModel.frame

.. automethod:: fatqat.emulator.superconducting.PhysicsModel.coupling

``bind_resource``, ``bind_control``, ``bind_frame``, and ``bind_coupling``
validate that a handle was minted by this exact model instance and return its
ordinal. They are useful inside a custom realization when translating the
ordered ``targets`` supplied by the implementation map.

.. automethod:: fatqat.emulator.superconducting.PhysicsModel.bind_resource

.. automethod:: fatqat.emulator.superconducting.PhysicsModel.bind_control

.. automethod:: fatqat.emulator.superconducting.PhysicsModel.bind_frame

.. automethod:: fatqat.emulator.superconducting.PhysicsModel.bind_coupling

Returned calibration interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``calibration.key`` identifies its exact model snapshot and
``calibration.recipes`` is recursively immutable. Custom rules should normally
read a named recipe through:

.. automethod:: fatqat.emulator.superconducting.CalibrationSpec.recipe

The built-in schema contains ``rx_ry``, ``iswap``, and per-edge ``cz``
recipes. ``RZ`` is virtual and deliberately has no calibration recipe.

Pulse implementation maps
-------------------------

A pulse implementation map is the direct analog of the matrix family's
:py:class:`~fatqat.implementation.MatrixImplementationMap`:

.. code-block:: text

   SimulatorBackend: operation -> matrix rule -> matrix -> matrix plan step
   PulseBackend:     operation -> pulse rule  -> PulseDefinition -> PulseBlock

The last ``PulseBlock`` is private. Lowering attaches occurrence-specific
conditions, resolved noise, engine indices, and optional schedule position;
a reusable :py:class:`~fatqat.backends.PulseDefinition` contains none of
those facts.

A rule has exactly this callable shape:

.. code-block:: python

   def rule(operation, *, targets, model, calibration):
       return fq.backends.PulseDefinition(...)

``targets`` are ordered model-minted subsystem-resource handles corresponding
to the operation's ordered device operands. They are not program
``RegisterRef`` values and not engine indices. The rule may inspect immutable
model facts and calibration recipes and must return one pulse definition.

Registration follows the matrix map's two mutually exclusive modes per
operation family: one unconstrained rule, or a finite table keyed by ordered
``device_operands``. Calling ``add`` again replaces a rule in the same mode.
Call ``remove(op)`` before changing modes. A custom rule's deliberate
:py:class:`~fatqat.errors.BackendValidationError` (including
:py:class:`~fatqat.errors.UnsupportedOperationError`) propagates unchanged;
other exceptions and non-``PulseDefinition`` returns become
:py:class:`~fatqat.errors.PulseImplementationError`.

.. autofunction:: fatqat.backends.default_superconducting_pulse_implementation_map

.. autoclass:: fatqat.backends.PulseImplementationMap
   :members:

Pulse-authoring values
----------------------

All pulse time values use ``model.time_unit``. The authoring types themselves
are unit-neutral. The built-in transmon model's unit is nanoseconds.

A positive-duration definition requires at least one sampled control; a
zero-duration definition forbids controls. Every control must end within the
enclosing duration. A definition always claims at least one model resource,
including for virtual operations. Controls sharing one channel must be summed
explicitly before construction rather than relying on implicit addition.

.. autoclass:: fatqat.backends.PulseDefinition
   :members:

.. autoclass:: fatqat.backends.SampledControl
   :members:

.. autoclass:: fatqat.backends.PhaseShift
   :members:

.. autoclass:: fatqat.backends.PhaseSwap
   :members:

Custom realization example
~~~~~~~~~~~~~~~~~~~~~~~~~~

Start from a fresh default map when changing one built-in mechanism. The
backend copies that map at construction, so later changes to the caller's map
do not affect it:

.. code-block:: python

   import numpy as np
   import fatqat as fq

   def custom_cz(operation, *, targets, model, calibration):
       first, second = (
           model.subsystem_ids[model.bind_resource(target)] for target in targets
       )
       duration = 20.0
       samples = np.linspace(0.0, duration, 129)
       envelope = np.zeros_like(samples)
       return fq.backends.PulseDefinition(
           duration=duration,
           controls=(
               fq.backends.SampledControl(
                   model.exchange_control(first, second), samples, envelope
               ),
           ),
           resource_claims=(
               model.resource(first),
               model.resource(second),
               model.coupling(first, second),
           ),
       )

   implementations = (
       fq.backends.default_superconducting_pulse_implementation_map()
   )
   implementations.add(fq.ops.CZ, custom_cz)
   backend = fq.backends.PulseBackend(
       model,
       calibration,
       pulse_implementation_map=implementations,
   )

The zero envelope is intentionally only a structural example, not a physical
CZ implementation. A real rule must choose a calibrated waveform and any
required post-frame correction.

Lindblad implementation extension
---------------------------------

The pulse backend's channel-capability declaration is a
:py:class:`~fatqat.noise.LindbladImplementationMap`. A rule receives
``(channel, *, physical_dimension, duration)`` and returns one or more local
square NumPy collapse-operator matrices. ``duration`` is the realized block
duration for operation-scoped noise and ``None`` for always-on noise. The
backend-neutral map does not tensor-expand matrices or decide their activation
interval; lowering and the concrete adapter own those steps.

.. autoclass:: fatqat.noise.LindbladImplementationMap
   :members:
   :inherited-members:

.. autofunction:: fatqat.noise.default_lindblad_implementation_map

Use :py:meth:`~fatqat.backends.PulseBackend.validate_noise` for an advisory,
instance-sensitive capability report. Execution additionally validates each
noise selector against the current program/resource layout.

