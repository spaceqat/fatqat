Superconducting pulse emulator
==============================

The superconducting pulse emulator is the physical-control counterpart to
:py:class:`~fatqat.simulator.Simulator`. Both accept an ordinary
:py:class:`~fatqat.Program`, validate and lower it through an implementation
map, and return an eager :py:class:`~fatqat.Job`. They differ after lowering:
the matrix backend applies finite matrices or Kraus maps to program-level
subsystems, while :py:class:`~fatqat.emulator.TransmonEmulator` schedules sampled
controls and integrates the full three-level transmon model.

Gate-authored and direct-control programs are independent paths on this
backend. A gate is resolved through ``gate_implementation_map``; a direct
:py:class:`~fatqat.operations.PulseOperation` already carries physical
controls and bypasses that map.

All supported imports on this page come from ``fatqat.emulator``. The classes
that schedule lowered blocks, manage shots, or adapt them to QuTiP live under
``fatqat.emulator`` but remain private implementation details. In particular,
applications do not construct ``PulseEngine``, ``PulseBlock``, or
``_TransmonQutipAdapter``.

Backend lifecycle
-----------------

The common path selects and inspects a named reference document, constructs
the family model explicitly, and uses a nominal package calibration internally:

.. code-block:: python

   import json
   import fatqat as fq

   model_document = fq.emulator.load_model_document("transmon.reference")
   print(model_document["model"])
   print(model_document["units"])
   print(model_document["parameters"])
   print(model_document.get("references", []))
   model = fq.emulator.TransmonModel.from_document(model_document)
   backend = fq.emulator.TransmonEmulator(model)

For an explicit calibration, compile the portable document into a map before
constructing the emulator:

.. code-block:: python

   with open("calibration.json", encoding="utf-8") as stream:
       calibration = fq.emulator.TransmonCalibration(json.load(stream))
   gate_map = fq.emulator.default_transmon_gate_implementation_map(
       model=model,
       calibration=calibration,
   )
   backend = fq.emulator.TransmonEmulator(
       model,
       gate_implementation_map=gate_map,
   )

The package default is a nominal simulation baseline, not a hardware-fidelity
guarantee. A custom calibration is a complete separate document rather than a
patch applied to the packaged JSON.

Program qubits bind to ``model.subsystem_ids`` in declaration order by
default. ``run(resource_layout=...)`` and ``propagator(resource_layout=...)``
accept an explicit :py:class:`~fatqat.ResourceLayout` whose device operands
are model subsystem IDs. Unaddressed model transmons still participate in the
full physical state and therefore still contribute factors of three to result
and propagator dimensions. Their ordered public identities appear in result
``state_axes`` metadata; private tensor indices do not.

``TransmonEmulator(...)`` accepts these optional extension inputs:

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
   * - ``gate_implementation_map``
     - A :py:class:`~fatqat.emulator.PulseImplementationMap` mapping operation
       families/ordered device operands to reusable pulse definitions. The
       backend copies it; ``None`` uses the built-in map.

Run configuration and results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:meth:`~fatqat.emulator.TransmonEmulator.run` has the same eager job boundary as
:py:meth:`~fatqat.simulator.Simulator.run`, but pulse-specific
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

These are the only two keys. The matrix backend's ``shot_parallelism``,
``kernel_parallelism``, ``max_workers``, and ``fusion`` are rejected here
rather than accepted and ignored: pulse execution is a single serial solver
call, with no engine those settings could steer.

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

Result metadata keeps the effective ``simulation_config`` and
``result_config`` plus common solver facts. It does not duplicate the model or
target document; retain those inputs separately when provenance is required.

Validation failures (invalid documents, model capacity/dimension, selectors,
configuration, unsupported operations, and pulse-rule failures) raise directly
from ``run()``. Failures after solver execution begins are represented by a
failed returned job; ``job.result()`` raises
:py:class:`~fatqat.errors.BackendExecutionError`.

Coherent propagators
~~~~~~~~~~~~~~~~~~~~

:py:meth:`~fatqat.emulator.TransmonEmulator.propagator` returns a complex NumPy
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

.. autoclass:: fatqat.emulator.TransmonEmulator
   :members: run, propagator, check_noise_support

Physics model and calibration
-----------------------------

``TransmonModel.from_document(...)`` accepts a decoded JSON-compatible model
mapping; direct model construction is forbidden. The calibration constructor
separately accepts its decoded calibration mapping. Use
``load_model_document("transmon.reference")`` for the packaged reference, while
applications own file and ``json.load`` handling for custom documents.
Documents are exact-schema, data-only envelopes: missing or unknown keys,
unsupported format versions, non-finite values, or executable Python objects
are rejected. Private package-owned registries dispatch the structured
``format`` ID/version before family-specific body validation.

Constructing the same persisted document twice creates semantically equal,
unhashable values with the same durable identity. Public control and frame
addresses are structural and may be reused with a compatible model. During
lowering, the bound target derives opaque target-local scheduling claims from
those addresses.

The built-in model contains fixed qutrit transmons and an arbitrary undirected
coupling graph. A coupling declares where controlled exchange operations may
be driven; it is not a residual always-on exchange Hamiltonian. Frequencies
define the implicit resonant carriers. The current model uses ``Delta_i = 0``,
so changing a frequency alone does not alter current numerical evolution.

Construction and identity reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fatqat.emulator.FormatIdentity

.. autoclass:: fatqat.emulator.ModelIdentity

.. autoclass:: fatqat.emulator.CalibrationIdentity

.. autofunction:: fatqat.emulator.available_model_documents

.. autofunction:: fatqat.emulator.load_model_document

.. autoclass:: fatqat.emulator.TransmonModel

.. autoclass:: fatqat.emulator.TransmonCalibration

.. autofunction:: fatqat.emulator.default_transmon_calibration

``model.format``, ``model.kind``, and ``model.identity`` identify the source
grammar and durable model snapshot. ``calibration.format`` and
``calibration.identity`` identify the portable calibration snapshot. It has
no target-model field.
The catalog exposes only stable logical names and fresh decoded dictionaries;
it does not expose package paths, a generic model builder, or model keys.

``model.subsystems``
   Ordered immutable transmon records with ``id``, ``frequency_ghz``, and
   ``anharmonicity_ghz``.

``model.couplings``
   Immutable undirected edge records with ``id`` and two ``subsystem_ids``.

``model.annihilation``, ``model.number``
   Read-only local qutrit matrices. They are local operators, not full-model
   tensor expansions. The raising operator is not stored separately; derive it
   as ``model.annihilation.conj().T``.

Units
~~~~~

Two units govern every pulse a rule emits, and both come from the model:

``model.time_unit`` (``"ns"``)
   The coordinate for ``PulseDefinition.duration`` and for each
   ``PulseControl`` waveform and ``start_offset``.

``model.control_unit`` (``"rad/ns"``)
   The unit of every ``SampledWaveform.values`` entry, for all three
   channel kinds. This is an **angular** rate, not an ordinary frequency.

Model and calibration documents store ordinary frequencies in GHz under the
unsuffixed keys ``frequency``, ``anharmonicity``, and ``detuning``. A custom rule must
convert those to angular rates before using them as coefficients; emitting
GHz directly is wrong by a factor of ``2*pi`` and cannot be detected from
the samples, so it produces a silently incorrect simulation rather than an
error. Use
:py:func:`fatqat.emulator.superconducting.angular_rate_from_ghz`, which is
the single definition of that conversion and is what the built-in
realizations and the solver adapter both call.

.. autoattribute:: fatqat.emulator.superconducting.TransmonModel.time_unit

.. autoattribute:: fatqat.emulator.superconducting.TransmonModel.control_unit

.. autoattribute:: fatqat.emulator.superconducting.TransmonModel.subsystem_ids

.. autoattribute:: fatqat.emulator.superconducting.TransmonModel.physical_dimension

The immutable ``model.control`` namespace chooses the Hamiltonian mechanism,
while ``frame`` selects a virtual-drive phase ledger. Its selectors are also
available by name through the immutable ``model.available_controls`` mapping.
Each mapping entry describes a supported control kind, not every fully bound
channel instance. Selectors expose ``scope``, required ``operands``,
``coefficient_domain``, and ``coefficient_unit`` for lightweight inspection.
Calling one produces a structural address; the final target still validates
operand existence, declared pairs, waveform type, and coefficient values.

.. code-block:: python

   drive = model.control.drive("q0")
   detuning = model.control.detuning("q1")
   exchange = model.control.exchange("q0", "q1")

   assert model.available_controls["drive"] is model.control.drive

   for name, selector in model.available_controls.items():
       print(name, selector.scope, selector.operands,
             selector.coefficient_domain, selector.coefficient_unit)

.. autoattribute:: fatqat.emulator.superconducting.TransmonModel.control

.. autoattribute:: fatqat.emulator.superconducting.TransmonModel.available_controls

.. automethod:: fatqat.emulator.superconducting.TransmonModel.frame

The emulator's private bound target validates these structural addresses once
during program preparation. Pulse rules normally consume ordered
``device_operands`` values directly; neither public models nor numerical
adapters expose a second binding API.

Returned calibration interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The built-in calibration schema contains ``rx_ry``, ``iswap``, and per-edge
``cz`` recipes. Those persisted recipes normalize to private, unit-explicit
values captured by the built-in realization. Raw ``recipes`` and a generic
``recipe()`` accessor are deliberately not public; custom recipe schemas are
outside the v1 extension contract. ``RZ`` is virtual and has no calibration
recipe.

The public scalar unit accessors ``recipe_time_unit``,
``recipe_frequency_unit``, and ``recipe_dimensionless_unit`` describe the
persisted recipe quantities. They are distinct from the model's pulse
coordinate ``time_unit`` and ``control_unit``.

Pulse implementation maps
-------------------------

A :py:class:`~fatqat.emulator.PulseImplementationMap` is the map value type
used for gate realization. The public type keeps its established pulse-rule
name, while the constructor keyword is the capability-specific
``gate_implementation_map``. Direct controls do not require or consult it.

The map is the direct analogue of the matrix family's
:py:class:`~fatqat.implementation.MatrixImplementationMap`:

.. code-block:: text

   Simulator: operation -> matrix rule -> matrix -> matrix plan step
   TransmonEmulator:     operation -> pulse rule  -> PulseDefinition -> PulseBlock

The last ``PulseBlock`` is private. Lowering attaches occurrence-specific
conditions, resolved noise, engine indices, and optional schedule position;
a reusable :py:class:`~fatqat.emulator.PulseDefinition` contains none of
those facts.

A reusable operand-aware rule has this callable shape:

.. code-block:: python

   def rule(operation, *, device_operands):
       return fq.emulator.PulseDefinition(...)

``device_operands`` is the exact ordered tuple used for map selection, such as
``("q0", "q1")``. It contains neither program ``RegisterRef`` values nor
engine indices. Model and calibration facts needed by realization should already
be compiled into the rule closure or fixed definition.

Registration accepts a fixed ``PulseDefinition``, an operand-unaware callable
registered with explicit ``device_operands``, or an operand-aware callable
with an explicitly named ``device_operands`` parameter. It follows the matrix
map's two mutually exclusive modes per operation family: one unconstrained
operand-aware rule, or a finite table keyed by ordered ``device_operands``.
Calling ``add`` again replaces a rule in the same mode.
Call ``remove(op)`` before changing modes. A custom rule's deliberate
:py:class:`~fatqat.errors.BackendValidationError` (including
:py:class:`~fatqat.errors.UnsupportedOperationError`) propagates unchanged;
other exceptions and non-``PulseDefinition`` returns become
:py:class:`~fatqat.errors.PulseImplementationError`.

.. autofunction:: fatqat.emulator.default_transmon_gate_implementation_map

.. autoclass:: fatqat.emulator.PulseImplementationMap
   :members:

Pulse-authoring values
----------------------

All pulse time values use ``model.time_unit``. The authoring types themselves
are unit-neutral. The built-in transmon model's unit is nanoseconds.

A positive-duration definition requires at least one sampled control; a
zero-duration definition forbids controls. Every control must end within the
enclosing duration. Lowering derives claims from the occurrence, controls,
and frames. Controls sharing one channel must be summed explicitly before
construction rather than relying on implicit addition.

.. autoclass:: fatqat.emulator.PulseDefinition
   :members:

.. autoclass:: fatqat.emulator.PulseControl
   :members:
   :no-index:

.. autoclass:: fatqat.waveforms.SampledWaveform
   :members:
   :no-index:

.. autoclass:: fatqat.emulator.PhaseShift
   :members:

.. autoclass:: fatqat.emulator.PhaseSwap
   :members:

Custom realization example
~~~~~~~~~~~~~~~~~~~~~~~~~~

Start from a fresh default map when changing one built-in mechanism. The
backend copies that map at construction, so later changes to the caller's map
do not affect it:

.. code-block:: python

   import numpy as np
   import fatqat as fq

   def custom_cz(operation, *, device_operands):
       first, second = device_operands
       duration = 20.0
       samples = np.linspace(0.0, duration, 129)
       envelope = np.zeros_like(samples)
       return fq.emulator.PulseDefinition(
           duration=duration,
           controls=(
               fq.emulator.PulseControl(
                   model.control.exchange(first, second),
                   fq.waveforms.SampledWaveform(samples, envelope),
               ),
           ),
       )

   calibration = fq.emulator.TransmonCalibration(calibration_document)
   implementations = fq.emulator.default_transmon_gate_implementation_map(
       model=model,
       calibration=calibration,
   )
   implementations.remove(fq.ops.CZ)
   implementations.add(fq.ops.CZ, custom_cz)
   backend = fq.emulator.TransmonEmulator(
       model,
       gate_implementation_map=implementations,
   )

The zero envelope is intentionally only a structural example, not a physical
CZ implementation. A real rule must choose a calibrated waveform and any
required post-frame correction.

Direct controls
~~~~~~~~~~~~~~~

The same model control addresses can be used without a gate-realization
callback. A direct operation carries no ordinary program targets because its
controls already contain their physical addresses:

.. code-block:: python

   duration = 20.0
   drive = fq.emulator.PulseControl(
       model.control.drive("q0"),
       fq.waveforms.SampledWaveform((0.0, duration), (0.02, 0.02j)),
   )
   exchange = fq.emulator.PulseControl(
       model.control.exchange("q0", "q1"),
       fq.waveforms.SampledWaveform((0.0, duration), (0.01, 0.01)),
   )
   program = fq.Program(2)
   program.add(fq.ops.PulseOperation(duration, (drive, exchange)))

Complex drive values encode the model's two quadratures. ``iSwap`` is a gate
whose built-in realization uses the ``exchange`` mechanism; ``iSwap`` is not
a channel name.

Lindblad implementation extension
---------------------------------

The pulse backend's channel-capability declaration is a
:py:class:`~fatqat.noise.LindbladImplementationMap`. A rule receives
``(declaration, *, physical_dimension)`` and returns one or more local square
NumPy collapse-operator matrices. The declaration must already express its
generator physics; rules do not receive a duration and do not convert finite
probabilities. An operation-bound term evolves over its matched pulse block,
while a target-only term is background noise over elapsed scheduled time.
Resolved generators are single-subsystem; correlated or nonlocal collapse
operators are not expanded through this local payload.

.. autoclass:: fatqat.noise.LindbladImplementationMap
   :members:
   :inherited-members:

.. autofunction:: fatqat.noise.default_lindblad_implementation_map

Use :py:meth:`~fatqat.emulator.TransmonEmulator.check_noise_support` for an advisory,
instance-sensitive capability report. Execution additionally validates each
noise selector against the current program/resource layout.

Neutral-atom pulse emulators
----------------------------

The three-level and two-level atom backends share this pulse execution
foundation. Both accept optional gate and Lindblad implementation maps.
``Atom3LevelEmulator`` has built-in gate recipes and selected-site direct
controls. ``Atom2LevelEmulator`` has an empty built-in gate map and global
direct controls; user-supplied gate rules use the same shared path.
Their task-oriented reference and generated public API live on
:doc:`atom-emulators`. Start with :doc:`../guide/neutral-atoms` when choosing
between them.
