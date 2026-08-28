Superconducting pulse emulator
==============================

The :py:class:`~fatqat.emulator.TransmonEmulator` runs a
:py:class:`~fatqat.Program` against a three-level transmon model. It evolves
sampled controls over the full physical model, and ``run()`` returns an eager
:py:class:`~fatqat.Job`.

Ordinary gates use ``gate_implementation_map``. A
:py:class:`~fatqat.operations.PulseOperation` contains its own physical
channels and does not use that map. See :doc:`pulse-control/index` for direct
pulse authoring.

Unless an import path is written explicitly, supported imports on this page
come from ``fatqat.emulator``. ``Transmon``, ``Coupling``, and the GHz
conversion helper are imported from ``fatqat.emulator.superconducting``.

Create the emulator
-------------------

Load the packaged model and construct the emulator:

.. code-block:: python

   import json
   import fatqat as fq

   model_document = fq.emulator.load_model_document("transmon.reference")
   model = fq.emulator.TransmonModel.from_document(model_document)
   backend = fq.emulator.TransmonEmulator(model)

For an explicit calibration, build a gate map from the calibration before
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

The packaged calibration is a reference configuration for simulation, not a
hardware calibration. To customize it, supply a complete calibration document
rather than a partial patch.

Program qubits bind to ``model.subsystem_ids`` in declaration order by
default. ``run(resource_layout=...)`` and ``propagator(resource_layout=...)``
accept an explicit :py:class:`~fatqat.ResourceLayout` whose device labels
are model subsystem IDs. Unaddressed model transmons still participate in the
full physical state and therefore still contribute factors of three to result
and propagator dimensions. Their ordered public identities appear in result
``state_axes`` metadata.

``TransmonEmulator(...)`` accepts these optional arguments:

.. list-table:: Constructor options
   :header-rows: 1
   :widths: 28 72

   * - Argument
     - Meaning
   * - ``noise``
     - A :py:class:`~fatqat.NoiseModel`. ``None`` means no noise.
   * - ``lindblad_implementation_map``
     - A :py:class:`~fatqat.noise.LindbladImplementationMap` mapping channel
       descriptors to local collapse operators. ``None`` uses
       :py:func:`~fatqat.noise.default_lindblad_implementation_map`. See
       :ref:`noise-emulator-support` for built-in coverage.
   * - ``gate_implementation_map``
     - A :py:class:`~fatqat.emulator.PulseImplementationMap` mapping operation
       families and device labels to pulse definitions. ``None`` uses the
       built-in map.

Run
---

:py:meth:`~fatqat.emulator.TransmonEmulator.run` accepts these
``simulation_config`` keys:

.. list-table:: ``simulation_config`` keys
   :header-rows: 1
   :widths: 18 24 16 42

   * - Key
     - Type
     - Default
     - Effect and constraints
   * - ``seed``
     - ``int`` or ``None``; not ``bool``
     - ``None``
     - Seed measurement and readout sampling. Use a non-negative integer;
       ``None`` uses fresh entropy.
   * - ``schedule_mode``
     - ``"ASAP"`` or ``"ALAP"``
     - ``"ASAP"``
     - Place operations as early or as late as possible while preserving
       dependencies and physical-resource conflicts.

These are the only two keys. Pulse emulators reject the matrix backend's
``shot_parallelism``, ``kernel_parallelism``, ``max_workers``, and ``fusion``
settings.

.. list-table:: ``result_config`` keys
   :header-rows: 1
   :widths: 18 24 16 42

   * - Key
     - Type
     - Default
     - Effect and constraints
   * - ``counts``
     - ``bool`` or ``None``
     - ``None``
     - ``True`` requests sampled classical counts, ``False`` suppresses them,
       and ``None`` enables them when measurement exists. Counts require a
       positive integer ``shots`` value.
   * - ``final_state``
     - ``bool`` or ``None``
     - ``None``
     - ``True`` requests the terminal full-model physical density matrix,
       ``False`` suppresses it, and ``None`` enables it when measurement is
       absent. With measurement, it requires ``shots == 1``.

Both configuration arguments must be a ``dict`` or ``None``; unknown keys
are rejected.

Every run begins in the product state with each transmon in physical
``|0>``. Pulse emulators do not accept an ``initial_state`` argument.

The density matrix has shape ``(3**m, 3**m)`` for all ``m`` model transmons.
Measurement first samples a physical level, maps ``0, 1, 2`` to ``0, 1, 1``,
then applies any classical readout-confusion matrix. Reset prepares physical
``|0>``.

Result metadata includes the effective run and result settings, but not the
model or calibration documents.

``run()`` raises validation errors before returning a job. If execution fails
after a job is returned, ``job.result()`` raises
:py:class:`~fatqat.errors.BackendExecutionError`.

Propagators
-----------

:py:meth:`~fatqat.emulator.TransmonEmulator.propagator` returns a complex NumPy
array for the complete physical model. Measurement, reset, and classical
conditions are rejected because they do not define one coherent operator.
Programs that apply Lindblad noise during nonzero-duration evolution are also
rejected. Rate-based noise has no effect when no time elapses.

Intermediate virtual-frame updates always rotate later phase-sensitive
controls. ``apply_final_frame=True`` (the default) additionally composes the
remaining terminal virtual-frame transformation. ``False`` omits only this
last transform.
The result uses per-subsystem near-resonant rotating frames and may differ from
a conventional qubit ``RZ`` by global phase; compare against ideal matrices
phase-invariantly.

Reference
---------

.. autoclass:: fatqat.emulator.TransmonEmulator
   :members: run, propagator, validate_noise_model

Physics model and calibration
-----------------------------

``TransmonModel.from_document(...)`` accepts a decoded JSON-compatible model
mapping; direct model construction is forbidden. The calibration constructor
separately accepts its decoded calibration mapping. Use
``load_model_document("transmon.reference")`` for the packaged reference.
Use ``json.load`` or another JSON reader for custom documents.
Documents must match the selected ``format`` ID and version. Missing or
unknown keys, unsupported versions, non-finite values, and values outside the
documented JSON-compatible types are rejected.

Control and frame addresses name model resources. Invalid addresses are
reported when you call ``run()`` or ``propagator()``.

The built-in model contains fixed qutrit transmons and an arbitrary undirected
coupling graph. A coupling declares where controlled exchange operations may
be driven; it is not a residual always-on exchange Hamiltonian. Frequencies
define the implicit resonant carriers.

Model documents
~~~~~~~~~~~~~~~

.. autoclass:: fatqat.emulator.FormatIdentity

.. autoclass:: fatqat.emulator.ModelIdentity

.. autoclass:: fatqat.emulator.CalibrationIdentity

.. autofunction:: fatqat.emulator.available_model_documents

.. autofunction:: fatqat.emulator.load_model_document

.. py:class:: fatqat.emulator.TransmonModel

   Create instances with
   :py:meth:`~fatqat.emulator.TransmonModel.from_document`; direct construction
   is not supported.

.. automethod:: fatqat.emulator.TransmonModel.from_document

.. autoclass:: fatqat.emulator.TransmonCalibration

.. autoclass:: fatqat.emulator.superconducting.Transmon
   :no-inherited-members:

.. autoclass:: fatqat.emulator.superconducting.Coupling
   :no-inherited-members:

.. autofunction:: fatqat.emulator.default_transmon_calibration

``model.format`` identifies the document schema; ``model.kind`` and
``model.identity`` identify the model family and snapshot.
``calibration.format`` and ``calibration.identity`` identify the calibration
document, which has no target-model field.

``model.subsystems``
   Ordered transmon records with ``id``, ``frequency_ghz``, and
   ``anharmonicity_ghz``.

``model.couplings``
   Undirected edge records with ``id`` and two ``subsystem_ids``.

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

Model and calibration documents store ordinary frequencies in GHz. Pulse
waveforms use angular rates, so custom gate rules should convert document
values with
:py:func:`fatqat.emulator.superconducting.angular_rate_from_ghz`.

.. autofunction:: fatqat.emulator.superconducting.angular_rate_from_ghz

.. autoattribute:: fatqat.emulator.TransmonModel.time_unit

.. autoattribute:: fatqat.emulator.TransmonModel.control_unit

.. autoattribute:: fatqat.emulator.TransmonModel.subsystem_ids

.. autoattribute:: fatqat.emulator.TransmonModel.physical_dimension

The ``model.control`` namespace chooses the Hamiltonian mechanism, while
``frame`` selects a virtual-drive phase. Its methods are also available by
name through the ``model.available_controls`` mapping.
Each mapping entry describes a supported control kind, not every fully bound
channel instance. Selectors expose ``scope``, required ``operands``,
``coefficient_domain``, and ``coefficient_unit`` for lightweight inspection.
Calling a selector returns a channel address. When you run the program, the
emulator checks resource names, declared pairs, waveform type, and values.

.. code-block:: python

   drive = model.control.drive("q0")
   detuning = model.control.detuning("q1")
   exchange = model.control.exchange("q0", "q1")

   assert model.available_controls["drive"] is model.control.drive

   for name, selector in model.available_controls.items():
       print(name, selector.scope, selector.operands,
             selector.coefficient_domain, selector.coefficient_unit)

.. autoattribute:: fatqat.emulator.TransmonModel.control

.. autoattribute:: fatqat.emulator.TransmonModel.available_controls

.. automethod:: fatqat.emulator.TransmonModel.frame

Calibration recipes
~~~~~~~~~~~~~~~~~~~

The built-in calibration schema contains ``rx_ry``, ``iswap``, and per-edge
``cz`` recipes. :py:class:`~fatqat.operations.RZ` is virtual and has no
calibration recipe.

The public scalar unit accessors ``recipe_time_unit``,
``recipe_frequency_unit``, and ``recipe_dimensionless_unit`` describe the
stored recipe quantities. They are distinct from the model's pulse
coordinate ``time_unit`` and ``control_unit``.

Pulse implementation maps
-------------------------

A :py:class:`~fatqat.emulator.PulseImplementationMap` realizes ordinary gates.
The transmon constructor names this capability ``gate_implementation_map``;
direct controls bypass it. The standard builder returns a new map containing
the built-in ``RX``, ``RY``, ``RZ``, ``iSwap``, and ``CZ`` rules for one model
and calibration.

.. autofunction:: fatqat.emulator.default_transmon_gate_implementation_map

See :doc:`pulse-control/gate-realization` for accepted rule forms and errors.

Direct controls
---------------

The same model channels can be used without a gate-realization rule.
Drive and detuning resolve one declared transmon; exchange resolves two
transmons and their declared coupling. Drive accepts a complex envelope for
the two quadratures, while detuning and exchange require real values. Pulse
times use the model units described above. The current transmon model
does not add amplitude or duration limits beyond requiring finite values.

See :doc:`pulse-control/pulse-operation`,
:doc:`pulse-control/pulse-control`, and
:doc:`pulse-control/sampled-waveform` for construction and timing.
``iSwap`` is a gate whose built-in realization uses exchange;
``iSwap`` is not a channel name.

Lindblad noise and custom rules
-------------------------------

Pass a :py:class:`~fatqat.noise.LindbladImplementationMap` to add or replace
Lindblad noise rules. See :doc:`noise/custom-implementations` for the map API
and :ref:`noise-emulator-support` for the built-in backend support table.
When the map is omitted, the default registers
:py:class:`~fatqat.noise.AmplitudeDamping`,
:py:class:`~fatqat.noise.PhaseDamping`, and
:py:class:`~fatqat.noise.ThermalRelaxation`; an explicit map replaces those
rules. Qutrit amplitude damping requires two adjacent-level rates. Rates use
inverse nanoseconds, while ``t1``, ``t2``, and ``t_phi`` use nanoseconds.
Background and ordinary-operation-scoped generators are accepted. Finite
probability forms, ``Loss``, and nonlocal declarations are rejected.

Probability-form channels are not converted to rates. In particular,
:py:class:`~fatqat.noise.PauliChannel` remains Simulator-only even when a rule
is registered. See :ref:`pulse-probability-noise`. A rate-form
``Depolarizing`` declaration also requires a Lindblad map that registers it;
the transmon default map does not.

Call :py:meth:`~fatqat.emulator.TransmonEmulator.validate_noise_model` before
running a program to validate its noise model. Program-specific selectors are
checked at run time.

Neutral-atom pulse emulators
----------------------------

The three-level and two-level atom backends also accept optional gate and
Lindblad implementation maps.
``Atom3LevelEmulator`` has built-in gate recipes and per-site direct
controls. ``Atom2LevelEmulator`` has an empty built-in gate map and global
direct controls; user-supplied maps can add gate rules.
See :doc:`atom-emulators` for their API and
:doc:`../guide/neutral-atoms` for help choosing between them.
