Neutral-atom emulators
======================

Both neutral-atom emulators follow the standard
:py:class:`~fatqat.simulator.Simulator` workflow. Pass a
:py:class:`~fatqat.Program` to ``run()``, then call ``job.result()`` on the
eager :py:class:`~fatqat.Job` to get a :py:class:`~fatqat.Result`. They are
pulse-resolved physical emulators rather than modes of
:py:class:`~fatqat.simulator.AtomArraySimulator`.

Use :py:class:`~fatqat.emulator.Atom3LevelEmulator` for calibrated gates or
selected-site direct controls in the physical ``|0>, |1>, |r>`` model. Use
:py:class:`~fatqat.emulator.Atom2LevelEmulator` for directly authored global
controls in the ``|g>, |r>`` model. The comparison and executable workflows
are in :doc:`../guide/neutral-atom-emulation`.

Arrangements and program resources
----------------------------------

Both backends require a regular
:py:class:`~fatqat.emulator.AtomArrangement`. Coordinates are row-major,
``(column * spacing, row * spacing, 0)``, and the current atom models interpret
spacing in micrometres. A program must declare exactly one dimension-two
quantum resource per site; declaration order binds resources to coordinates.
The arrangement describes fixed geometry; it does not track atom loading or
loss.
``arrangement.num_sites`` and ``len(arrangement)`` both return the coordinate
count, which a pulse program must match exactly.
By contrast, ``AtomArraySimulator(num_sites=6)`` declares a maximum gate-level
device capacity and accepts programs with at most six resources; omitting its
``num_sites`` argument leaves that simulator unbounded.

.. autoclass:: fatqat.emulator.AtomArrangement
   :members: chain, rectangular, num_sites, distance_unit

Run configuration and results
-----------------------------

Both ``run()`` methods have the signature ``(program, *, shots=1024,
resource_layout=None, simulation_config=None, result_config=None)``. The
optional layout must still cover every arrangement site exactly once; the
default uses declaration order. Validation errors are raised
directly from ``run()`` before a job is returned. A failure after execution
starts is represented by a failed job, and ``job.result()`` raises
:py:class:`~fatqat.errors.BackendExecutionError`.

``simulation_config`` accepts only ``seed`` and ``schedule_mode``:

.. list-table:: Simulation configuration
   :header-rows: 1
   :widths: 18 24 16 42

   * - Key
     - Type
     - Default
     - Effect and constraints
   * - ``seed``
     - ``int`` or ``None``; not ``bool``
     - ``None``
     - Random seed for measurement, readout, and trajectory sampling. Integers
       must be non-negative; ``None`` chooses a fresh seed.
   * - ``schedule_mode``
     - ``"ASAP"`` or ``"ALAP"``
     - ``"ASAP"``
     - Place operations as early or as late as their dependencies allow.

``result_config`` accepts only these keys:

.. list-table:: Result configuration
   :header-rows: 1
   :widths: 18 24 16 42

   * - Key
     - Type
     - Default
     - Effect and constraints
   * - ``counts``
     - ``bool`` or ``None``
     - ``None``
     - ``True`` requests classical counts, ``False`` suppresses them, and
       ``None`` enables them when measurement exists. Counts require a
       positive integer ``shots`` value.
   * - ``final_state``
     - ``bool`` or ``None``
     - ``None``
     - ``True`` requests the model- and mode-specific terminal state,
       ``False`` suppresses it, and ``None`` enables it when measurement is
       absent. With physical measurement, it requires ``shots == 1``.

Both configuration arguments must be a ``dict`` or ``None``; unknown keys
are rejected.

Each run starts from a fixed product state: ``|0>`` on the three-level backend
and ``|g>`` on the two-level backend. Neither constructor accepts an
``initial_state`` argument.

The available final-state result depends on the backend and execution mode:

.. list-table:: Final-state representations
   :header-rows: 1
   :widths: 32 24 24 20

   * - Backend/mode
     - Result accessor
     - Shape for ``N`` sites
     - Interpretation
   * - Three-level, no measurement
     - :py:meth:`~fatqat.Result.get_density_matrix`
     - ``(3**N, 3**N)``
     - Exact ensemble state.
   * - Three-level, measured
     - :py:meth:`~fatqat.Result.get_density_matrix` when requested
     - ``(3**N, 3**N)``
     - One sampled posterior state; requires ``shots == 1``.
   * - Two-level, no Lindblad evolution
     - :py:meth:`~fatqat.Result.get_statevector`
     - ``(2**N,)``
     - Pure state, or one sampled posterior state after measurement.
   * - Two-level unmeasured Lindblad
     - :py:meth:`~fatqat.Result.get_density_matrix`
     - ``(2**N, 2**N)``
     - Exact ensemble state.
   * - Two-level measured Lindblad
     - :py:meth:`~fatqat.Result.get_statevector` when requested
     - ``(2**N,)``
     - One sampled trajectory and posterior state; requires ``shots == 1``.

Use ``result.available_data`` when code must handle more than one execution
mode. Neither backend exposes QuTiP values.

Three-level atom emulator
-------------------------

:py:class:`~fatqat.emulator.Atom3LevelEmulator` takes a physical model and an
arrangement. Its default gate map supports
:py:class:`~fatqat.operations.RX`,
:py:class:`~fatqat.operations.RY`, :py:class:`~fatqat.operations.RZ`, and
:py:data:`~fatqat.operations.CZ`. It also supports measurement, reset,
barriers, and classical conditions.

The local physical basis is ``|0>, |1>, |r>``. Measurement maps those levels
to ``0, 1, 1`` before binary readout confusion. The complete qutrit state is
retained; ``|r>`` is coherent leakage and is not physical atom loss.

The signed ``C6/R^6`` drift includes every occupied pair. The calibrated CZ
pulse is fixed and is not retuned when spacing or ``C6`` changes. Binary
``2 x 2`` classical readout confusion is built in. Continuous Lindblad
declarations are not supported by this family. See
:ref:`noise-emulator-support` for the family support table.

Construction and execution
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fatqat.emulator.Atom3LevelEmulator
   :members: model, arrangement, run, propagator, validate_noise_model

``propagator()`` returns the coherent full-qutrit ``(3**N, 3**N)`` operator.
It rejects measurement, reset, and conditions. ``apply_final_frame=True``
includes the final virtual-frame transformation; ``False`` omits that final
transformation. Readout-only noise does not affect the propagator.

Model and calibration values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Atom3LevelModel.from_document(...)`` parses a decoded mapping that matches
the model schema exactly. Use ``load_model_document("atom3level.reference")``
for the packaged reference; load and decode custom files before passing the
mapping. The calibration class accepts its own decoded calibration mapping.
The model defines species, basis and transitions, quantity units, mass, and
signed ``C6``. The calibration supplies the Raman and CZ recipe values.
Unlike the narrower Transmon and two-level runtime models, the current
three-level model retains its public ``kind``, ``local_dimension``, ``species``,
state and interaction parameters, and ``mass_unit``, ``distance_unit``,
``time_unit``, ``angular_frequency_unit``, and ``c6_unit`` inventory. That
family-specific inventory is unchanged by this cleanup.

.. py:class:: fatqat.emulator.Atom3LevelModel

   Create instances with
   :py:meth:`~fatqat.emulator.Atom3LevelModel.from_document`; direct
   construction is not supported.

.. automethod:: fatqat.emulator.Atom3LevelModel.from_document

.. autoattribute:: fatqat.emulator.Atom3LevelModel.control

.. autoattribute:: fatqat.emulator.Atom3LevelModel.available_controls

.. automethod:: fatqat.emulator.Atom3LevelModel.frame

.. autoattribute:: fatqat.emulator.Atom3LevelModel.time_unit

.. autoclass:: fatqat.emulator.Atom3LevelCalibration
   :members:

.. autofunction:: fatqat.emulator.default_atom_3level_calibration

.. autofunction:: fatqat.emulator.default_atom_3level_gate_implementation_map

The standard builder requires ``model=`` and ``calibration=`` and returns a
new :py:class:`~fatqat.emulator.PulseImplementationMap`. Its rules use that
model's channels and frames. Geometry and C6 affect physical evolution but do
not retune the built-in pulse shapes.

Two-level atom emulator
-----------------------

:py:class:`~fatqat.emulator.Atom2LevelEmulator` requires an
``Atom2LevelModel`` and an arrangement. Its global drive and detuning channels
act on every site; see :doc:`pulse-control/pulse-operation` for how to add a
direct pulse block. A terminal measurement may follow the pulse program, and
barriers are ignored. The built-in gate map is empty, so ordinary gates
require a custom map. Reset, conditions, per-site controls, mid-circuit
measurement, and pulses after measurement are not supported.

Construction and execution
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fatqat.emulator.Atom2LevelEmulator
   :members: model, arrangement, interaction_cutoff, run, propagator, validate_noise_model

``propagator()`` returns a coherent ``(2**N, 2**N)`` operator. It rejects
measurement, and rejects Lindblad noise when the program has nonzero duration.
A zero-duration program returns identity even with such noise because no time
elapses.

Model and controls
~~~~~~~~~~~~~~~~~~

The runtime model exposes basis order ``("g", "r")``, the pulse time unit,
and global control selectors. It contains no geometry or calibration. Retain
the decoded source document when application code needs persisted species,
state labels, signed ``C6``, interaction-law metadata, parameter units, or
channel bounds. Derive the local dimension as ``len(model.basis_order)``.

.. py:class:: fatqat.emulator.Atom2LevelModel

   Create instances with
   :py:meth:`~fatqat.emulator.Atom2LevelModel.from_document`; direct
   construction is not supported.

.. automethod:: fatqat.emulator.Atom2LevelModel.from_document

.. autoattribute:: fatqat.emulator.Atom2LevelModel.control

.. autoattribute:: fatqat.emulator.Atom2LevelModel.available_controls

.. autoattribute:: fatqat.emulator.Atom2LevelModel.basis_order

.. autoattribute:: fatqat.emulator.Atom2LevelModel.time_unit

The global drive accepts a complex :py:class:`~fatqat.emulator.SampledWaveform`;
its complex values encode amplitude and phase together. The global detuning
accepts real samples. Both use ``rad/us`` and apply to every arrangement site.
The selector's ``coefficient_unit`` property is the runtime source for that
unit.

Interaction cutoff
~~~~~~~~~~~~~~~~~~

The default ``interaction_cutoff=None`` keeps every pair and preserves the
complete ``C6/R^6`` Hamiltonian. A finite nonnegative cutoff keeps pairs whose
Euclidean distance is at or below that value in
``arrangement.distance_unit`` (currently micrometres); ``0.0`` disables pair
interactions. For a rectangular
arrangement, ``interaction_cutoff=arrangement.spacing`` keeps only horizontal
and vertical nearest pairs. This is a numerical Hamiltonian truncation, not a
physical blockade radius.

Lindblad noise and result types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The built-in forms are listed at :ref:`noise-emulator-support`. Each background
registration names one site; enumerate sites explicitly to apply the same noise
at several sites. Rates use inverse microseconds and relaxation times use
microseconds. Finite ``p`` forms are not converted with a pulse duration.
Binary :py:class:`~fatqat.noise.ReadoutConfusion` is a classical report channel
applied only to the reported digit after physical collapse, not a Lindblad
operator.

The family-owned built-ins are amplitude damping, phase damping, thermal
relaxation, and depolarizing noise. They accept background declarations only;
operation-scoped continuous noise is unsupported.

With no Lindblad noise declaration, the two-level backend uses pure-state
evolution. An unmeasured noisy program returns an exact ensemble density
matrix. A noisy program with terminal measurement uses seeded trajectories. A
zero-time measured program samples the initial state without time evolution.
Even a zero-rate Lindblad declaration selects the noisy result type.

See :doc:`pulse-control/index` for direct pulse authoring and
:doc:`../guide/neutral-atom-emulation` for the complete two-level workflow.
