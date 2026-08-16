Neutral-atom emulators
======================

The two neutral-atom emulators share the application-facing lifecycle of
:py:class:`~fatqat.simulator.Simulator`: construct a backend, pass it a
:py:class:`~fatqat.Program`, receive an eager :py:class:`~fatqat.Job`, and read
the requested fields from :py:class:`~fatqat.Result`. They are pulse-resolved
physical emulators rather than modes of
:py:class:`~fatqat.simulator.AtomGridSimulator`.

Use :py:class:`~fatqat.emulator.Atom3LevelEmulator` for calibrated gates or
selected-site direct controls in the physical ``|0>, |1>, |r>`` model. Use
:py:class:`~fatqat.emulator.Atom2LevelEmulator` for directly authored global
controls in the ``|g>, |r>`` model. The comparison and complete executable
workflows are in :doc:`../guide/neutral-atoms`,
:doc:`../guide/atom-3level`, and
:doc:`../guide/atom-2level`.

Common arrangement and binding
------------------------------

Both backends require an immutable rectangular
:py:class:`~fatqat.AtomArrangement`. Coordinates are row-major,
``(column * spacing, row * spacing, 0)``, and the current atom models interpret
spacing in micrometres. A program must declare exactly one dimension-two
quantum resource per site; declaration order binds resources to coordinates.
Every site is initially occupied.

.. autoclass:: fatqat.AtomArrangement
   :members: rectangular, cardinality

Run and result contract
-----------------------

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
   :widths: 24 20 56

   * - Key
     - Default
     - Meaning
   * - ``seed``
     - ``None``
     - Integer seed for physical measurement, readout, and trajectory
       sampling, as applicable.
   * - ``schedule_mode``
     - ``"ASAP"``
     - ``"ASAP"`` or ``"ALAP"`` placement within each continuous region.

``result_config`` accepts only ``counts`` and ``final_state``. Counts default
on when measurement exists; the natural final state defaults on when it does
not. Counts require a positive integer ``shots``. A sampled physical
measurement can return one posterior final state only when ``shots == 1``.

The concrete final-state field differs by backend and two-level execution mode:

.. list-table:: Final-state representations
   :header-rows: 1
   :widths: 32 24 24 20

   * - Backend/mode
     - Result accessor
     - Shape for ``N`` sites
     - Interpretation
   * - Three-level atom
     - :py:meth:`~fatqat.Result.get_density_matrix`
     - ``(3**N, 3**N)``
     - ``exact_density_matrix``
   * - Two-level ideal
     - :py:meth:`~fatqat.Result.get_statevector`
     - ``(2**N,)``
     - ``pure_state``
   * - Two-level unmeasured Lindblad
     - :py:meth:`~fatqat.Result.get_density_matrix`
     - ``(2**N, 2**N)``
     - ``exact_density_matrix``
   * - Two-level measured Lindblad
     - :py:meth:`~fatqat.Result.get_statevector` when requested
     - ``(2**N,)``
     - ``sampled_quantum_trajectory``

Use ``result.available_data`` when code must handle more than one execution
mode. Neither backend exposes QuTiP values.

Family metadata records target model ``format`` and model kind/ID/revision.
Two-level metadata keeps arrangement coordinates, interaction policy, basis/order,
solver, and unit provenance without defining a content digest.

Three-level atom emulator
-------------------------

:py:class:`~fatqat.emulator.Atom3LevelEmulator` requires a strict geometry-free
physics model and an arrangement. Its nominal map is compiled internally; its native gates
are :py:class:`~fatqat.operations.RX`,
:py:class:`~fatqat.operations.RY`, :py:class:`~fatqat.operations.RZ`, and
:py:data:`~fatqat.operations.CZ`. Measurement, reset, barriers, and classical
conditions follow the shared pulse-program execution contract.

The local physical basis is ``|0>, |1>, |r>``. Measurement maps those levels
to ``0, 1, 1`` before binary readout confusion. The complete qutrit state is
retained; ``|r>`` is coherent leakage and is not physical atom loss.

The signed ``C6/R^6`` drift includes every occupied pair. The calibrated CZ
pulse is fixed and is not retuned when spacing or ``C6`` changes. Binary
``2 x 2`` classical readout confusion is built in. The default Lindblad map is
empty; a supplied map enables registered qutrit channel descriptors.

Construction and execution
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fatqat.emulator.Atom3LevelEmulator
   :members: model, arrangement, run, propagator, validate_noise

``propagator()`` returns the coherent full-qutrit ``(3**N, 3**N)`` operator.
It rejects measurement, reset, and conditions. ``apply_final_frame=True``
includes the terminal virtual-frame ledger; ``False`` exposes the raw physical
propagator before that final composition. Readout-only noise is inert.

Model and calibration values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The direct constructors accept exact-schema, decoded JSON-compatible mappings
and return semantically comparable, unhashable immutable objects. Application
code owns file I/O. The model owns species, explicit basis and transition
facts, top-level quantity-kind units, mass, and signed ``C6``. The calibration
owns portable Raman and CZ control recipes and contains no model identity.

.. autoclass:: fatqat.emulator.Atom3LevelModel
   :members:

.. autoclass:: fatqat.emulator.Atom3LevelCalibration
   :members:

.. autofunction:: fatqat.emulator.default_atom_3level_calibration

.. autofunction:: fatqat.emulator.default_atom_3level_gate_implementation_map

The standard builder requires ``model=`` and ``calibration=`` and returns a
fresh portable :py:class:`~fatqat.emulator.PulseImplementationMap`. The v1
atom recipes deliberately do not inspect or retain source C6 or geometry.
``Atom3LevelEmulator`` copies a supplied map; direct controls bypass it.
Its optional ``lindblad_implementation_map`` is also copied. Registered
operation-scoped or target-local background generators then use shared
pulse-noise resolution, with ``3 x 3`` local operators and two rates for
qutrit amplitude damping. Finite channel probabilities are not converted.

Two-level atom emulator
-----------------------

:py:class:`~fatqat.emulator.Atom2LevelEmulator` requires a strict
geometry-free two-level model and arrangement. It accepts zero-target
:py:class:`~fatqat.operations.PulseOperation` values containing global drive
and detuning addresses returned by the model, followed by an optional terminal
measurement suffix. Barriers are structural no-ops. Its built-in gate map is
empty, so ordinary gates reject by default; a supplied map can define them.
Reset, conditions, local direct targets, and a pulse after measurement remain
rejected.

Construction and execution
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: fatqat.emulator.Atom2LevelEmulator
   :members: model, arrangement, interaction_policy, run, propagator, validate_noise

``propagator()`` returns a coherent ``(2**N, 2**N)`` operator. It rejects
measurement and any nonempty elapsed plan with accepted Lindblad noise. An
empty plan returns identity even with such noise because no time elapses.

Model and controls
~~~~~~~~~~~~~~~~~~

The model fixes basis order ``("g", "r")``, unit spellings, signed ``C6``,
the ``C6/R^6`` interaction law, and optional channel bounds. It contains no
geometry or calibration.

.. autoclass:: fatqat.emulator.Atom2LevelModel
   :members: angular_frequency_unit, drive_control, detuning_control

The global drive accepts a complex :py:class:`~fatqat.waveforms.SampledWaveform`;
its complex values encode amplitude and phase together. The global detuning
accepts real samples. Both use ``rad/us`` and apply to every arrangement site.

Interaction policy
~~~~~~~~~~~~~~~~~~

The default nearest-neighbor policy includes row/column four-neighbor edges
without enumerating all pairs. The explicit full-pair policy includes every
``i < j`` pair. Both are static during evolution.

.. autoclass:: fatqat.emulator.GridInteractionPolicy
   :members: nearest_neighbor, full_pair

Lindblad capability and mode selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two-level emulator accepts local background, rate-form
:py:class:`~fatqat.noise.AmplitudeDamping` and
:py:class:`~fatqat.noise.PhaseDamping`. Each background registration must name
one site; enumerate sites explicitly when the same generator is present on
several. This is the built-in Lindblad-map behavior. Supplying a replacement
map can add operation-scoped or background generator declarations, subject to
the two-level dimension and local selector restrictions. Finite ``p`` forms
are rejected rather than converted with a pulse duration. Amplitude damping
requires exactly one adjacent-transition rate. Readout confusion and
unregistered channel families are rejected.

With no Lindblad registration, the two-level backend uses a ket-preserving solve.
An unmeasured noisy program uses an exact ensemble density-matrix solve. A
noisy program with terminal measurement uses one seeded trajectory batch. A
zero-time measured program samples the initial ket without a dynamical solver.
Mode selection is presence-based, so an accepted zero-rate channel still
selects the noisy representation.

Public pulse authoring values are documented with the other operations in
:doc:`operations`; interpolation and metadata semantics are explained in
:doc:`../guide/atom-2level`.
