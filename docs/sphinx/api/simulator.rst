Simulator
=========

.. currentmodule:: fatqat.simulator

:class:`Simulator` runs gate-level programs with matrix operations and finite
noise channels. It supports qubits, qudits, mixed register dimensions, and
custom implementation maps. It runs the program as written; it does not
transpile or route it.

Quick start
-----------

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as ops

   bell = fq.Program(2, 2)
   bell.add(ops.H, 0)
   bell.add(ops.CX, (0, 1))
   bell.measure_all()

   backend = fq.simulator.Simulator(method="statevector")
   counts = backend.run(bell, shots=1000).result().get_counts()

The default implementation map covers FATQAT's built-in matrix gates.
State methods also support measurement, reset, and classical conditions.
``Barrier`` has no numerical effect.

Methods
-------

Method names are case-insensitive. ``SV`` and ``DM`` are aliases; the
read-only :attr:`Simulator.method` property returns the full name. If the
program's Hilbert-space dimension is ``D``:

.. list-table:: Simulation methods
   :header-rows: 1
   :widths: 18 26 30 26

   * - Method
     - Result
     - Reset and finite channels
     - Restrictions
   * - ``statevector`` / ``SV``
     - ``statevector``, shape ``(D,)``
     - Samples one trajectory
     - A stochastic final state represents one shot
   * - ``density_matrix`` / ``DM``
     - ``density_matrix``, shape ``(D, D)``
     - Applies them exactly
     - Uses more memory than ``statevector``
   * - ``unitary``
     - ``unitary``, shape ``(D, D)``
     - Rejected
     - Rejects measurement, conditions, counts, and ``initial_state``
   * - ``superop``
     - ``superop``, shape ``(D**2, D**2)``
     - Applies them exactly
     - Rejects measurement, conditions, counts, and ``initial_state``

Super-operators use column-stacking vectorization:

.. code-block:: python

   rho_out = (
       superop @ rho_in.reshape(-1, order="F")
   ).reshape(rho_in.shape, order="F")

For a noise-free program, ``superop`` equals
``numpy.kron(unitary.conj(), unitary)``. A unitary contains ``4**n`` complex
entries for ``n`` qubits, while a super-operator contains ``16**n``; use the
operator methods only for programs small enough to hold the result.

Runtime and execution
---------------------

``runtime`` is chosen when the backend is created. ``"numba"`` is the default
for :class:`Simulator` and the superconducting profiles; it compiles kernels
on first use and supports threaded kernels. ``"numpy"`` runs directly without
compilation and is the default for :class:`AtomArraySimulator`. Both runtimes
support all four methods, but need not produce bit-identical floating-point or
sampled results.

``simulation_config`` changes one call to :meth:`Simulator.run`. Its string
values are case-sensitive.

.. list-table:: Simulation controls
   :header-rows: 1
   :widths: 22 16 62

   * - Key
     - Default
     - Accepted values and effect
   * - ``seed``
     - ``None``
     - Use a non-negative ``int`` or ``None``; booleans are rejected. It
       controls measurement, reset, channel, loss, and readout sampling. A
       negative value is rejected when execution starts, so
       :meth:`fatqat.Job.result` raises ``ValueError``.
   * - ``shot_parallelism``
     - ``"auto"``
     - ``"auto"``, ``"serial"``, ``"threads"``, or ``"processes"``.
       Explicit parallel modes require an eligible counts-only, per-shot run
       with at least two shots and workers. Threads require a compatible Numba
       statevector run.
   * - ``kernel_parallelism``
     - ``"auto"``
     - ``"auto"``, ``"serial"``, or ``"threads"``. Threads require Numba
       and cannot be requested together with parallel shots.
   * - ``max_workers``
     - ``None``
     - ``None`` or a positive ``int``. It caps the selected parallel mode;
       ``1`` conflicts with an explicitly parallel request.
   * - ``fusion``
     - ``False``
     - A ``bool``. ``True`` combines compatible adjacent operations and is
       supported by Numba for ``density_matrix``, ``unitary``, and ``superop``.

Automatic selection uses at most one parallel axis. An explicit unsupported
choice raises an error instead of falling back. See :doc:`../guide/advanced`
for the full eligibility rules and reproducibility boundary.

Customize the backend
---------------------

The constructor also accepts:

.. list-table:: Backend options
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Meaning
   * - ``implementation_map``
     - Matrix rules for operations. ``None`` uses FATQAT's built-in gate set.
   * - ``noise``
     - A :class:`~fatqat.NoiseModel` used by every run. ``None`` is ideal.
   * - ``channel_implementation_map``
     - Rules that turn supported channel descriptors into finite channels.
       ``None`` uses FATQAT's built-in rules.

Run inputs and results
----------------------

Besides ``simulation_config``, :meth:`Simulator.run` accepts:

.. list-table:: Run inputs
   :header-rows: 1
   :widths: 22 18 60

   * - Argument
     - Default
     - Meaning
   * - ``shots``
     - ``1024``
     - Samples used for counts or a stochastic final state. A deterministic
       state-only or operator result does not use this value.
   * - ``resource_layout``
     - ``None``
     - Assigns every program quantum reference to a device label. The generic
       simulator uses integer labels in declaration order. A supplied layout
       must be complete, one-to-one, and compatible with the backend.
   * - ``initial_state``
     - ``None``
     - Starts every shot from this state rather than the all-zero state.
       ``statevector`` accepts shape ``(D,)``; ``density_matrix`` accepts
       ``(D,)`` or ``(D, D)``. Operator methods reject it.

Only the initial state's shape is checked. You are responsible for
normalization and, for density matrices, Hermiticity and positivity.

``result_config`` has two keys. Each accepts ``True``, ``False``, or ``None``;
an omitted or ``None`` value uses the default shown below.

.. list-table:: Result fields
   :header-rows: 1
   :widths: 20 42 38

   * - Key
     - Default
     - Constraint
   * - ``counts``
     - Enabled when the program measures
     - Requires an integer ``shots > 0``
   * - ``final_state``
     - Enabled when the method-native state or map is deterministic
     - A requested stochastic final state requires ``shots == 1``

The concrete final-state field is named ``statevector``, ``density_matrix``,
``unitary``, or ``superop``. Check :attr:`fatqat.Result.available_data` before
reading a field that may not have been requested.

``run()`` returns an eager :class:`~fatqat.Job`. Program and option validation
errors normally raise directly. Errors during execution or result assembly are
stored on the job and re-raised by :meth:`fatqat.Job.result`. See
:doc:`../guide/running-and-results` for result accessors, count ordering, and
state-axis metadata.

Noise
-----

Matrix simulation has no physical timeline. Built-in damping and depolarizing
descriptors therefore use their probability form and apply at operation
boundaries. Rate forms, background sources, and
:class:`~fatqat.noise.ThermalRelaxation` are rejected; convert thermal
relaxation with ``as_channels(duration)`` first. Custom descriptors require a
matching channel rule. :class:`AtomArraySimulator` additionally supports atom
loss.

:meth:`Simulator.validate_noise_model` validates a model without running a
program. A method can still impose a stricter rule: for example, ``unitary``
rejects a finite channel that the backend otherwise recognizes when that
channel matches the program. See :doc:`noise` for selectors and the support
table.

Sweeps
------

:meth:`Simulator.run_sweep` binds each row of a complete object-keyed
parameter batch and returns one eager job containing an ordered
``list[Result]``. Batch and row validation errors raise directly; an execution
failure produces a failed sweep job, and no partial result list is returned.
It reuses a supplied seed for every row, so sampled errors can be correlated.
See :doc:`../guide/parameters-and-sweeps` for accepted batch shapes.

API
---

.. autoclass:: Simulator
   :class-doc-from: both
