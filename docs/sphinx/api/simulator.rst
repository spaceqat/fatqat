Simulator
=========

.. currentmodule:: fatqat.simulator

:class:`Simulator` is FATQAT's general-purpose gate-level backend. It applies
operations as matrices or finite channels and can return a simulated state,
sampled counts, or the map represented by a program. It supports qubits,
qudits, mixed register dimensions, and custom implementation maps.

Basic use
---------

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as ops

   bell = fq.Program(2, 2)
   bell.add(ops.H, 0)
   bell.add(ops.CX, (0, 1))
   bell.measure_all()

   backend = fq.simulator.Simulator(method="statevector")
   result = backend.run(bell, shots=1000).result()
   counts = result.get_counts()

The default implementation map covers FATQAT's built-in matrix gates. State
methods also support measurement, reset, and classical feedforward; a
``Barrier`` remains in the program but has no numerical effect. The simulator
executes the program as written and does not perform transpilation or routing.

Choose a method
---------------

``method`` is case-insensitive. ``SV`` and ``DM`` are short aliases; the
read-only :attr:`Simulator.method` property always returns the canonical name.
For a system with total Hilbert-space dimension ``D``:

.. list-table:: Simulation methods
   :header-rows: 1
   :widths: 18 23 32 27

   * - Method
     - Native result and shape
     - Non-unitary operations
     - Important limits
   * - ``statevector`` / ``SV``
     - ``statevector``, shape ``(D,)``
     - Reset and finite channels sample a trajectory
     - A stochastic final state represents one shot
   * - ``density_matrix`` / ``DM``
     - ``density_matrix``, shape ``(D, D)``
     - Reset and finite channels are applied exactly
     - Uses quadratically more state storage than ``statevector``
   * - ``unitary``
     - ``unitary``, shape ``(D, D)``
     - Rejected
     - Rejects measurement, conditions, counts, and ``initial_state``
   * - ``superop``
     - ``superop``, shape ``(D**2, D**2)``
     - Reset and finite channels are applied exactly
     - Rejects measurement, conditions, counts, and ``initial_state``

The public super-operator uses column-stacking vectorization of density
matrices:

.. code-block:: python

   rho_out = (
       superop @ rho_in.reshape(-1, order="F")
   ).reshape(rho_in.shape, order="F")

Column-stacking describes the mathematical vectorization of ``rho_in``, not
the NumPy memory layout of the returned matrix. For a noise-free program,
``superop`` equals ``numpy.kron(unitary.conj(), unitary)``. A unitary contains
``4**n`` complex entries for ``n`` qubits; a super-operator contains ``16**n``,
so operator methods are practical only for small programs.

Runtime and execution controls
------------------------------

``runtime`` selects the numerical engine when the backend is constructed and
is fixed for its lifetime. Runtime names are case-insensitive. Both runtimes
support all four methods with the same simulation semantics and
numerical-tolerance contract, but floating-point results and seeded samples are
not guaranteed to be bit-identical across runtimes.

The general and superconducting simulators default to ``"numba"``;
:class:`AtomArraySimulator` defaults to ``"numpy"``.

.. list-table:: Runtime comparison
   :header-rows: 1
   :widths: 26 37 37

   * - Feature
     - ``runtime="numpy"``
     - ``runtime="numba"``
   * - Execution
     - Runs the NumPy kernels directly, with no JIT warm-up
     - Lazily JIT-compiles numerical kernels; a first run can include
       compilation work
   * - Methods
     - ``statevector``, ``density_matrix``, ``unitary``, and ``superop``
     - The same four methods
   * - Kernel threads
     - Not available
     - Available for every method
   * - Shot parallelism
     - Eligible counts-only per-shot runs can use worker processes
     - The same process path, plus threads for compatible statevector
       counts-only plans
   * - Fusion
     - Not available
     - Available for ``density_matrix``, ``unitary``, and ``superop``

``simulation_config`` changes execution for one call to ``run()``. String
values in this dictionary are case-sensitive.

.. list-table:: Simulation controls
   :header-rows: 1
   :widths: 22 16 62

   * - Key
     - Default
     - Accepted values and effect
   * - ``seed``
     - ``None``
     - ``None`` or a non-negative ``int``; booleans are rejected. Seeds
       measurement, reset, channel, loss, and readout sampling for this run.
   * - ``shot_parallelism``
     - ``"auto"``
     - ``"auto"``, ``"serial"``, ``"threads"``, or ``"processes"``. Controls
       independent per-shot evolutions; it does not control terminal sampling
       from one shared final state.
   * - ``kernel_parallelism``
     - ``"auto"``
     - ``"auto"``, ``"serial"``, or ``"threads"``. Controls numerical work
       inside one evolution without reordering program operations. Threads
       require Numba.
   * - ``max_workers``
     - ``None``
     - ``None`` or a positive integer. Caps whichever parallel axis is chosen;
       it does not make an otherwise ineligible run parallel.
   * - ``fusion``
     - ``False``
     - A boolean that allows compatible adjacent operations to be combined.
       It is independent of parallelism and has the runtime support shown
       above.

FATQAT uses at most one parallel axis. Explicit shot threads or processes need
a shardable counts-only per-shot run, at least two shots, and at least two
workers; ordinary terminal sampling after one static evolution is not
eligible. An explicit mode raises an error when the runtime or program cannot
honor it rather than silently falling back. ``max_workers=1`` keeps automatic
selection serial but contradicts an explicit parallel request. See
:doc:`../guide/advanced` for the complete decision table and reproducibility
boundary.

Backend customization and lifetime
----------------------------------

The remaining constructor arguments customize the operations and noise known
to the backend:

.. list-table:: Backend customization
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Meaning
   * - ``implementation_map``
     - Maps operation families to matrix builders. ``None`` uses FATQAT's
       built-in matrix gate set.
   * - ``noise``
     - A :class:`~fatqat.NoiseModel` used by every run. ``None`` means ideal
       execution.
   * - ``channel_implementation_map``
     - Maps finite-channel descriptors to their numerical implementations.

The simulator copies the registration containers at construction, so later
additions or removals on the supplied objects do not alter the backend. Rule
and declaration objects are not deep-copied; treat them as immutable after
constructing the backend.

Reuse one backend for sequential runs and to retain its numerical caches. Do
not call ``run()`` concurrently on the same instance; create one backend per
concurrent caller.

Run inputs and results
----------------------

In addition to the program and ``simulation_config``, ``run()`` accepts these
principal inputs:

.. list-table:: Run inputs
   :header-rows: 1
   :widths: 22 18 60

   * - Argument
     - Default
     - Meaning
   * - ``shots``
     - ``1024``
     - Number of samples when counts or a stochastic final state is requested.
       Deterministic final-state-only and operator runs do not use it.
   * - ``resource_layout``
     - ``None``
     - Maps every program quantum reference to a device label. On
       :class:`Simulator`, the default assigns integer labels in declaration
       order. An explicit layout must be complete, one-to-one, and
       dimension-compatible.
   * - ``initial_state``
     - ``None``
     - Starts every shot from a supplied state instead of the all-zero state.
       Statevector accepts shape ``(D,)``; density matrix accepts ``(D,)`` or
       ``(D, D)``. Operator methods reject it.

Only the initial state's shape is validated. The caller is responsible for
normalization and, for density matrices, Hermiticity and positivity. Sampling
normalizes a distribution with positive probability mass but cannot sample a
state with none.

``result_config`` selects returned artifacts with two keys. Each value is
``True``, ``False``, or ``None``; omitted and ``None`` values use the defaults
below.

.. list-table:: Result fields
   :header-rows: 1
   :widths: 20 34 46

   * - Key
     - Default and meaning
     - Constraint
   * - ``counts``
     - Histogram of the final classical register; enabled when the program
       measures
     - Requires an integer ``shots > 0``
   * - ``final_state``
     - Method-native state or map; enabled when that artifact is deterministic
     - An explicitly requested stochastic final state requires ``shots == 1``

.. list-table:: Common defaults
   :header-rows: 1
   :widths: 55 45

   * - Run
     - Available data by default
   * - Measured state run
     - ``counts``
   * - Unmeasured deterministic state run
     - ``statevector`` or ``density_matrix``
   * - Unmeasured stochastic state run
     - No artifacts
   * - ``unitary`` or ``superop`` run
     - The computed map

For statevector simulation, reset and finite channels make the final state a
sampled trajectory. Atom loss makes both state methods stochastic. By
contrast, density-matrix reset and finite channels remain exact, so their
unmeasured result is deterministic. A result with both requested fields set to
``False`` is valid and contains no artifacts.

``final_state`` is a request name. The returned field is ``statevector``,
``density_matrix``, ``unitary``, or ``superop`` according to the selected
method. Inspect ``Result.available_data`` before reading optional fields.

``run()`` validates and lowers the program before returning an eager
:class:`~fatqat.Job`. Validation errors are raised directly. An error during
numerical execution or result assembly is stored on the job and re-raised by
:meth:`~fatqat.Job.result`. See :doc:`../guide/running-and-results` for result
accessors, count ordering, and state-axis metadata.

Noise models and parameter sweeps
---------------------------------

Matrix simulation has no physical timeline. It accepts operation-bound finite
channels and readout confusion, but rejects continuous background sources,
built-in damping descriptors in rate mode, and ``ThermalRelaxation`` until it
is converted with ``as_channels(duration)``.
:meth:`Simulator.check_noise_support` checks those source forms and the channel
map without executing a program. A method can impose a further restriction at
run time: ``unitary``, for example, rejects a finite channel even when the
backend knows how to build it. Statevector simulation samples channel branches;
density-matrix and super-operator simulation apply the exact channel. See
:doc:`noise` for supported descriptors and selector semantics.

:meth:`Simulator.run_sweep` binds every row of a complete object-keyed
parameter batch, calls ``run()`` for each row, and returns one eager job whose
result is an ordered ``list[Result]``. A supplied seed is reused for every row,
so sampled errors across rows are correlated. See
:doc:`../guide/parameters-and-sweeps` for accepted batch shapes.

API
---

.. autoclass:: Simulator
   :class-doc-from: both
