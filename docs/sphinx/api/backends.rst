Backends (``fq.backends``)
==========================

A backend validates and executes a :py:class:`~fatqat.Program`. Normal applications use a
backend through :py:meth:`run <~fatqat.backends.SimulatorBackend.run>` and read the resulting :py:class:`~fatqat.Result`; they do not
need to customize its implementation map. For custom matrix rules or
device-specific maps, see :doc:`experimental`.

General-purpose simulator
-------------------------

:py:class:`~fatqat.backends.SimulatorBackend` (``method="statevector", noise=None``)

- ``method="statevector"`` is the default pure-state simulator. ``"SV"``
  is an accepted alias.
- ``method="density_matrix"`` simulates exact mixed states. ``"DM"`` is
  an accepted alias.
- Pass a :py:class:`~fatqat.NoiseModel` with ``noise=...`` to run the same program with
  channel or readout noise.

Run a program
-------------

:py:meth:`run <~fatqat.backends.SimulatorBackend.run>` (``program, *, shots=1024, result_config=None, seed=None``)

- ``shots`` controls how many samples are collected when counts are
  requested.
- ``result_config`` selects ``counts`` and the method’s native state field.
- ``seed`` makes sampled results reproducible.
- ``run(...)`` returns a :py:class:`~fatqat.Job`. Call ``job.result()`` to obtain the
  :py:class:`~fatqat.Result`. Its status and failure lifecycle controls are described in
  :doc:`experimental` as an evolving API.

See :doc:`../guide/running-and-results` for complete counts, statevector,
and density-matrix examples. The optional parallel-shot settings are
documented in :doc:`../guide/advanced`.

Constrained simulated targets
-----------------------------

:py:class:`~fatqat.backends.SCQubitIBMSimulator`,
:py:class:`~fatqat.backends.SCQubitGoogleSimulator`, and
:py:class:`~fatqat.backends.AtomGridSimulator` are optional
simulated targets with fixed native-gate and connectivity constraints. Use
them when those constraints are part of an experiment or test, not as the
default backend for a first program.

Compiler-facing target information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These backends execute programs already expressed in their native gate sets.
They do **not** decompose generic gates, route non-neighbouring operations,
schedule operations, or choose a placement. A compiler should query the
backend's :py:attr:`implementation_map
<fatqat.backends.SCQubitIBMSimulator.implementation_map>` rather than
hard-code a target:

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as op

   target = fq.backends.SCQubitIBMSimulator().implementation_map
   native_families = target.supported_operations()
   cz_edges = target.device_operands_for(op.CZ)

   assert target.supports(op.SX)
   assert not target.device_operands_for(op.SX)  # uniform support
   assert target.supports(op.CZ, device_operands=(0, 1))
   assert not target.supports(op.CZ, device_operands=(0, 5))

.. list-table:: Interpreting an implementation map
   :header-rows: 1
   :widths: 40 60

   * - Result
     - Meaning for a compiler
   * - ``not target.supports(op)``
     - The operation family is not native and must be rewritten before
       emission.
   * - ``target.supports(op)`` and an empty
       ``target.device_operands_for(op)``
     - The operation is native on every target of the correct arity.
   * - A non-empty ``target.device_operands_for(op)``
     - The operation is native only on the returned ordered device-site
       tuples.

Measurement and :py:data:`~fatqat.operations.Reset` are accepted on valid
qubits independently of these native-unitary maps.

Configurable superconducting grid targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The superconducting targets accept ``grid_size=(rows, cols)`` and default to
the following 4 by 4 row-major layout:

.. code-block:: text

    0   1   2   3
    4   5   6   7
    8   9  10  11
    12 13 14 15

Two-qubit gates are legal only on horizontal or vertical neighbours. Both
directions of each edge are registered, giving 48 directed tuples on the
default shape. Thus
``(0, 1)`` and ``(1, 0)`` are legal but ``(0, 5)`` is not. A plain program
maps qubits in declaration order to sites ``0`` onward. One
:py:class:`~fatqat.GridRegister` may bind top-left in row-major order, but
cannot be combined with another quantum register. For example,
``SCQubitIBMSimulator(grid_size=(2, 3))`` has six sites and the corresponding
2 by 3 nearest-neighbour topology.

.. list-table:: Native superconducting gate sets
   :header-rows: 1
   :widths: 28 40 32

   * - Backend
     - Uniform native gates
     - Neighbour-only native gates
   * - :py:class:`~fatqat.backends.SCQubitIBMSimulator`
     - :py:data:`~fatqat.operations.X`,
       :py:data:`~fatqat.operations.SX`, and
       :py:class:`~fatqat.operations.RZ`
     - :py:data:`~fatqat.operations.CZ`
   * - :py:class:`~fatqat.backends.SCQubitGoogleSimulator`
     - :py:class:`~fatqat.operations.RX`,
       :py:class:`~fatqat.operations.RY`, and
       :py:class:`~fatqat.operations.RZ`
     - :py:data:`~fatqat.operations.iSwap` and
       :py:data:`~fatqat.operations.CZ`

:py:data:`~fatqat.operations.CX` is not accepted directly by either target,
even on adjacent sites. A compiler may emit a replacement only after applying
a tested decomposition for its selected basis and checking every emitted
two-qubit operation against the implementation map.

Both targets are ideal by default. To enable their calibration-derived noise:

.. code-block:: python

   import fatqat as fq

   Sim = fq.backends.SCQubitIBMSimulator
   backend = Sim(noise=Sim.default_noise_model())

The common calibration values are ``T1 = 60 us``, ``T2 = 48 us``, and
asymmetric readout probabilities ``P(report 1 | true 0) = 0.02`` and
``P(report 0 | true 1) = 0.04``. Readout noise applies to every measurement.

.. list-table:: Superconducting calibration-derived noise
   :header-rows: 1
   :widths: 22 20 42 16

   * - Backend
     - Gate
     - Gate-time relaxation
     - Joint depolarizing noise
   * - IBM-style
     - :py:data:`~fatqat.operations.X`,
       :py:data:`~fatqat.operations.SX`
     - T1/T2-derived amplitude and phase damping; 20 ns
     - None
   * - IBM-style
     - :py:data:`~fatqat.operations.CZ`
     - The same relaxation on **each** participating qubit; 50 ns
     - ``p = 0.001``
   * - IBM-style
     - :py:class:`~fatqat.operations.RZ`
     - None; virtual gate
     - None
   * - Google-style
     - :py:class:`~fatqat.operations.RX`,
       :py:class:`~fatqat.operations.RY`,
       :py:class:`~fatqat.operations.RZ`
     - The same relaxation; 20 ns
     - None
   * - Google-style
     - :py:data:`~fatqat.operations.iSwap`
     - The same relaxation on each participating qubit; 30 ns
     - ``p = 0.001``
   * - Google-style
     - :py:data:`~fatqat.operations.CZ`
     - The same relaxation on **each** participating qubit; 50 ns
     - ``p = 0.001``

See :doc:`noise` for channel execution, readout semantics, and custom noise
models.

Neutral-atom grid target
~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~fatqat.backends.AtomGridSimulator` accepts a ``grid_size=(rows,
cols)`` layout (``(4, 5)`` by default). Its uniform native gates are
:py:class:`~fatqat.operations.RX`, :py:class:`~fatqat.operations.RY`, and
:py:class:`~fatqat.operations.RZ`; :py:data:`~fatqat.operations.CZ` is native
only on directed nearest-neighbour pairs. Query those pairs through
``implementation_map`` rather than deriving them from the shape.

Every atom-grid program must begin with an unconditional
:py:class:`~fatqat.operations.LoadAtoms` that fits the device. It loads the
top-left rectangle named by ``LoadAtoms(rows, cols)``. A later ``LoadAtoms``
is rejected; a gate or reset touching an unloaded site is a no-op. Measurement
remains valid on unloaded sites and reports the initial ``0`` in ideal
execution, subject to any supplied readout noise.

.. code-block:: python

   import fatqat as fq

   atoms = fq.GridRegister(2, 3, name="atoms")
   program = fq.Program([atoms])
   program.add(op.LoadAtoms(2, 3))
   program.add(op.RX(0.2), atoms.row(0))
   program.add(op.CZ, (atoms[0], atoms[3]))

   backend = fq.backends.AtomGridSimulator()  # 4 by 5 device

On that default device, the 2 by 3 frontend grid occupies labels ``0, 1, 2,
5, 6, 7``: its second logical row starts at device label ``5``, not ``3``.
Use device labels for connectivity checks, but program references or flat
indices for :py:class:`~fatqat.NoiseModel` selectors. This backend has no
calibration-derived default noise model; a supplied
:py:class:`~fatqat.NoiseModel` is a user simulation choice.

Detailed reference
------------------

Superconducting pulse backend
-----------------------------

:py:class:`~fatqat.backends.PulseBackend` is a separately configured,
physical three-level-transmon simulator. It does not replace the matrix
family's IBM- and Google-style superconducting targets above.

Load a data-only model snapshot with
:py:func:`~fatqat.backends.load_physics_model`, then load the matching,
identity-bound calibration using
:py:func:`~fatqat.backends.load_calibration_spec`. The only public factory
for the built-in model family is
:py:class:`~fatqat.backends.SCTransmonExchangeBuilder`; scheduler, pulse, and
QuTiP objects are not public API.

Snapshot frequencies are nominal 0-to-1 transition frequencies defining the
implicit per-subsystem resonant frames. Since this model currently uses
``Delta_i = 0``, changing a frequency alone does not numerically change its
simulated dynamics. A frame-explicit interpretation requires a new model
version.

The native operation set is ``RX``, ``RY``, virtual ``RZ``, ``iSwap``, and
oriented ``CZ`` on declared coupling edges. ``simulation_config`` accepts
``seed`` and serial execution controls plus ``schedule_mode="ASAP"`` or
``"ALAP"``. These private scheduling modes preserve program dependencies and
resource exclusivity; they do not expose a hardware schedule.

Each native operation resolves to its physical realization through a
:py:class:`~fatqat.backends.PulseImplementationMap`, passed as
``pulse_implementation_map=``. Omit it to use
:py:func:`~fatqat.backends.default_superconducting_pulse_implementation_map`;
supply a copy of that default map with one gate replaced to change how a
gate is physically realized without touching calibration data or
subclassing :py:class:`~fatqat.backends.PulseBackend`. See
:doc:`experimental` for the full customization workflow.

Request counts with ``result_config={"counts": True}`` and the final physical
density matrix with ``result_config={"final_state": True}``. That density
matrix includes every transmon in the selected model, even if the program did
not address it, and has shape ``(3**m, 3**m)`` for ``m`` model subsystems.

:py:meth:`~fatqat.backends.PulseBackend.propagator` returns the coherent
full-model program propagator as a complex NumPy array. It includes terminal
virtual-frame updates by default; ``apply_final_frame=False`` exposes the
Hamiltonian-generated evolution before only that terminal transformation.
The returned array follows the model's near-resonant rotating-frame
convention. Its virtual-Z representation can differ from a conventional
qubit ``RZ`` matrix by a global phase, so comparisons with ideal circuit
unitaries should be phase-invariant.
Programs with measurement, reset, or classical conditions are rejected.
Bound collapse terms are rejected when the program has nonzero elapsed pulse
evolution. Rate-based noise has no effect on a frame-only, zero-duration
program because no time elapses.

Pulse execution supports :py:class:`~fatqat.noise.ThermalRelaxation` (T1/T2)
and rate-mode :py:class:`~fatqat.noise.AmplitudeDamping`/
:py:class:`~fatqat.noise.PhaseDamping` as always-on noise, plus the same
damping descriptors in operation-scoped probability or rate mode. Both
activation scopes use :py:meth:`fatqat.NoiseModel.add_channel`; omitting
``operation`` means always-on. Readout confusion is also supported. Coherent
ZZ and every other channel type are not supported by this backend in v0.1.

.. autoclass:: fatqat.backends.SimulatorBackend
   :members:
   :show-inheritance:

.. autoclass:: fatqat.backends.SCQubitIBMSimulator
   :members:
   :show-inheritance:

.. autoclass:: fatqat.backends.SCQubitGoogleSimulator
   :members:
   :show-inheritance:

.. autoclass:: fatqat.backends.AtomGridSimulator
   :members:
   :show-inheritance:

.. autoclass:: fatqat.backends.PulseBackend
   :members:

.. autoclass:: fatqat.backends.SCTransmonExchangeBuilder
   :members:

.. autofunction:: fatqat.backends.load_physics_model

.. autofunction:: fatqat.backends.load_calibration_spec
