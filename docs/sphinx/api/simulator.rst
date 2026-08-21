Simulator (``fq.simulator``)

A backend validates and executes a :py:class:`~fatqat.Program`. Normal applications use a
backend through :py:meth:`run <~fatqat.simulator.Simulator.run>` and read the resulting :py:class:`~fatqat.Result`; they do not
need to customize its implementation map. For custom matrix rules or
device-specific maps, see :doc:`experimental`.

This page covers gate-level simulation, where operations are applied as
finite matrices or Kraus maps. For pulse-resolved physical models, see the
:doc:`superconducting reference <pulse-emulator>` or the
:doc:`three-level and two-level neutral-atom reference <atom-emulators>`.

General-purpose simulator
-------------------------

:py:class:`~fatqat.simulator.Simulator` (``method="statevector", noise=None``)

- ``method="statevector"`` is the default pure-state simulator. ``"SV"``
  is an accepted alias.
- ``method="density_matrix"`` simulates exact mixed states. ``"DM"`` is
  an accepted alias.
- ``method="unitary"`` and ``method="superop"`` return the program's *map*
  instead of a state under it. See :ref:`operator-methods` below.
- Pass a :py:class:`~fatqat.NoiseModel` with ``noise=...`` to run the same program with
  channel or readout-confusion noise.

.. _operator-methods:

Operator methods
----------------

``method="unitary"`` returns the program's ``(D, D)`` unitary and
``method="superop"`` its ``(D**2, D**2)`` super-operator, where ``D`` is the
product of the subsystem dimensions. Both run one deterministic pass with no
shots and no sampling, so ``final_state`` defaults to ``True`` and ``shots``
is ignored.

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as op

   bell = fq.Program(2)
   bell.add(op.H, 0)
   bell.add(op.CX, (0, 1))

   unitary = fq.simulator.Simulator("unitary").run(bell).result().get_unitary()
   superop = fq.simulator.Simulator("superop").run(bell).result().get_superop()

``unitary[:, 0]`` is the statevector the same program prepares. The
super-operator is row-major vectorized, matching what
:py:meth:`~fatqat.Result.get_density_matrix` output flattens into:
``superop @ rho.reshape(-1)`` reshaped back to ``(D, D)`` is the program
applied to ``rho``. For a noise-free program it equals
``numpy.kron(unitary, unitary.conj())``.

.. list-table:: What each operator method accepts
   :header-rows: 1
   :widths: 34 22 22 22

   * - Program feature
     - ``statevector`` / ``density_matrix``
     - ``unitary``
     - ``superop``
   * - Gates
     - yes
     - yes
     - yes
   * - :py:data:`~fatqat.operations.Reset`, channel noise
     - yes
     - **rejected**
     - yes (exact channels)
   * - Measurement, feedforward, ``counts``
     - yes
     - **rejected**
     - **rejected**

Rejections raise :py:class:`~fatqat.errors.BackendValidationError` directly
from ``run()``, before any execution. Memory grows as ``4**n`` for ``unitary``
and ``16**n`` for ``superop``, so keep super-operator circuits small.

Run a program
-------------

:py:meth:`run <~fatqat.simulator.Simulator.run>` (``program, *, shots=1024, simulation_config=None, result_config=None``)

- ``shots`` controls how many samples are collected when counts are
  requested.
- ``result_config`` selects ``counts`` and the method’s native state field.
- A ``seed`` inside ``simulation_config`` makes sampled results reproducible.
- ``run(...)`` returns a :py:class:`~fatqat.Job`. Call ``job.result()`` to obtain the
  :py:class:`~fatqat.Result`. Its status and failure lifecycle controls are described in
  :doc:`experimental` as an evolving API.

:py:attr:`method <fatqat.simulator.Simulator.method>` reports the canonical
name of the state representation the backend runs — ``"statevector"``,
``"density_matrix"``, ``"unitary"``, or ``"superop"``. Aliases are normalized
away, so ``Simulator(method="SV").method`` is ``"statevector"``. It is the same
string that appears as ``result.metadata["method"]`` and as the result's native
state field, and reading it runs nothing, so a tool that supports only some
representations can check it as a precondition rather than discovering the
mismatch through a missing result field.

See :doc:`../guide/running-and-results` for complete counts, statevector,
and density-matrix examples. The optional parallel-shot settings are
documented in :doc:`../guide/advanced`.

:py:meth:`run_sweep <~fatqat.simulator.Simulator.run_sweep>` accepts an
object-keyed parameter batch and returns one eager job whose payload is an
ordered ``list[Result]``. See :doc:`../guide/parameters-and-sweeps` for shapes,
binding rules, and repeated-run semantics.

Constrained simulated targets
-----------------------------

:py:class:`~fatqat.simulator.SCQubitIBMSimulator`,
:py:class:`~fatqat.simulator.SCQubitGoogleSimulator`, and
:py:class:`~fatqat.simulator.AtomArraySimulator` are optional
simulated targets with fixed native-gate and connectivity constraints. Use
them when those constraints are part of an experiment or test, not as the
default backend for a first program.

Compiler-facing target information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These backends execute programs already expressed in their native gate sets.
They do **not** decompose generic gates, route non-neighbouring operations,
schedule operations, or choose a placement. A compiler should query the
backend's :py:attr:`implementation_map
<fatqat.simulator.SCQubitIBMSimulator.implementation_map>` rather than
hard-code a target:

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as op

   target = fq.simulator.SCQubitIBMSimulator().implementation_map
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
   * - :py:class:`~fatqat.simulator.SCQubitIBMSimulator`
     - :py:data:`~fatqat.operations.X`,
       :py:data:`~fatqat.operations.SX`, and
       :py:class:`~fatqat.operations.RZ`
     - :py:data:`~fatqat.operations.CZ`
   * - :py:class:`~fatqat.simulator.SCQubitGoogleSimulator`
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

   Sim = fq.simulator.SCQubitIBMSimulator
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

Neutral-atom array target
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~fatqat.simulator.AtomArraySimulator` accepts an optional
``num_sites`` capacity (unbounded by default): the number of trap sites, each
holding at most one atom, with no fixed topology. Leave it unset for no
capacity limit, or pass a positive integer to model a device of fixed size. Its uniform native gates are
:py:class:`~fatqat.operations.RX`, :py:class:`~fatqat.operations.RY`, and
:py:class:`~fatqat.operations.RZ`; :py:data:`~fatqat.operations.CZ` is native on
any connected pair. Two-qubit-gate legality follows a dynamic connectivity
graph (:py:class:`~fatqat.connectivity.AtomConnectivity`): a ``CZ`` on a pair
that is not currently paired is silently dropped, like a gate on an empty site.

Every site starts empty. :py:data:`~fatqat.operations.Put` loads a fresh
``|0⟩`` atom into its targets; :py:data:`~fatqat.operations.Pair` connects two
atoms and :py:data:`~fatqat.operations.Unpair` disconnects them. A gate whose
target is never loaded by any ``Put`` is dropped; a program that uses neither
``Put`` nor atom loss keeps every declared qubit present. A lost or empty site
measures the erasure digit ``2``. Attach
:py:class:`~fatqat.noise.Loss` to a gate to eject atoms per shot, to
``Put`` to model imperfect loading, or to ``Pair``/``Unpair`` to model movement
cost.

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as op

   atoms = fq.QuantumRegister(2, name="atoms")
   program = fq.Program([atoms])
   program.add(op.Put, (0, 1))     # load both atoms
   program.add(op.Pair, (0, 1))    # connect them so CZ is legal
   program.add(op.RX(0.2), 0)
   program.add(op.CZ, (0, 1))

   backend = fq.simulator.AtomArraySimulator()  # unbounded capacity by default

Registers map to device labels in declaration order; a
:py:class:`~fatqat.GridRegister`, if passed, is treated as a plain flat register
(no coordinates). Use program references or flat indices for
:py:class:`~fatqat.NoiseModel` selectors. This backend has no
calibration-derived default noise model; a supplied
:py:class:`~fatqat.NoiseModel` is a user simulation choice.

Detailed reference
------------------

.. autoclass:: fatqat.simulator.Simulator
   :members:
   :show-inheritance:

.. autoclass:: fatqat.simulator.SCQubitIBMSimulator
   :members:
   :show-inheritance:

.. autoclass:: fatqat.simulator.SCQubitGoogleSimulator
   :members:
   :show-inheritance:

.. autoclass:: fatqat.simulator.AtomArraySimulator
   :members:
   :show-inheritance:

.. autoclass:: fatqat.connectivity.AtomConnectivity
   :members:
   :show-inheritance:
