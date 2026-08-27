AtomArraySimulator
==================

.. currentmodule:: fatqat.simulator

:class:`AtomArraySimulator` adds neutral-atom occupancy, loss, and dynamic
pairing to the :class:`Simulator` execution model. It is useful for testing
programs and compiler passes against these constraints. It is not a
Hamiltonian or transport model; use the :doc:`neutral-atom emulators
<../atom-emulators>` when pulse timing and physical interactions matter.

.. list-table:: Hardware profile
   :header-rows: 1
   :widths: 28 72

   * - Property
     - Value
   * - Capacity
     - Unbounded by default; ``num_sites`` sets a positive fixed limit
   * - Native gates
     - :class:`fatqat.operations.RX`, :class:`fatqat.operations.RY`,
       :class:`fatqat.operations.RZ`, and :data:`fatqat.operations.CZ`
   * - Connectivity
     - No fixed topology; ``CZ`` is legal only while its two atoms are paired
   * - Dimensions
     - Qubits only
   * - Methods
     - All :class:`Simulator` methods for programs without ``Put`` or loss;
       the atom lifecycle requires ``statevector`` or ``density_matrix``
   * - Runtime
     - ``numpy`` by default; ``numba`` is also supported
   * - Noise
     - Ideal by default; no calibration-derived model

Capacity, mapping, and native operations
----------------------------------------

``num_sites=None`` places no capacity limit. A positive value rejects programs
that declare more quantum subsystems than available sites. Registers map to
integer device labels in declaration order; a :class:`~fatqat.GridRegister`
is flattened and its coordinates have no physical meaning on this backend.

The implementation map contains ``RX``, ``RY``, ``RZ``, and ``CZ``. It does
not decompose other gates, so a program containing ``CX`` is rejected even if
the atoms are paired. A compiler can inspect the fixed map through
:attr:`AtomArraySimulator.implementation_map`. The map reports ``CZ`` as
uniformly available because it cannot express a pairing graph that changes
during the program; pairing legality is enforced separately.

Pairing is program state, not a static device edge. An unconditional
:data:`fatqat.operations.Pair` connects two sites and
:data:`fatqat.operations.Unpair` disconnects them. A ``CZ`` on an unpaired
pair is a validation error. Pairing operations do not change the quantum
state, but noise attached to them is still applied; conditional ``Pair`` and
``Unpair`` operations are rejected.

Occupancy and loss
------------------

Occupancy is tracked separately for every shot. Its initial value depends on
whether the program uses the atom lifecycle:

.. list-table:: Occupancy rules
   :header-rows: 1
   :widths: 37 63

   * - Program
     - Initial occupancy and behavior
   * - No ``Put`` and no matching :class:`fatqat.noise.Loss` source
     - Every declared site is present. The program behaves like the general
       simulator, apart from the native gate and pairing rules.
   * - Contains ``Put`` or lowers a matching atom-loss source
     - Every site starts empty. :data:`fatqat.operations.Put` loads a fresh
       ``|0>`` atom at its targets.

``Put`` on an occupied site is a no-op. A gate on an empty or previously lost
site is also a no-op for that shot. If a program contains ``Put``, gates and
reset on a site that is never the target of any ``Put`` are omitted because
that site can never become occupied. Measurement remains so it can report an
erasure, and pairing operations still update connectivity. A later ``Put`` can
refill a lost site.

:class:`fatqat.noise.Loss` can be attached to gates to eject their targets,
to ``Put`` to model failed loading, or to ``Pair``/``Unpair`` to model movement
loss. A registered loss source changes initial occupancy only when its selector
matches an operation and the source is lowered into the execution plan. This
is the only gate-level simulator that accepts ``Loss``. A missing atom at an
otherwise valid paired ``CZ`` makes that gate a per-shot no-op; it is not the
same as the compile-time error for an unpaired ``CZ``.

Measurement of an empty site reports the erasure digit ``2``. Erasure bypasses
readout-confusion noise because there is no occupied qubit to read. Atom loss
makes the final state stochastic, so ``final_state=True`` requires
``shots == 1``. A measured lossy run returns counts by default but not an
arbitrary trajectory's final state.

Example
-------

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as ops

   program = fq.Program(2, 2)
   program.add(ops.Put, (0, 1))
   program.add(ops.Pair, (0, 1))
   program.add(ops.RY(0.4), 0)
   program.add(ops.CZ, (0, 1))
   program.measure_all()

   backend = fq.simulator.AtomArraySimulator(num_sites=2)
   counts = backend.run(program, shots=1000).result().get_counts()

The atom lifecycle cannot be represented by ``unitary`` or ``superop`` because
occupancy is state outside the quantum matrix. Pairing alone is allowed by
operator methods: it changes which native two-qubit operations are legal but
does not create occupancy state.

API
---

The inherited :meth:`Simulator.run`, :meth:`Simulator.run_sweep`, and
:meth:`Simulator.check_noise_support` methods have the same arguments and
result rules as the general simulator.

.. autoclass:: AtomArraySimulator
   :class-doc-from: both
