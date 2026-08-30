AtomArraySimulator
==================

.. currentmodule:: fatqat.simulator

:class:`AtomArraySimulator` adds neutral-atom occupancy, loss, and dynamic
pairing to the :class:`Simulator` execution model. Use it to check a program
against those constraints. It is not a Hamiltonian or transport model; use
the :doc:`neutral-atom emulator
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
     - Ideal by default; no built-in reference model

Capacity, mapping, and native operations
----------------------------------------

``num_sites=None`` places no capacity limit. A positive value rejects programs
that declare more quantum subsystems than available sites. Registers map to
integer device labels in declaration order; a :class:`~fatqat.GridRegister`
is flattened and its coordinates have no physical meaning on this backend.

The native operations are ``RX``, ``RY``, ``RZ``, and ``CZ``. The simulator
does not decompose other gates, so ``CX`` is rejected even when the atoms are
paired. :attr:`AtomArraySimulator.implementation_map` lists the native gate
set; the program's current ``Pair`` state determines whether a particular
``CZ`` is allowed.

Pairing changes as the program runs. An unconditional
:data:`fatqat.operations.Pair` connects two sites and
:data:`fatqat.operations.Unpair` disconnects them. A ``CZ`` on an unpaired
pair is rejected. Pairing operations do not change the quantum state, though
noise attached to them still applies. Conditional ``Pair`` and ``Unpair`` are
not supported.

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
   * - Contains ``Put``, or an operation matches a
       :class:`fatqat.noise.Loss` source, even when ``p=0``
     - Every site starts empty. :data:`fatqat.operations.Put` loads a fresh
       ``|0>`` atom at its targets.

``Put`` on an occupied site has no effect. A gate or reset on an empty or
previously lost site likewise has no effect for that shot. Measurement still
reports an erasure, pairing still changes connectivity, and a later ``Put``
can refill a lost site.

:class:`fatqat.noise.Loss` can eject gate targets, make ``Put`` fail, or model
loss during ``Pair`` and ``Unpair``. It affects a run only when its selector
matches an operation; a matching source with ``p=0`` still activates explicit
occupancy. This is the only gate-level simulator that accepts ``Loss``. A
missing atom makes an otherwise valid paired ``CZ`` do nothing for that shot;
an unpaired ``CZ`` is rejected before execution.

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

:attr:`Simulator.method`, :attr:`AtomArraySimulator.implementation_map`,
:meth:`Simulator.run`, :meth:`Simulator.run_sweep`, and
:meth:`Simulator.validate_noise_model` follow the general Simulator API and are
included below for a complete class reference.

.. autoclass:: AtomArraySimulator
   :class-doc-from: both
