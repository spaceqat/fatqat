SCQubitIBMSimulator
===================

.. currentmodule:: fatqat.simulator

:class:`SCQubitIBMSimulator` applies the :class:`Simulator` execution model to
a configurable IBM-style superconducting grid. Use it when native gates,
capacity, and nearest-neighbour connectivity are part of the experiment. It is
a hardware-profile simulator for compiler development and program validation,
not a model of an IBM device: it does not transpile, route, schedule, or
reproduce a named processor.

.. list-table:: Hardware profile
   :header-rows: 1
   :widths: 28 72

   * - Property
     - Value
   * - Default device
     - ``grid_size=(4, 4)``; 16 row-major qubits
   * - Uniform native gates
     - :data:`fatqat.operations.X`, :data:`fatqat.operations.SX`,
       :class:`fatqat.operations.RZ`
   * - Connected native gate
     - :data:`fatqat.operations.CZ` on horizontal or vertical neighbours
   * - Other built-in operations
     - Measurement and :data:`fatqat.operations.Reset` bypass the native gate
       map; method-specific restrictions still apply
   * - Methods
     - All methods supported by :class:`Simulator`; default ``statevector``
   * - Runtime
     - ``numba`` by default; ``numpy`` is also supported
   * - Noise
     - Ideal by default; calibration-derived model available explicitly

Native gates and layout
-----------------------

The default row-major numbering is:

.. code-block:: text

    0   1   2   3
    4   5   6   7
    8   9  10  11
   12  13  14  15

Both operand orders of every grid edge are legal, so ``CZ(0, 1)`` and
``CZ(1, 0)`` are accepted while ``CZ(0, 5)`` is not. The same rule is rebuilt
for any positive ``grid_size=(rows, columns)``.

With the automatic layout, an ordinary program maps its qubits to device labels
``0, 1, ...`` in declaration order. A program containing one
:class:`~fatqat.GridRegister` maps that register into the device's top-left
corner while preserving its row and column coordinates. In this automatic
mode, the grid register must be the program's only quantum register and must
fit along both device axes. An explicit, complete
:class:`~fatqat.ResourceLayout` can place program references differently.
Capacity and the two-dimensional qubit requirement still apply.

The backend executes only programs already written in its native gate set.
For example, :data:`fatqat.operations.CX` is rejected even on neighbouring
qubits. A compiler can inspect :attr:`SCQubitIBMSimulator.implementation_map`
instead of hard-coding these rules:

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as ops

   backend = fq.simulator.SCQubitIBMSimulator(grid_size=(2, 3))
   native = backend.implementation_map

   assert native.supports(ops.SX)
   assert native.supports(ops.CZ, device_operands=(0, 1))
   assert not native.supports(ops.CZ, device_operands=(0, 4))

``device_operands_for(operation)`` returns an empty set for a uniformly
available gate and explicit ordered tuples for a connectivity-limited gate.
The property returns a copy, so changing it does not change the backend.

Calibration-derived noise
-------------------------

The simulator remains ideal unless a noise model is supplied. To use the
built-in profile, request it explicitly:

.. code-block:: python

   import fatqat as fq

   Sim = fq.simulator.SCQubitIBMSimulator
   backend = Sim(noise=Sim.default_noise_model())

The profile uses ``T1 = 60 us`` and ``T2 = 48 us``. Each call returns a fresh
:class:`~fatqat.NoiseModel` that can be inspected or extended.

.. list-table:: Built-in profile
   :header-rows: 1
   :widths: 23 23 54

   * - Operation
     - Duration
     - Noise
   * - ``X``, ``SX``
     - 20 ns
     - T1/T2-derived amplitude and phase damping
   * - ``RZ``
     - 0 ns (virtual)
     - None
   * - ``CZ``
     - 50 ns
     - Relaxation on each qubit, followed by joint depolarizing noise with
       ``p = 0.001``
   * - Measurement
     - —
     - ``P(report 1 | true 0) = 0.02`` and
       ``P(report 0 | true 1) = 0.04``

See :doc:`../noise` for how the selected simulation method applies this model.

API
---

The inherited :meth:`Simulator.run`, :meth:`Simulator.run_sweep`, and
:meth:`Simulator.check_noise_support` methods have the same arguments and
result rules as the general simulator.

.. autoclass:: SCQubitIBMSimulator
   :class-doc-from: both
