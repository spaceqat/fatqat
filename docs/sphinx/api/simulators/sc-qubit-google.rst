SCQubitGoogleSimulator
======================

.. currentmodule:: fatqat.simulator

:class:`SCQubitGoogleSimulator` applies the :class:`Simulator` execution model
to a configurable Google-style superconducting grid. Use it to test native
rotation gates and nearest-neighbour ``iSwap``/``CZ`` programs. It is a
hardware-profile simulator for compiler development and program validation,
not a model of a named Google processor: it does not transpile, route,
schedule, or reproduce hardware calibration data.

.. list-table:: Hardware profile
   :header-rows: 1
   :widths: 28 72

   * - Property
     - Value
   * - Default device
     - ``grid_size=(4, 4)``; 16 row-major qubits
   * - Uniform native gates
     - :class:`fatqat.operations.RX`, :class:`fatqat.operations.RY`,
       :class:`fatqat.operations.RZ`
   * - Connected native gates
     - :data:`fatqat.operations.iSwap` and :data:`fatqat.operations.CZ` on
       horizontal or vertical neighbours
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

Device labels are row-major. On the default grid they are:

.. code-block:: text

    0   1   2   3
    4   5   6   7
    8   9  10  11
   12  13  14  15

Both operand orders of every grid edge are registered. Thus ``iSwap(0, 1)``
and ``iSwap(1, 0)`` are legal, while ``iSwap(0, 5)`` is not. ``CZ`` follows
the same connectivity rule. The topology is rebuilt for any positive
``grid_size=(rows, columns)``.

With the automatic layout, an ordinary program maps its qubits to device labels
``0, 1, ...`` in declaration order. One :class:`~fatqat.GridRegister` instead
maps into the device's top-left corner while preserving row and column
coordinates. In this automatic mode, it must be the program's only quantum
register and must fit along both device axes. An explicit, complete
:class:`~fatqat.ResourceLayout` can place program references differently.
Capacity and the two-dimensional qubit requirement still apply.

The backend does not decompose non-native operations. In particular,
:data:`fatqat.operations.CX` and :data:`fatqat.operations.SX` are rejected.
Use :attr:`SCQubitGoogleSimulator.implementation_map` as the compiler-facing
description of the gate set and legal operand tuples:

.. code-block:: python

   import fatqat as fq
   import fatqat.operations as ops

   backend = fq.simulator.SCQubitGoogleSimulator(grid_size=(2, 3))
   native = backend.implementation_map

   assert native.supports(ops.RY)
   assert native.supports(ops.iSwap, device_operands=(1, 4))
   assert not native.supports(ops.iSwap, device_operands=(0, 4))

``device_operands_for(operation)`` returns an empty set for a uniformly
available gate and explicit ordered tuples for a connectivity-limited gate.
The property returns a copy, so changing it does not change the backend.

Calibration-derived noise
-------------------------

The simulator remains ideal unless a noise model is supplied. To use the
built-in profile, request it explicitly:

.. code-block:: python

   import fatqat as fq

   Sim = fq.simulator.SCQubitGoogleSimulator
   backend = Sim(noise=Sim.default_noise_model())

The profile uses ``T1 = 60 us`` and ``T2 = 48 us``. Each call returns a fresh
:class:`~fatqat.NoiseModel` that can be inspected or extended.

.. list-table:: Built-in profile
   :header-rows: 1
   :widths: 23 23 54

   * - Operation
     - Duration
     - Noise
   * - ``RX``, ``RY``, ``RZ``
     - 20 ns
     - T1/T2-derived amplitude and phase damping
   * - ``iSwap``
     - 30 ns
     - Relaxation on each qubit, followed by joint depolarizing noise with
       ``p = 0.001``
   * - ``CZ``
     - 50 ns
     - Relaxation on each qubit, followed by joint depolarizing noise with
       ``p = 0.001``
   * - Measurement
     - —
     - ``P(report 1 | true 0) = 0.02`` and
       ``P(report 0 | true 1) = 0.04``

Unlike the IBM-style profile, ``RZ`` is a physical, noisy 20 ns rotation in
this profile. See :doc:`../noise` for method-dependent channel execution.

API
---

The inherited :meth:`Simulator.run`, :meth:`Simulator.run_sweep`, and
:meth:`Simulator.check_noise_support` methods have the same arguments and
result rules as the general simulator.

.. autoclass:: SCQubitGoogleSimulator
   :class-doc-from: both
