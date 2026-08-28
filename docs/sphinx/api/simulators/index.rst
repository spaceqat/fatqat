Simulators
==========

.. currentmodule:: fatqat.simulator

FATQAT provides a general matrix simulator and three hardware profiles. They
use the same run and result API. Choose :class:`Simulator` for unrestricted
gate-level work, or a profile when the program must obey a native gate set,
layout, or connectivity rule.

The superconducting profiles use a fixed rectangular grid.
:class:`SCQubitIBMSimulator` accepts IBM-style native gates and
nearest-neighbour ``CZ``. :class:`SCQubitGoogleSimulator` accepts native
rotations and nearest-neighbour ``iSwap`` and ``CZ``. Both offer an optional
reference noise model.

:class:`AtomArraySimulator` has no fixed connectivity. ``Pair`` and ``Unpair``
change which atoms can interact, while ``Put`` and ``Loss`` control occupancy.

The profiles validate the program as written: they do not transpile or route
it, and they do not reproduce a named processor. Use the
:doc:`pulse emulators <../emulators/index>` when timing or Hamiltonian
evolution matters.

.. list-table:: Choose a simulator
   :header-rows: 1
   :widths: 28 37 35

   * - Class
     - Use it for
     - Main constraint
   * - :class:`Simulator`
     - General circuit simulation and custom matrix implementations
     - No device topology
   * - :class:`SCQubitIBMSimulator`
     - IBM-style native-gate and grid experiments
     - ``X``, ``SX``, ``RZ``; nearest-neighbour ``CZ``
   * - :class:`SCQubitGoogleSimulator`
     - Google-style native-gate and grid experiments
     - ``RX``, ``RY``, ``RZ``; nearest-neighbour ``iSwap`` and ``CZ``
   * - :class:`AtomArraySimulator`
     - Neutral-atom occupancy, loss, and dynamic connectivity
     - ``RX``, ``RY``, ``RZ``, and paired ``CZ``

.. toctree::
   :maxdepth: 1

   ../simulator
   sc-qubit-ibm
   sc-qubit-google
   atom-array
