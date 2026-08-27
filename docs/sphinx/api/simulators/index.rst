Simulators
==========

.. currentmodule:: fatqat.simulator

FATQAT provides one general matrix simulator and three hardware-profile
simulators. They share the same execution and result API; the hardware-profile
classes add a native gate set, resource rules, and device-specific behavior.

Why hardware-profile simulators?
--------------------------------

:class:`Simulator` answers "what does this program compute?" The concrete
subclasses add selected features of a hardware family. They are useful when
developing compilers because they check native operations, layout,
connectivity, and other device rules. They also make characteristic
behavior—such as local coupling, dynamic pairing, or atom loss—visible in
simulation without claiming to reproduce a complete physical device.

The two superconducting simulators use a rectangular grid whose connectivity
is fixed when the simulator is created. Single-qubit gates are available
across the grid, while two-qubit gates follow the local couplings typical of
these processors. :class:`SCQubitIBMSimulator` checks IBM-style native gates,
grid layout, and nearest-neighbour ``CZ``. :class:`SCQubitGoogleSimulator`
checks native rotations and nearest-neighbour ``iSwap`` and ``CZ``. Both offer
an optional reference noise profile.

:class:`AtomArraySimulator` has no fixed connectivity. ``Pair`` and ``Unpair``
change which atoms may interact as the program runs; the simulator also checks
explicit loading, empty sites, and atom loss.

These subclasses are intended for compiler development and program validation.
They do not transpile or route programs, and they are not high-fidelity models
of specific processors. Use the :doc:`transmon <../pulse-emulator>` or
:doc:`neutral-atom <../atom-emulators>` emulators when pulse timing or
Hamiltonian evolution matters.

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
