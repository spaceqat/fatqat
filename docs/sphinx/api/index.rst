API reference
=============

This reference documents the supported, application-facing objects in
``fatqat``. Start with the :doc:`../guide/quickstart` and use these pages
when you need an exact signature or return value. Pulse emulation is split
between the transmon reference and a dedicated neutral-atom reference.

Physics emulator capabilities
-----------------------------

Gate-authored and direct-control programming are independent capabilities.
A backend can support either or both:

.. list-table:: Physics systems
   :header-rows: 1
   :widths: 20 22 18 18 22

   * - System
     - Physical basis
     - Gate-authored
     - Direct control
     - Gate implementation map
   * - :py:class:`~fatqat.emulator.TransmonEmulator`
     - Three-level transmons
     - Yes
     - Yes
     - Public optional ``gate_implementation_map=``; built-in default
   * - :py:class:`~fatqat.emulator.Atom3LevelEmulator`
     - ``|0>, |1>, |r>``
     - Yes
     - Yes
     - Public optional ``gate_implementation_map=``; built-in default
   * - :py:class:`~fatqat.emulator.Atom2LevelEmulator`
     - ``|g>, |r>``
     - Custom rules only
     - Yes, global
     - Public optional ``gate_implementation_map=``; empty built-in default

All three physics-emulator ``run()`` methods construct the model family's
fixed product initial state and do not accept ``initial_state``. Transmon and
three-level atom runs start in ``|0>`` on every subsystem; the two-level atom
run starts in ``|g>`` at every site. By contrast,
:py:meth:`~fatqat.simulator.Simulator.run` accepts ``initial_state`` for its
statevector and density-matrix methods. Each emulator class page documents its
complete run and coherent-propagator constraints.

``PulseImplementationMap`` is the public map value type for all three
families. Their constructors name the capability
``gate_implementation_map`` because direct ``PulseOperation`` controls bypass
gate realization entirely. Calibration documents are inputs to the standard
map builders, never emulator constructor arguments.

The reference deliberately focuses on building and running programs. It
does not expose simulator-engine, compiler, or backend-author machinery as
part of the normal user workflow.

Each page begins with a short task-oriented overview, followed by the
generated signatures, attributes, and methods for that supported surface.

.. toctree::
   :maxdepth: 2

   program
   registers
   operations
   simulators/index
   pulse-emulator
   atom-emulators
   noise
   estimator
   job
   result
   exceptions
   experimental
