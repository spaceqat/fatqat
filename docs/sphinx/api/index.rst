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
     - No
     - Yes, global
     - None

``PulseImplementationMap`` is the public map value type for both gate-capable
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
   :maxdepth: 1

   program
   registers
   operations
   simulator
   pulse-emulator
   atom-emulators
   noise
   estimator
   result
   exceptions
   experimental
