Gate realization
================

.. currentmodule:: fatqat.emulator

:py:class:`PulseImplementationMap` maps ordinary gates to pulse
definitions. A direct :doc:`pulse-operation` already contains its controls and
does not use this map.

Rules
-----

A rule is called as ``rule(operation, *, device_operands=...)`` and must return
a :py:class:`PulseDefinition`. Register a general rule to handle every ordered
device-operand tuple, or register separate rules for specific tuples. A
tuple-specific entry may also be a fixed definition or a callable that only
accepts ``operation``. The tuple contains ordered physical labels such as
``("q0", "q1")``, not program register references.

Definitions
-----------

A :py:class:`PulseDefinition` contains a duration, a tuple of controls, and
optional ``PhaseShift`` or ``PhaseSwap`` actions. Conditions and noise remain
on the operation in the program.

``PhaseShift`` changes one model frame after the pulse. ``PhaseSwap`` exchanges
two frames. Direct pulse operations do not have post-actions.

The emulator calls and validates a selected rule when the gate is used. Raise
:py:exc:`~fatqat.errors.BackendValidationError` to report unsupported operands
or parameters. Other exceptions and return values that are not
``PulseDefinition`` are reported as ``PulseImplementationError``.

The :doc:`transmon <../pulse-emulator>` and
:doc:`neutral-atom <../atom-emulators>` pages show the built-in maps and full
workflows.

Reference
---------

.. autoclass:: fatqat.emulator.PulseImplementationMap
   :members:

.. autoclass:: fatqat.emulator.PulseDefinition
   :members:

.. autoclass:: fatqat.emulator.PhaseShift
   :members:

.. autoclass:: fatqat.emulator.PhaseSwap
   :members:
