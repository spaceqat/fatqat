PulseOperation
==============

.. currentmodule:: fatqat.operations

:py:class:`PulseOperation` adds an explicit pulse block to a program. Import
it from ``fatqat.operations``, normally as ``ops.PulseOperation``.

An ordinary gate is added by passing logical targets to ``Program.add``.
For a ``PulseOperation``, do not pass targets: every
:py:class:`~fatqat.emulator.PulseControl` already names the physical channel
to drive. Add it with ``program.add(operation)``. A
:py:class:`~fatqat.ResourceLayout` does not remap its channels.

Conditions and noise
--------------------

``TransmonEmulator`` allows ``program.add(operation, condition=...)``. If the
condition is false, the controls are skipped but the block still takes its full
duration. Model drift and background Lindblad noise continue during that time.
``Atom2LevelEmulator`` does not support conditions.

Operation-scoped noise cannot be attached to a direct pulse block, so
``noise.add(..., operation=ops.PulseOperation)`` raises ``ValueError``.
Background noise selected by target or device label still applies.

Support
-------

The three pulse emulators listed in :doc:`index` support
``PulseOperation``. :doc:`Matrix simulators and their device profiles
<../simulators/index>` do not; neither do circuit drawing or OpenQASM export.

Reference
---------

.. autoclass:: fatqat.operations.PulseOperation
   :no-members:
   :no-inherited-members:
   :show-inheritance:
