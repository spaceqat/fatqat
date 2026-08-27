Direct pulse control
====================

.. currentmodule:: fatqat.operations

:py:class:`PulseOperation` groups concurrent physical controls for one
duration. It does not take a separate ``targets`` argument; construct the
operation and add it with ``program.add(operation)``. Each
:py:attr:`~fatqat.emulator.PulseControl.channel` identifies the physical
resource or resources it drives, and the selected emulator resolves those
channel addresses while preparing the program.

Ordinary operations receive logical :py:class:`~fatqat.RegisterRef` operands,
which pass through :py:class:`~fatqat.ResourceLayout` when the backend binds
the program. Direct channel addresses instead bind against the emulator's
physical model and are not remapped by ``ResourceLayout``. A direct control
may therefore address a modeled subsystem that no ordinary operation uses.
For the transmon model, a drive or detuning channel resolves one subsystem;
an exchange channel resolves two subsystems and their declared coupling.

Construction requires a positive finite duration, at least one control, unique
channels, and controls that end within the block. The chosen pulse emulator
then checks that channel addresses name compatible model resources, values obey
the channel's real/complex and amplitude limits, the duration is supported,
and concurrent controls can share physical resources. These late checks raise
:py:exc:`~fatqat.errors.BackendValidationError`. An operation can be reused
with another compatible model and arrangement. The matrix
:py:class:`~fatqat.simulator.Simulator` rejects direct pulse operations with
:py:exc:`~fatqat.errors.UnsupportedOperationError`. Operation-scoped noise
cannot be attached to them; :py:meth:`fatqat.NoiseModel.add` raises
:py:exc:`ValueError` for that selector.

``Program.add(condition=...)`` may guard a direct pulse operation. When the
condition is false, the controls are disabled but the full duration still
elapses: model drift and background Lindblad sources continue over that
interval. As with other conditions, support depends on the backend and
execution method.

See :doc:`../pulse-emulator` for the owning
:py:class:`~fatqat.emulator.PulseControl` and
:py:class:`~fatqat.emulator.SampledWaveform` references, model control
factories, interpolation rules, and complete workflows.

.. autoclass:: fatqat.operations.PulseOperation
   :members:
   :no-inherited-members:
   :show-inheritance:
