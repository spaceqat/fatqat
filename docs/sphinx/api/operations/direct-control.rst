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
Lowering obtains scheduling resource claims and execution engine indices from
these channel bindings.

The operation constructor checks common structure and containment. The chosen
pulse emulator later checks whether channel addresses are compatible with its
model family and name valid resources, whether values obey that channel's
real/complex and amplitude limits, whether the duration is supported, and
whether concurrent controls can share physical resources. Addresses and
controls are structural immutable values rather than handles owned by one
model instance, so an operation can be reused with another compatible model
and arrangement. The matrix :py:class:`~fatqat.simulator.Simulator` rejects
direct pulse operations, and operation-scoped noise cannot be attached to
them.

``Program.add(condition=...)`` may guard a direct pulse operation. Pulse
lowering preserves the resolved condition. When it is false, the block's
controls and condition-scoped generators are disabled, but the scheduled
duration still elapses: model drift and background Lindblad sources continue
over that interval. As with other conditions, the frontend does not promise
support from every backend or execution method.

See :doc:`../pulse-emulator` for the owning
:py:class:`~fatqat.emulator.PulseControl` and
:py:class:`~fatqat.emulator.SampledWaveform` references, model control
factories, interpolation rules, and complete workflows.

.. autoclass:: fatqat.operations.PulseOperation
   :members:
   :no-inherited-members:
   :show-inheritance:
