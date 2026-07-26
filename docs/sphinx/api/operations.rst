Operations (``op``)
=======================

Add all normal operations through :py:meth:`~fatqat.Program.add`. The
exported operation values and classes documented below are the supported gate
surface. Internal implementation classes whose names end in ``Gate`` are not
the construction surface for application code.

Examples use ``import fatqat.operations as op``.

The entries below retain their generated constructors and public members
for exact interface details.

.. autoclass:: fatqat.operations.Operation
   :members:
   :show-inheritance:

Fixed single-qubit gates
------------------------

:py:obj:`~fatqat.operations.I`, :py:obj:`~fatqat.operations.H`,
:py:obj:`~fatqat.operations.S`, :py:obj:`~fatqat.operations.Sdg`,
:py:obj:`~fatqat.operations.SX`,
:py:obj:`~fatqat.operations.T`, :py:obj:`~fatqat.operations.Tdg`,
:py:obj:`~fatqat.operations.X`, :py:obj:`~fatqat.operations.Y`, and
:py:obj:`~fatqat.operations.Z` are ready-to-use values. For example:
``program.add(op.H, 0)``.

.. autodata:: fatqat.operations.I
.. autodata:: fatqat.operations.H
.. autodata:: fatqat.operations.S
.. autodata:: fatqat.operations.Sdg
.. autodata:: fatqat.operations.SX
.. autodata:: fatqat.operations.T
.. autodata:: fatqat.operations.Tdg
.. autodata:: fatqat.operations.X
.. autodata:: fatqat.operations.Y
.. autodata:: fatqat.operations.Z

Atom loading
------------

.. autoclass:: fatqat.operations.LoadAtom
   :members:
   :show-inheritance:

Parametric gates
-----------------

:py:obj:`~fatqat.operations.RX` (``theta``),
:py:obj:`~fatqat.operations.RY` (``theta``),
:py:obj:`~fatqat.operations.RZ` (``theta``),
:py:obj:`~fatqat.operations.Phase` (``theta``), and
:py:obj:`~fatqat.operations.CPhase` (``theta``) take angles in radians.
:py:obj:`~fatqat.operations.CPhase` uses ``(control, target)`` target
order.

.. autoclass:: fatqat.operations.RX
   :members:
   :show-inheritance:

.. autoclass:: fatqat.operations.RY
   :members:
   :show-inheritance:

.. autoclass:: fatqat.operations.RZ
   :members:
   :show-inheritance:

.. autoclass:: fatqat.operations.Phase
   :members:
   :show-inheritance:

.. autoclass:: fatqat.operations.CPhase
   :members:
   :show-inheritance:

Fixed multi-qubit gates
------------------------

:py:obj:`~fatqat.operations.CX`, :py:obj:`~fatqat.operations.CZ`,
:py:obj:`~fatqat.operations.Swap`, :py:obj:`~fatqat.operations.CY`,
:py:obj:`~fatqat.operations.CS`, :py:obj:`~fatqat.operations.iSwap`,
:py:obj:`~fatqat.operations.CCX`, and :py:obj:`~fatqat.operations.CSwap`
are ready-to-use values. For controlled operations, controls come before
targets: ``program.add(op.CX, (control, target))``.

.. autodata:: fatqat.operations.CX
.. autodata:: fatqat.operations.CZ
.. autodata:: fatqat.operations.Swap
.. autodata:: fatqat.operations.CY
.. autodata:: fatqat.operations.CS
.. autodata:: fatqat.operations.iSwap
.. autodata:: fatqat.operations.CCX
.. autodata:: fatqat.operations.CSwap

Reset
-----

:py:data:`~fatqat.operations.Reset` prepares one or more targets in ``|0⟩``:
``program.add(op.Reset, 0)``. See
:doc:`../guide/measurement-and-conditions` for reset and conditions.

.. autodata:: fatqat.operations.Reset

Qudit gates
-----------

:py:obj:`~fatqat.operations.Shift` (``power``),
:py:obj:`~fatqat.operations.Clock` (``power``),
:py:obj:`~fatqat.operations.Sum`,
:py:obj:`~fatqat.operations.SwapLevels` (``j, k``),
:py:obj:`~fatqat.operations.Fourier`,
:py:obj:`~fatqat.operations.Fourierdg`,
:py:obj:`~fatqat.operations.SubspaceRX` (``theta, subspace``),
:py:obj:`~fatqat.operations.SubspaceRY` (``theta, subspace``),
:py:obj:`~fatqat.operations.SubspaceRZ` (``theta, subspace``), and
:py:obj:`~fatqat.operations.CClock` (``power``) works with
higher-dimensional registers. Read
:doc:`../guide/advanced` for the qutrit workflow.
The :doc:`../guide/gates` guide explains singleton versus parametric gate
syntax, target order, and grid selections.

.. autoclass:: fatqat.operations.Shift
   :members:
   :show-inheritance:

.. autoclass:: fatqat.operations.Clock
   :members:
   :show-inheritance:

.. autodata:: fatqat.operations.Sum

.. autoclass:: fatqat.operations.SwapLevels
   :members:
   :show-inheritance:

.. autodata:: fatqat.operations.Fourier
.. autodata:: fatqat.operations.Fourierdg

.. autoclass:: fatqat.operations.SubspaceRX
   :members:
   :show-inheritance:

.. autoclass:: fatqat.operations.SubspaceRY
   :members:
   :show-inheritance:

.. autoclass:: fatqat.operations.SubspaceRZ
   :members:
   :show-inheritance:

.. autoclass:: fatqat.operations.CClock
   :members:
   :show-inheritance:
