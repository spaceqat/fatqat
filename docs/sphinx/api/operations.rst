Operations (``qs.ops``)
========================

.. autoclass:: qnsim.operations.Operation
   :members:
   :show-inheritance:

Single-qubit fixed gates
------------------------

Fixed gates take no parameters and are exported as ready-to-use singleton
values (e.g. ``qs.ops.H``, not ``qs.ops.H()``).

.. autodata:: qnsim.operations.I
.. autodata:: qnsim.operations.H
.. autodata:: qnsim.operations.S
.. autodata:: qnsim.operations.Sdg
.. autodata:: qnsim.operations.T
.. autodata:: qnsim.operations.Tdg
.. autodata:: qnsim.operations.X
.. autodata:: qnsim.operations.Y
.. autodata:: qnsim.operations.Z

Parametric gates
-----------------

Parametric gates are exported as classes and must be instantiated with their
parameter, e.g. ``qs.ops.RX(0.2)``.

.. autoclass:: qnsim.operations.RX
   :members:
   :show-inheritance:
.. autoclass:: qnsim.operations.RY
   :members:
   :show-inheritance:
.. autoclass:: qnsim.operations.RZ
   :members:
   :show-inheritance:
.. autoclass:: qnsim.operations.Phase
   :members:
   :show-inheritance:
.. autoclass:: qnsim.operations.CPhase
   :members:
   :show-inheritance:

Multi-qubit fixed gates
------------------------

.. autodata:: qnsim.operations.CX
.. autodata:: qnsim.operations.CZ
.. autodata:: qnsim.operations.Swap
.. autodata:: qnsim.operations.CY
.. autodata:: qnsim.operations.CS
.. autodata:: qnsim.operations.iSwap
.. autodata:: qnsim.operations.CCX
.. autodata:: qnsim.operations.CSwap

Dimension-generic (qudit) gates
---------------------------------

``Shift``, ``Clock``, ``SwapLevels``, ``SubspaceRX``, ``SubspaceRY``,
``SubspaceRZ``, and ``CClock`` are exported as classes and must be
instantiated with their parameters; ``Sum``, ``Fourier``, and ``Fourierdg``
take no parameters and are exported as singletons.

.. autoclass:: qnsim.operations.Shift
   :members:
   :show-inheritance:
.. autoclass:: qnsim.operations.Clock
   :members:
   :show-inheritance:
.. autodata:: qnsim.operations.Sum
.. autoclass:: qnsim.operations.SwapLevels
   :members:
   :show-inheritance:
.. autodata:: qnsim.operations.Fourier
.. autodata:: qnsim.operations.Fourierdg
.. autoclass:: qnsim.operations.SubspaceRX
   :members:
   :show-inheritance:
.. autoclass:: qnsim.operations.SubspaceRY
   :members:
   :show-inheritance:
.. autoclass:: qnsim.operations.SubspaceRZ
   :members:
   :show-inheritance:
.. autoclass:: qnsim.operations.CClock
   :members:
   :show-inheritance:

Reset
------

.. autodata:: qnsim.operations.Reset
