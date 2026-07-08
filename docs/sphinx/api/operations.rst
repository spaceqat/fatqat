Operations (``fq.ops``)
========================

.. autoclass:: fatqat.operations.Operation
   :members:
   :show-inheritance:

Single-qubit fixed gates
------------------------

Fixed gates take no parameters and are exported as ready-to-use singleton
values (e.g. ``fq.ops.H``, not ``fq.ops.H()``).

.. autodata:: fatqat.operations.I
.. autodata:: fatqat.operations.H
.. autodata:: fatqat.operations.S
.. autodata:: fatqat.operations.Sdg
.. autodata:: fatqat.operations.T
.. autodata:: fatqat.operations.Tdg
.. autodata:: fatqat.operations.X
.. autodata:: fatqat.operations.Y
.. autodata:: fatqat.operations.Z

Parametric gates
-----------------

Parametric gates are exported as classes and must be instantiated with their
parameter, e.g. ``fq.ops.RX(0.2)``.

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

Multi-qubit fixed gates
------------------------

.. autodata:: fatqat.operations.CX
.. autodata:: fatqat.operations.CZ
.. autodata:: fatqat.operations.Swap
.. autodata:: fatqat.operations.CY
.. autodata:: fatqat.operations.CS
.. autodata:: fatqat.operations.iSwap
.. autodata:: fatqat.operations.CCX
.. autodata:: fatqat.operations.CSwap

Dimension-generic (qudit) gates
---------------------------------

``Shift``, ``Clock``, ``SwapLevels``, ``SubspaceRX``, ``SubspaceRY``,
``SubspaceRZ``, and ``CClock`` are exported as classes and must be
instantiated with their parameters; ``Sum``, ``Fourier``, and ``Fourierdg``
take no parameters and are exported as singletons.

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

Reset
------

.. autodata:: fatqat.operations.Reset
