Operations (``fqc.ops``)
========================

.. autoclass:: fatqcat.operations.Operation
   :members:
   :show-inheritance:

Single-qubit fixed gates
------------------------

Fixed gates take no parameters and are exported as ready-to-use singleton
values (e.g. ``fqc.ops.H``, not ``fqc.ops.H()``).

.. autodata:: fatqcat.operations.I
.. autodata:: fatqcat.operations.H
.. autodata:: fatqcat.operations.S
.. autodata:: fatqcat.operations.Sdg
.. autodata:: fatqcat.operations.T
.. autodata:: fatqcat.operations.Tdg
.. autodata:: fatqcat.operations.X
.. autodata:: fatqcat.operations.Y
.. autodata:: fatqcat.operations.Z

Parametric gates
-----------------

Parametric gates are exported as classes and must be instantiated with their
parameter, e.g. ``fqc.ops.RX(0.2)``.

.. autoclass:: fatqcat.operations.RX
   :members:
   :show-inheritance:
.. autoclass:: fatqcat.operations.RY
   :members:
   :show-inheritance:
.. autoclass:: fatqcat.operations.RZ
   :members:
   :show-inheritance:
.. autoclass:: fatqcat.operations.Phase
   :members:
   :show-inheritance:
.. autoclass:: fatqcat.operations.CPhase
   :members:
   :show-inheritance:

Multi-qubit fixed gates
------------------------

.. autodata:: fatqcat.operations.CX
.. autodata:: fatqcat.operations.CZ
.. autodata:: fatqcat.operations.Swap
.. autodata:: fatqcat.operations.CY
.. autodata:: fatqcat.operations.CS
.. autodata:: fatqcat.operations.iSwap
.. autodata:: fatqcat.operations.CCX
.. autodata:: fatqcat.operations.CSwap

Dimension-generic (qudit) gates
---------------------------------

``Shift``, ``Clock``, ``SwapLevels``, ``SubspaceRX``, ``SubspaceRY``,
``SubspaceRZ``, and ``CClock`` are exported as classes and must be
instantiated with their parameters; ``Sum``, ``Fourier``, and ``Fourierdg``
take no parameters and are exported as singletons.

.. autoclass:: fatqcat.operations.Shift
   :members:
   :show-inheritance:
.. autoclass:: fatqcat.operations.Clock
   :members:
   :show-inheritance:
.. autodata:: fatqcat.operations.Sum
.. autoclass:: fatqcat.operations.SwapLevels
   :members:
   :show-inheritance:
.. autodata:: fatqcat.operations.Fourier
.. autodata:: fatqcat.operations.Fourierdg
.. autoclass:: fatqcat.operations.SubspaceRX
   :members:
   :show-inheritance:
.. autoclass:: fatqcat.operations.SubspaceRY
   :members:
   :show-inheritance:
.. autoclass:: fatqcat.operations.SubspaceRZ
   :members:
   :show-inheritance:
.. autoclass:: fatqcat.operations.CClock
   :members:
   :show-inheritance:

Reset
------

.. autodata:: fatqcat.operations.Reset
