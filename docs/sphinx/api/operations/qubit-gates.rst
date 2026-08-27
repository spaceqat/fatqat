Qubit gates
===========

.. currentmodule:: fatqat.operations

The default matrix implementation defines the gates on this page for
dimension-2 targets. The frontend deliberately allows one to be added to a
higher-dimensional target, but the default matrix backend then raises
:py:exc:`~fatqat.errors.BackendValidationError` during preparation because the
matrix shape does not match the target dimension. Hardware-profile and pulse
backends may support only a subset; consult the selected backend page.

Fixed gates
-----------

Fixed gates are immutable singleton values and must not be called. The
single-qubit matrices use ``|0>, |1>`` basis order.

.. list-table:: Fixed single-qubit gates
   :header-rows: 1
   :widths: 14 31 55

   * - Value
     - Matrix or basis action
     - Meaning
   * - :py:data:`I`
     - ``[[1, 0], [0, 1]]``
     - Identity.
   * - :py:data:`H`
     - ``[[1, 1], [1, -1]] / sqrt(2)``
     - Hadamard superposition transform.
   * - :py:data:`X`
     - ``[[0, 1], [1, 0]]``
     - Exchanges ``|0>`` and ``|1>``.
   * - :py:data:`Y`
     - ``[[0, -i], [i, 0]]``
     - Pauli-Y bit-and-phase flip.
   * - :py:data:`Z`
     - ``diag(1, -1)``
     - Negates the ``|1>`` amplitude.
   * - :py:data:`S`
     - ``diag(1, i)``
     - Square root of Z.
   * - :py:data:`Sdg`
     - ``diag(1, -i)``
     - Inverse of S.
   * - :py:data:`SX`
     - ``[[1+i, 1-i], [1-i, 1+i]] / 2``
     - Principal square root of X.
   * - :py:data:`T`
     - ``diag(1, exp(i*pi/4))``
     - Applies a ``pi/4`` phase to ``|1>``.
   * - :py:data:`Tdg`
     - ``diag(1, exp(-i*pi/4))``
     - Inverse of T.

.. autodata:: fatqat.operations.I
.. autodata:: fatqat.operations.H
.. autodata:: fatqat.operations.X
.. autodata:: fatqat.operations.Y
.. autodata:: fatqat.operations.Z
.. autodata:: fatqat.operations.S
.. autodata:: fatqat.operations.Sdg
.. autodata:: fatqat.operations.SX
.. autodata:: fatqat.operations.T
.. autodata:: fatqat.operations.Tdg

For the multi-qubit values below, targets are ordered exactly as shown.

.. list-table:: Fixed multi-qubit gates
   :header-rows: 1
   :widths: 14 36 50

   * - Value
     - Target order
     - Basis action
   * - :py:data:`CX`
     - ``(control, target)``
     - Applies X to the target when the control is ``|1>``.
   * - :py:data:`CY`
     - ``(control, target)``
     - Applies Y to the target when the control is ``|1>``.
   * - :py:data:`CZ`
     - ``(control, target)``
     - Negates ``|11>``.
   * - :py:data:`CS`
     - ``(control, target)``
     - Applies S to the target when the control is ``|1>``;
       ``diag(1, 1, 1, i)``.
   * - :py:data:`Swap`
     - ``(target0, target1)``
     - Exchanges the two target states.
   * - :py:data:`iSwap`
     - ``(target0, target1)``
     - Maps ``|01>`` to ``i|10>`` and ``|10>`` to ``i|01>``.
   * - :py:data:`CCX`
     - ``(control0, control1, target)``
     - Toffoli: applies X when both controls are ``|1>``.
   * - :py:data:`CSwap`
     - ``(control, target0, target1)``
     - Fredkin: exchanges the two targets when the control is ``|1>``.

.. autodata:: fatqat.operations.CX
.. autodata:: fatqat.operations.CY
.. autodata:: fatqat.operations.CZ
.. autodata:: fatqat.operations.CS
.. autodata:: fatqat.operations.Swap
.. autodata:: fatqat.operations.iSwap
.. autodata:: fatqat.operations.CCX
.. autodata:: fatqat.operations.CSwap

Parameterized gates
-------------------

All angles are in radians. These constructors store their arguments unchanged;
they do not normalize angles. Every angle field accepts a
:py:class:`~fatqat.Parameter` for later binding. Every backend and exporter
rejects an unbound program with
:py:exc:`~fatqat.errors.BackendValidationError` before numeric realization.

Let ``c = cos(theta/2)`` and ``s = sin(theta/2)``. The following definitions
use ``|0>, |1>`` basis order.

.. list-table:: Parameterized qubit gates
   :header-rows: 1
   :widths: 22 28 50

   * - Constructor
     - Targets
     - Matrix or equivalence
   * - :py:class:`RX` ``(theta)``
     - One scalar or one view
     - ``[[c, -i*s], [-i*s, c]]``.
   * - :py:class:`RY` ``(theta)``
     - One scalar or one view
     - ``[[c, -s], [s, c]]``.
   * - :py:class:`RZ` ``(theta)``
     - One scalar or one view
     - ``diag(exp(-i*theta/2), exp(i*theta/2))``.
   * - :py:class:`Phase` ``(theta)``
     - One scalar
     - ``diag(1, exp(i*theta))``; differs from RZ only by global phase.
   * - :py:class:`U` ``(theta, phi, lam)``
     - One scalar
     - ``[[c, -exp(i*lam)*s], [exp(i*phi)*s,
       exp(i*(phi+lam))*c]]``.
   * - :py:class:`U1` ``(lam)``
     - One scalar
     - Equivalent to ``Phase(lam)``.
   * - :py:class:`U2` ``(phi, lam)``
     - One scalar
     - Equivalent to ``U(pi/2, phi, lam)``.
   * - :py:class:`U3` ``(theta, phi, lam)``
     - One scalar
     - Numerically identical to ``U(theta, phi, lam)``; legacy Qiskit name.
   * - :py:class:`CPhase` ``(theta)``
     - ``(control, target)`` scalars
     - ``diag(1, 1, 1, exp(i*theta))``.

Each class below shows its own constructor fields. Common operation properties
and validation hooks are documented on the :doc:`Operations overview
<../operations>`.

.. autoclass:: fatqat.operations.RX
   :members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: fatqat.operations.RY
   :members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: fatqat.operations.RZ
   :members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: fatqat.operations.Phase
   :members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: fatqat.operations.U
   :members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: fatqat.operations.U1
   :members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: fatqat.operations.U2
   :members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: fatqat.operations.U3
   :members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: fatqat.operations.CPhase
   :members:
   :no-inherited-members:
   :show-inheritance:
