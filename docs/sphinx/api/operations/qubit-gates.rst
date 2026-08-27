Qubit gates
===========

.. currentmodule:: fatqat.operations

The default matrix backend supports the gates on this page on dimension-2
targets. ``Program.add`` does not enforce that backend-specific constraint; the
matrix backend raises :py:exc:`~fatqat.errors.BackendValidationError` during
program preparation for a higher-dimensional target. Hardware-profile and
pulse backends may support only a subset; consult the selected backend page.

Fixed gates
-----------

Fixed gates are immutable singleton values and must not be called. The
single-qubit matrices use ``|0>, |1>`` basis order.

.. list-table:: Fixed single-qubit gates
   :header-rows: 1
   :widths: 18 82

   * - Value
     - Basis action
   * - :py:data:`I`
     - Leaves :math:`|0\rangle` and :math:`|1\rangle` unchanged.
   * - :py:data:`H`
     - Maps :math:`|0\rangle` to :math:`(|0\rangle+|1\rangle)/\sqrt{2}`
       and :math:`|1\rangle` to
       :math:`(|0\rangle-|1\rangle)/\sqrt{2}`.
   * - :py:data:`X`
     - Exchanges :math:`|0\rangle` and :math:`|1\rangle`.
   * - :py:data:`Y`
     - Maps :math:`|0\rangle` to :math:`i|1\rangle` and
       :math:`|1\rangle` to :math:`-i|0\rangle`.
   * - :py:data:`Z`
     - Maps :math:`|1\rangle` to :math:`-|1\rangle`.
   * - :py:data:`S`
     - Maps :math:`|1\rangle` to :math:`i|1\rangle`.
   * - :py:data:`Sdg`
     - Maps :math:`|1\rangle` to :math:`-i|1\rangle`.
   * - :py:data:`SX`
     - Two applications have the same action as X.
   * - :py:data:`T`
     - Maps :math:`|1\rangle` to :math:`e^{i\pi/4}|1\rangle`.
   * - :py:data:`Tdg`
     - Maps :math:`|1\rangle` to :math:`e^{-i\pi/4}|1\rangle`.

.. autodata:: fatqat.operations.I
   :no-value:
.. autodata:: fatqat.operations.H
   :no-value:
.. autodata:: fatqat.operations.X
   :no-value:
.. autodata:: fatqat.operations.Y
   :no-value:
.. autodata:: fatqat.operations.Z
   :no-value:
.. autodata:: fatqat.operations.S
   :no-value:
.. autodata:: fatqat.operations.Sdg
   :no-value:
.. autodata:: fatqat.operations.SX
   :no-value:
.. autodata:: fatqat.operations.T
   :no-value:
.. autodata:: fatqat.operations.Tdg
   :no-value:

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
     - Applies S to the target when the control is ``|1>``.
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

Matrix definitions
~~~~~~~~~~~~~~~~~~

These matrices act on column state vectors. For the single-qubit gates, rows
and columns use basis order
:math:`(|0\rangle,|1\rangle)`:

.. math::

   \begin{aligned}
   I &= \begin{pmatrix}1&0\\0&1\end{pmatrix},
   & H &= \frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\\[0.5em]
   X &= \begin{pmatrix}0&1\\1&0\end{pmatrix},
   & Y &= \begin{pmatrix}0&-i\\i&0\end{pmatrix},\\[0.5em]
   Z &= \begin{pmatrix}1&0\\0&-1\end{pmatrix},
   & S &= \begin{pmatrix}1&0\\0&i\end{pmatrix},\\[0.5em]
   \mathrm{Sdg} &= \begin{pmatrix}1&0\\0&-i\end{pmatrix},
   & \mathrm{SX} &= \frac{1}{2}\begin{pmatrix}1+i&1-i\\1-i&1+i\end{pmatrix},\\[0.5em]
   T &= \begin{pmatrix}1&0\\0&e^{i\pi/4}\end{pmatrix},
   & \mathrm{Tdg} &= \begin{pmatrix}1&0\\0&e^{-i\pi/4}\end{pmatrix}.
   \end{aligned}

For each two-qubit matrix, the targets are :math:`(q_0,q_1)` and rows and
columns use basis order
:math:`(|00\rangle,|01\rangle,|10\rangle,|11\rangle)`. The first operand
:math:`q_0` is the local most-significant bit. It is the control for ``CX``,
``CY``, ``CZ``, and ``CS``; :math:`q_1` is the target.

.. math::

   CX = \begin{pmatrix}
   1&0&0&0\\
   0&1&0&0\\
   0&0&0&1\\
   0&0&1&0
   \end{pmatrix}

.. math::

   CY = \begin{pmatrix}
   1&0&0&0\\
   0&1&0&0\\
   0&0&0&-i\\
   0&0&i&0
   \end{pmatrix}

.. math::

   CZ = \begin{pmatrix}
   1&0&0&0\\
   0&1&0&0\\
   0&0&1&0\\
   0&0&0&-1
   \end{pmatrix}

.. math::

   CS = \begin{pmatrix}
   1&0&0&0\\
   0&1&0&0\\
   0&0&1&0\\
   0&0&0&i
   \end{pmatrix}

For ``Swap`` and ``iSwap``, the same basis order applies with operand order
``(target0, target1)``.

.. math::

   \mathrm{Swap} = \begin{pmatrix}
   1&0&0&0\\
   0&0&1&0\\
   0&1&0&0\\
   0&0&0&1
   \end{pmatrix}

.. math::

   i\mathrm{Swap} = \begin{pmatrix}
   1&0&0&0\\
   0&0&i&0\\
   0&i&0&0\\
   0&0&0&1
   \end{pmatrix}

For ``CCX``, operand order is ``(control0, control1, target)``. Rows and
columns use basis order :math:`(|000\rangle,|001\rangle,|010\rangle,
|011\rangle,|100\rangle,|101\rangle,|110\rangle,|111\rangle)`, with the
first operand as the most-significant bit:

.. math::

   CCX = \begin{pmatrix}
   1&0&0&0&0&0&0&0\\
   0&1&0&0&0&0&0&0\\
   0&0&1&0&0&0&0&0\\
   0&0&0&1&0&0&0&0\\
   0&0&0&0&1&0&0&0\\
   0&0&0&0&0&1&0&0\\
   0&0&0&0&0&0&0&1\\
   0&0&0&0&0&0&1&0
   \end{pmatrix}

For ``CSwap``, operand order is ``(control, target0, target1)``. Rows and
columns use the same three-bit basis order, again with the first operand as
the most-significant bit:

.. math::

   \mathrm{CSwap} = \begin{pmatrix}
   1&0&0&0&0&0&0&0\\
   0&1&0&0&0&0&0&0\\
   0&0&1&0&0&0&0&0\\
   0&0&0&1&0&0&0&0\\
   0&0&0&0&1&0&0&0\\
   0&0&0&0&0&0&1&0\\
   0&0&0&0&0&1&0&0\\
   0&0&0&0&0&0&0&1
   \end{pmatrix}

.. autodata:: fatqat.operations.CX
   :no-value:
.. autodata:: fatqat.operations.CY
   :no-value:
.. autodata:: fatqat.operations.CZ
   :no-value:
.. autodata:: fatqat.operations.CS
   :no-value:
.. autodata:: fatqat.operations.Swap
   :no-value:
.. autodata:: fatqat.operations.iSwap
   :no-value:
.. autodata:: fatqat.operations.CCX
   :no-value:
.. autodata:: fatqat.operations.CSwap
   :no-value:

Parameterized gates
-------------------

All angles are in radians and are not normalized. Every angle field accepts a
:py:class:`~fatqat.Parameter` for later binding through
:py:meth:`fatqat.Program.assign_parameters`.

Let ``c = cos(theta/2)`` and ``s = sin(theta/2)``. The following definitions
use ``|0>, |1>`` basis order.

.. list-table:: Parameterized qubit gates
   :header-rows: 1
   :widths: 22 28 50

   * - Constructor
     - Targets
     - Definition
   * - :py:class:`RX` ``(theta)``
     - One scalar or one view
     - Rotation about the X axis by ``theta``.
   * - :py:class:`RY` ``(theta)``
     - One scalar or one view
     - Rotation about the Y axis by ``theta``.
   * - :py:class:`RZ` ``(theta)``
     - One scalar or one view
     - Rotation about the Z axis by ``theta``.
   * - :py:class:`Phase` ``(theta)``
     - One scalar
     - Differs from RZ only by global phase.
   * - :py:class:`U` ``(theta, phi, lam)``
     - One scalar
     - General Qiskit-compatible single-qubit gate.
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
     - Multiplies :math:`|11\rangle` by :math:`e^{i\theta}`.

Matrix definitions
~~~~~~~~~~~~~~~~~~

These matrices act on column state vectors. The single-qubit matrices below
use row and column basis order
:math:`(|0\rangle,|1\rangle)`. Let
:math:`c=\cos(\theta/2)` and :math:`s=\sin(\theta/2)`:

.. math::

   RX(\theta) = \begin{pmatrix}
   c&-is\\
   -is&c
   \end{pmatrix},
   \qquad
   RY(\theta) = \begin{pmatrix}
   c&-s\\
   s&c
   \end{pmatrix}

.. math::

   RZ(\theta) = \begin{pmatrix}
   e^{-i\theta/2}&0\\
   0&e^{i\theta/2}
   \end{pmatrix},
   \qquad
   \mathrm{Phase}(\theta) = \begin{pmatrix}
   1&0\\
   0&e^{i\theta}
   \end{pmatrix}

For ``U``, ``U1``, ``U2``, and ``U3``, operands still use the single-qubit
basis above and the parameter order is the constructor order shown in the
table:

.. math::

   U(\theta,\phi,\lambda) = \begin{pmatrix}
   \cos(\theta/2)&-e^{i\lambda}\sin(\theta/2)\\
   e^{i\phi}\sin(\theta/2)&e^{i(\phi+\lambda)}\cos(\theta/2)
   \end{pmatrix}

.. math::

   U1(\lambda) = \begin{pmatrix}
   1&0\\
   0&e^{i\lambda}
   \end{pmatrix},
   \qquad
   U2(\phi,\lambda) = \frac{1}{\sqrt{2}}\begin{pmatrix}
   1&-e^{i\lambda}\\
   e^{i\phi}&e^{i(\phi+\lambda)}
   \end{pmatrix}

.. math::

   U3(\theta,\phi,\lambda) = \begin{pmatrix}
   \cos(\theta/2)&-e^{i\lambda}\sin(\theta/2)\\
   e^{i\phi}\sin(\theta/2)&e^{i(\phi+\lambda)}\cos(\theta/2)
   \end{pmatrix}

For ``CPhase``, operand order is ``(control, target)``. Rows and columns use
basis order :math:`(|00\rangle,|01\rangle,|10\rangle,|11\rangle)`, with the
control as the local most-significant bit:

.. math::

   \mathrm{CPhase}(\theta) = \begin{pmatrix}
   1&0&0&0\\
   0&1&0&0\\
   0&0&1&0\\
   0&0&0&e^{i\theta}
   \end{pmatrix}

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
