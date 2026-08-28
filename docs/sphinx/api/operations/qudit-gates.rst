Qudit gates
===========

.. currentmodule:: fatqat.operations

The gates on this page are defined for finite-dimensional subsystems (qudits)
with local dimension ``d >= 2``. :py:meth:`~fatqat.Program.add` records the
operation but does not determine whether a selected compiler or backend
supports it for a particular device. Operation-level dimension constraints are
listed below; consult that compiler or backend's capability documentation for
supported operations and device constraints. ``power`` must be an integer;
negative and oversized powers are valid and are equivalent modulo the relevant
target dimension.

.. list-table:: Qudit gates
   :header-rows: 1
   :widths: 22 25 53

   * - Value or constructor
     - Targets and constraints
     - Basis action
   * - :py:class:`Shift` ``(power)``
     - One scalar; any ``d >= 2``
     - ``|k> -> |(k + power) mod d>``. ``Shift(1)`` is X for ``d=2``.
   * - :py:class:`Clock` ``(power)``
     - One scalar; any ``d >= 2``
     - ``|k> -> omega**(k*power)|k>``, ``omega=exp(2*pi*i/d)``.
       ``Clock(1)`` is Z for ``d=2``.
   * - :py:data:`Sum`
     - ``(control, target)`` with equal dimensions
     - ``|i,j> -> |i,(i+j) mod d>``. It is CX for ``d=2``.
   * - :py:class:`SwapLevels` ``(j, k)``
     - One scalar; ``0 <= j,k < d`` and ``j != k``
     - Exchanges ``|j>`` and ``|k>`` and fixes every other level.
   * - :py:data:`Fourier`
     - One scalar; any ``d >= 2``
     - ``|j> -> sum(exp(2*pi*i*j*k/d)|k>) / sqrt(d)``. It is H for
       ``d=2``.
   * - :py:data:`InverseFourier`
     - One scalar; any ``d >= 2``
     - Conjugate transpose of ``Fourier``; uses the negative exponent.
   * - :py:class:`SubspaceRX` ``(theta, (j, k))``
     - One scalar; two distinct in-range levels
     - With ``c=cos(theta/2)``, ``s=sin(theta/2)``: ``|j> -> c|j>-i*s|k>``
       and ``|k> -> -i*s|j>+c|k>``.
   * - :py:class:`SubspaceRY` ``(theta, (j, k))``
     - One scalar; two distinct in-range levels
     - ``|j> -> c|j>+s|k>`` and ``|k> -> -s|j>+c|k>``. Reversing
       ``(j, k)`` reverses the rotation.
   * - :py:class:`SubspaceRZ` ``(theta, (j, k))``
     - One scalar; two distinct in-range levels
     - ``|j>`` gains ``exp(-i*theta/2)`` and ``|k>`` gains
       ``exp(i*theta/2)``. Reversing the pair reverses the rotation.
   * - :py:class:`CClock` ``(power)``
     - ``(control, target)``; dimensions may differ
     - ``|i,j>`` gains ``omega**(i*j*power)`` using the target's
       ``omega=exp(2*pi*i/d_target)``. It is CZ for two qubits and power 1.

``SwapLevels`` and the subspace rotations require integer level indices. They
reject equal or negative indices at construction, and
:py:meth:`~fatqat.Program.add` rejects indices outside the resolved scalar
target dimension.

``Sum`` is defined for a control and target with equal local dimensions.
:py:meth:`~fatqat.Program.add` records mismatched targets; the selected
compiler or backend must reject them during program preparation.

Matrix definitions
------------------

The matrices below act on column vectors. For one-qudit gates, rows and
columns use the computational-basis order
:math:`\lvert 0\rangle,\lvert 1\rangle,\ldots,\lvert d-1\rangle`.

Shift and Clock
~~~~~~~~~~~~~~~

For ``Shift(power=p)`` and ``Clock(power=p)``, let
:math:`\omega_d=\exp(2\pi i/d)`. Their general operators are

.. math::

   X_d^p
   = \sum_{k=0}^{d-1}
     \left\lvert (k+p)\bmod d \right\rangle\!\left\langle k\right\rvert,
   \qquad
   Z_d^p
   = \sum_{k=0}^{d-1}
     \omega_d^{pk}\left\lvert k\right\rangle\!\left\langle k\right\rvert.

For a qutrit, ``Shift(1)`` and ``Clock(1)`` use the ordered basis
:math:`(\lvert0\rangle,\lvert1\rangle,\lvert2\rangle)` and have matrices

.. math::

   X_3 =
   \begin{bmatrix}
   0 & 0 & 1 \\
   1 & 0 & 0 \\
   0 & 1 & 0
   \end{bmatrix},
   \qquad
   Z_3 =
   \begin{bmatrix}
   1 & 0 & 0 \\
   0 & \omega_3 & 0 \\
   0 & 0 & \omega_3^2
   \end{bmatrix}.

Sum
~~~

``Sum`` takes operands as ``(control, target)``. For equal dimension ``d``,
the control is the local most-significant factor:

.. math::

   \operatorname{SUM}_d
   = \sum_{i,j=0}^{d-1}
     \left\lvert i,(i+j)\bmod d\right\rangle
     \!\left\langle i,j\right\rvert.

For :math:`d=3`, rows and columns use
:math:`(\lvert00\rangle,\lvert01\rangle,\lvert02\rangle,
\lvert10\rangle,\lvert11\rangle,\lvert12\rangle,
\lvert20\rangle,\lvert21\rangle,\lvert22\rangle)`, where the first digit is
the control and the second is the target:

.. math::

   \operatorname{SUM}_3 =
   \begin{bmatrix}
   I_3&0&0 \\
   0&X_3&0 \\
   0&0&X_3^2
   \end{bmatrix}.

SwapLevels
~~~~~~~~~~

For distinct levels :math:`j` and :math:`k`, the general operator is

.. math::

   S_{j,k}
   = I - \lvert j\rangle\!\langle j\rvert
       - \lvert k\rangle\!\langle k\rvert
       + \lvert j\rangle\!\langle k\rvert
       + \lvert k\rangle\!\langle j\rvert.

For a qutrit, ``SwapLevels(0, 2)`` uses the ordered basis
:math:`(\lvert0\rangle,\lvert1\rangle,\lvert2\rangle)`:

.. math::

   S_{0,2} =
   \begin{bmatrix}
   0&0&1 \\
   0&1&0 \\
   1&0&0
   \end{bmatrix}.

Fourier transforms
~~~~~~~~~~~~~~~~~~

With :math:`\omega_d=\exp(2\pi i/d)`, ``Fourier`` and ``InverseFourier``
are

.. math::

   F_d
   = \frac{1}{\sqrt d}\sum_{j,k=0}^{d-1}
     \omega_d^{jk}\lvert k\rangle\!\langle j\rvert,
   \qquad
   F_d^{-1}=F_d^\dagger
   = \frac{1}{\sqrt d}\sum_{j,k=0}^{d-1}
     \omega_d^{-jk}\lvert k\rangle\!\langle j\rvert.

For :math:`d=3`, both matrices use the ordered basis
:math:`(\lvert0\rangle,\lvert1\rangle,\lvert2\rangle)`:

.. math::

   F_3 = \frac{1}{\sqrt3}
   \begin{bmatrix}
   1&1&1 \\
   1&\omega_3&\omega_3^2 \\
   1&\omega_3^2&\omega_3
   \end{bmatrix},
   \qquad
   F_3^{-1} = \frac{1}{\sqrt3}
   \begin{bmatrix}
   1&1&1 \\
   1&\omega_3^2&\omega_3 \\
   1&\omega_3&\omega_3^2
   \end{bmatrix}.

Subspace rotations
~~~~~~~~~~~~~~~~~~

Let :math:`c=\cos(\theta/2)` and :math:`s=\sin(\theta/2)`. For the ordered
level pair ``subspace=(j, k)``, the general operators are

.. math::

   \begin{aligned}
   R_X^{(j,k)}(\theta)
   &= I + (c-1)(\lvert j\rangle\!\langle j\rvert
                   +\lvert k\rangle\!\langle k\rvert)
      -is(\lvert j\rangle\!\langle k\rvert
                   +\lvert k\rangle\!\langle j\rvert), \\
   R_Y^{(j,k)}(\theta)
   &= I + (c-1)(\lvert j\rangle\!\langle j\rvert
                   +\lvert k\rangle\!\langle k\rvert)
      -s\lvert j\rangle\!\langle k\rvert
      +s\lvert k\rangle\!\langle j\rvert, \\
   R_Z^{(j,k)}(\theta)
   &= I + (e^{-i\theta/2}-1)\lvert j\rangle\!\langle j\rvert
      +(e^{i\theta/2}-1)\lvert k\rangle\!\langle k\rvert.
   \end{aligned}

For a qutrit with ``subspace=(0, 2)``, rows and columns use
:math:`(\lvert0\rangle,\lvert1\rangle,\lvert2\rangle)` and the selected
levels retain the stated ``(j, k)`` order:

.. math::

   \begin{aligned}
   R_X^{(0,2)}(\theta)
   &= \begin{bmatrix}
      c&0&-is \\
      0&1&0 \\
      -is&0&c
      \end{bmatrix}, \\[1ex]
   R_Y^{(0,2)}(\theta)
   &= \begin{bmatrix}
      c&0&-s \\
      0&1&0 \\
      s&0&c
      \end{bmatrix}, \\[1ex]
   R_Z^{(0,2)}(\theta)
   &= \begin{bmatrix}
      e^{-i\theta/2}&0&0 \\
      0&1&0 \\
      0&0&e^{i\theta/2}
      \end{bmatrix}.
   \end{aligned}

CClock
~~~~~~

``CClock(power=p)`` takes operands as ``(control, target)``. If their
dimensions are :math:`d_c` and :math:`d_t`, the control is the local
most-significant factor, :math:`\omega_t=\exp(2\pi i/d_t)`, and

.. math::

   \operatorname{CClock}_{d_c,d_t}^{(p)}
   = \sum_{i=0}^{d_c-1}\sum_{j=0}^{d_t-1}
     \omega_t^{ijp}\lvert i,j\rangle\!\langle i,j\rvert.

For two qutrits and ``CClock(1)``, rows and columns use
:math:`(\lvert00\rangle,\lvert01\rangle,\lvert02\rangle,
\lvert10\rangle,\lvert11\rangle,\lvert12\rangle,
\lvert20\rangle,\lvert21\rangle,\lvert22\rangle)`, with the control digit
first:

.. math::

   \operatorname{CClock}_{3,3}^{(1)} =
   \begin{bmatrix}
   I_3&0&0 \\
   0&Z_3&0 \\
   0&0&Z_3^2
   \end{bmatrix}.

Each class below shows its own constructor fields and target checks. Common
operation properties are documented on the :doc:`Operations overview
<../operations>`.

.. autoclass:: fatqat.operations.Shift
   :members:
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: fatqat.operations.Clock
   :members:
   :no-inherited-members:
   :show-inheritance:

.. autodata:: fatqat.operations.Sum
   :no-value:

.. autoclass:: fatqat.operations.SwapLevels
   :members:
   :exclude-members: validate_targets
   :no-inherited-members:
   :show-inheritance:

.. autodata:: fatqat.operations.Fourier
   :no-value:
.. autodata:: fatqat.operations.InverseFourier
   :no-value:

.. autoclass:: fatqat.operations.SubspaceRX
   :members:
   :exclude-members: validate_targets
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: fatqat.operations.SubspaceRY
   :members:
   :exclude-members: validate_targets
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: fatqat.operations.SubspaceRZ
   :members:
   :exclude-members: validate_targets
   :no-inherited-members:
   :show-inheritance:

.. autoclass:: fatqat.operations.CClock
   :members:
   :no-inherited-members:
   :show-inheritance:
