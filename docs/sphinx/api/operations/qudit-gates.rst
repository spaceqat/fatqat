Qudit gates
===========

.. currentmodule:: fatqat.operations

The default matrix implementation derives these gates from the target register
dimensions and works for every local dimension ``d >= 2`` unless a row below
says otherwise. Device-specific and pulse backends may support a smaller set or
none. Power arguments are integers and are reduced modulo the relevant target
dimension at matrix lowering; the operation value retains the original
integer. The matrix simulator performs dimension-dependent realization when it
prepares the program.

The supported ``power`` inputs are integers. ``Shift``, ``Clock``, and
``CClock`` retain the supplied value without eager type validation, so a
non-integer or ``Parameter`` can survive construction; those forms are outside
the supported contract and may fail during matrix lowering or produce an
undocumented fractional phase. Parameter binding does not override an
operation field's declared type contract.

.. list-table:: Qudit gates
   :header-rows: 1
   :widths: 22 25 53

   * - Value or constructor
     - Targets and constraints
     - Basis action
   * - :py:class:`Shift` ``(power)``
     - One scalar; any dimension
     - ``|k> -> |(k + power) mod d>``. ``Shift(1)`` is X for ``d=2``.
   * - :py:class:`Clock` ``(power)``
     - One scalar; any dimension
     - ``|k> -> omega**(k*power)|k>``, ``omega=exp(2*pi*i/d)``.
       ``Clock(1)`` is Z for ``d=2``.
   * - :py:data:`Sum`
     - ``(control, target)`` with equal dimensions in the default matrix map
     - ``|i,j> -> |i,(i+j) mod d>``. It is CX for ``d=2``.
   * - :py:class:`SwapLevels` ``(j, k)``
     - One scalar; ``0 <= j,k < d`` and ``j != k``
     - Exchanges ``|j>`` and ``|k>`` and fixes every other level.
   * - :py:data:`Fourier`
     - One scalar; any dimension
     - ``|j> -> sum(exp(2*pi*i*j*k/d)|k>) / sqrt(d)``. It is H for
       ``d=2``.
   * - :py:data:`InverseFourier`
     - One scalar; any dimension
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

The supported level inputs are integers. ``SwapLevels`` and the subspace
rotations reject equal or negative values at construction, and ``Program.add``
then rejects values outside the resolved scalar target dimension. They do not
eagerly verify the index type. The subspace constructors also retain rather
than copy their two-item input; pass the documented tuple, because a mutable
container remains mutable and can make the frozen operation unhashable.
Non-integral values are unsupported and can fail during matrix lowering.

``Sum`` is different: equal dimensions are a constraint of the default
implementation rather than the frontend, so mismatched dimensions raise
:py:exc:`~fatqat.errors.MatrixImplementationError` during matrix lowering.

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

.. autoclass:: fatqat.operations.SwapLevels
   :members:
   :exclude-members: validate_targets
   :no-inherited-members:
   :show-inheritance:

.. autodata:: fatqat.operations.Fourier
.. autodata:: fatqat.operations.InverseFourier

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
