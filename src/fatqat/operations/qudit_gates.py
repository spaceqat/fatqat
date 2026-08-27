"""Dimension-generic gates (generalized Pauli group + controlled add).

Examples:
    ``Shift`` on a qutrit (``dim=3``) cyclically shifts the basis level:

    >>> import fatqat as fq
    >>> import fatqat.operations as ops
    >>> qutrit = fq.QuantumRegister(1, dim=3)
    >>> program = fq.Program([qutrit])
    >>> program.add(ops.Shift(1), 0)
    >>> result = fq.simulator.Simulator("SV").run(
    ...     program,
    ...     shots=1,
    ...     result_config={"counts": False, "final_state": True},
    ... ).result()
    >>> result.get_statevector()
    array([0.+0.j, 1.+0.j, 0.+0.j])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..parameters import Parameter
from .base import Operation


@dataclass(frozen=True)
class Shift(Operation):
    """Generalized-Pauli cyclic shift: ``|k> -> |(k + power) mod d>``.

    Dimension-free: applies to a subsystem of any dimension; its matrix is
    built from the target dimension at backend lowering. Reduces to X at
    ``dim=2, power=1``.

    .. math::

        \\mathrm{Shift}(d, p) : |k\\rangle \\mapsto |(k + p) \\bmod d\\rangle

    For example, at :math:`d = 3, p = 1`:

    .. math::

        \\mathrm{Shift}(3, 1) = \\begin{pmatrix}
        0&0&1\\\\ 1&0&0\\\\ 0&1&0
        \\end{pmatrix}

    Attributes:
        power: Shift amount (reduced modulo the target dimension at lowering).
    """

    power: int
    name: ClassVar[str] = "Shift"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class Clock(Operation):
    """Generalized-Pauli phase: ``|k> -> omega^(k*power) |k>``, omega=e^{2πi/d}.

    Reduces to Z at ``dim=2, power=1``.

    .. math::

        \\mathrm{Clock}(d, p) : |k\\rangle \\mapsto \\omega^{kp} |k\\rangle,
        \\quad \\omega = e^{2\\pi i/d}

    For example, at :math:`d = 3, p = 1`:

    .. math::

        \\mathrm{Clock}(3, 1) = \\begin{pmatrix}
        1&0&0\\\\ 0&\\omega&0\\\\ 0&0&\\omega^2
        \\end{pmatrix}, \\quad \\omega = e^{2\\pi i/3}

    Attributes:
        power: Phase power (reduced modulo the target dimension at lowering).
    """

    power: int
    name: ClassVar[str] = "Clock"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class SumGate(Operation):
    """Generalized controlled add: ``|i, j> -> |i, (i + j) mod d>``.

    ``targets = (control, target)``; operand 0 is the control. The default
    implementation requires equal target dimensions.

    .. math::

        \\mathrm{Sum} : |i, j\\rangle \\mapsto |i, (i + j) \\bmod d\\rangle

    For example, on two qutrits (:math:`d = 3`), the sum wraps modulo 3:

    .. math::

        \\mathrm{Sum} |2, 2\\rangle = |2, (2 + 2) \\bmod 3\\rangle = |2, 1\\rangle

    In matrix form, ``targets = (control, target)`` puts the control on the
    local most-significant digit, so the :math:`d = 3` matrix is block
    diagonal, one :math:`3 \\times 3` block per control value:

    .. math::

        \\mathrm{Sum}\\big|_{d=3} = \\begin{pmatrix}
        I_3 & 0 & 0 \\\\ 0 & P_1 & 0 \\\\ 0 & 0 & P_2
        \\end{pmatrix}

    where :math:`P_k` is the permutation matrix that cyclically shifts the
    target level by :math:`k \\bmod 3`:

    .. math::

        P_1 = \\begin{pmatrix} 0&0&1\\\\ 1&0&0\\\\ 0&1&0 \\end{pmatrix}
        \\qquad
        P_2 = \\begin{pmatrix} 0&1&0\\\\ 0&0&1\\\\ 1&0&0 \\end{pmatrix}

    At its smallest dimension, :math:`d = 2`, this reduces to exactly
    :class:`CXGate`.

    The class itself is not part of the ``fatqat.operations`` public surface (not in
    ``__all__``) but stays attribute-accessible for ``isinstance`` checks;
    ``Sum`` (the singleton) is the one users add to a program.
    """

    name: ClassVar[str] = "Sum"
    num_subsystems: ClassVar[int] = 2


# `Sum` takes no parameters, so - like the fixed gates - it is exported only
# as a singleton value.
Sum = SumGate()


@dataclass(frozen=True)
class SwapLevels(Operation):
    """Level-transposition gate: swaps basis levels ``j`` and ``k``, identity
    on every other level. Dimension-free: its matrix is built from the target
    dimension at backend lowering. Reduces to X at ``dim=2, (j,k)=(0,1)``.
    Hermitian and self-inverse (no ``dg`` variant).

    Known in the qutrit literature as X01/X02/X12 (the Muthukrishnan-Stroud
    gates) at dim=3.

    .. math::

        \\mathrm{SwapLevels}(j, k) : |j\\rangle \\leftrightarrow |k\\rangle,
        \\quad |m\\rangle \\mapsto |m\\rangle \\ (m \\neq j, k)

    For example, at :math:`d = 3, (j, k) = (0, 1)` (the X01 gate):

    .. math::

        \\mathrm{SwapLevels}(0, 1) = \\begin{pmatrix}
        0&1&0\\\\ 1&0&0\\\\ 0&0&1
        \\end{pmatrix}

    Attributes:
        j: First level index (distinct from k, non-negative).
        k: Second level index (distinct from j, non-negative).
    """

    j: int
    k: int
    name: ClassVar[str] = "SwapLevels"
    num_subsystems: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if self.j == self.k:
            raise ValueError(f"SwapLevels requires j != k, got j=k={self.j}")
        if self.j < 0 or self.k < 0:
            raise ValueError(
                f"SwapLevels level indices must be non-negative, got ({self.j}, {self.k})"
            )

    def validate_targets(self, targets) -> None:
        dim = targets[0].register.dim
        if self.j >= dim or self.k >= dim:
            raise ValueError(
                f"SwapLevels({self.j}, {self.k}) invalid for target dimension "
                f"{dim}: level indices must satisfy 0 <= j, k < dim"
            )


@dataclass(frozen=True)
class FourierGate(Operation):
    """Single-qudit discrete Fourier transform (Chrestenson gate): the H
    analogue for any dimension. Dimension-free; matrix built from the target
    dimension at backend lowering. Reduces to H at dim=2.

    Known in the qutrit literature as THadamard at dim=3.

    .. math::

        \\mathrm{Fourier} : |j\\rangle \\mapsto \\frac{1}{\\sqrt{d}}
        \\sum_{k=0}^{d-1} \\omega^{jk} |k\\rangle, \\quad \\omega = e^{2\\pi i/d}

    For example, at :math:`d = 3`:

    .. math::

        \\mathrm{Fourier}\\big|_{d=3} = \\frac{1}{\\sqrt{3}}\\begin{pmatrix}
        1&1&1\\\\ 1&\\omega&\\omega^2\\\\ 1&\\omega^2&\\omega
        \\end{pmatrix}, \\quad \\omega = e^{2\\pi i/3}

    Internal only: unlike `SumGate` (attribute-accessible via `ops.SumGate`
    though excluded from `__all__`), this class is not imported into
    `operations/__init__.py` at all, so it is not reachable as `ops.
    FourierGate`. `Fourier` (the singleton) is the only public surface.
    """

    name: ClassVar[str] = "Fourier"
    num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class FourierdgGate(Operation):
    """Inverse of FourierGate (conjugate transpose). Coincides with Fourier
    at dim=2 (H is self-adjoint) but differs for d > 2.

    .. math::

        \\mathrm{Fourier}^\\dagger : |j\\rangle \\mapsto \\frac{1}{\\sqrt{d}}
        \\sum_{k=0}^{d-1} \\omega^{-jk} |k\\rangle, \\quad \\omega = e^{2\\pi i/d}

    For example, at :math:`d = 3`:

    .. math::

        \\mathrm{Fourier}^\\dagger\\big|_{d=3} = \\frac{1}{\\sqrt{3}}\\begin{pmatrix}
        1&1&1\\\\ 1&\\omega^2&\\omega\\\\ 1&\\omega&\\omega^2
        \\end{pmatrix}, \\quad \\omega = e^{2\\pi i/3}

    The public singleton is :data:`~fatqat.operations.InverseFourier`.
    """

    name: ClassVar[str] = "InverseFourier"
    num_subsystems: ClassVar[int] = 1


# Parameterless, so exported only as singleton values (see the "Public
# fixed-gate instances" convention already used for Sum). Unlike SumGate,
# the classes themselves are not imported into operations/__init__.py.
Fourier = FourierGate()
InverseFourier = FourierdgGate()


@dataclass(frozen=True)
class SubspaceRX(Operation):
    """Rotation around X embedded in a 2-level subspace of a d-level qudit,
    identity on the complementary levels. subspace[0] plays the ``|0>`` role,
    subspace[1] the ``|1>`` role. Reduces to RX(theta) at dim=2, subspace=(0,1).

    .. math::

        \\mathrm{SubspaceRX}(\\theta, (j, k)) : \\begin{cases}
        |j\\rangle \\mapsto \\cos\\frac{\\theta}{2}|j\\rangle
            - i\\sin\\frac{\\theta}{2}|k\\rangle \\\\
        |k\\rangle \\mapsto -i\\sin\\frac{\\theta}{2}|j\\rangle
            + \\cos\\frac{\\theta}{2}|k\\rangle \\\\
        |m\\rangle \\mapsto |m\\rangle & (m \\neq j, k)
        \\end{cases}

    For example, at :math:`d = 3, (j, k) = (0, 1)`:

    .. math::

        \\mathrm{SubspaceRX}(\\theta, (0, 1))\\big|_{d=3} = \\begin{pmatrix}
        \\cos\\frac{\\theta}{2} & -i\\sin\\frac{\\theta}{2} & 0 \\\\
        -i\\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2} & 0 \\\\
        0 & 0 & 1
        \\end{pmatrix}

    Attributes:
        theta: Rotation angle in radians.
        subspace: Pair of distinct, non-negative level indices (j, k).
    """

    theta: float | Parameter
    subspace: tuple[int, int]
    name: ClassVar[str] = "SubspaceRX"
    num_subsystems: ClassVar[int] = 1

    def __post_init__(self) -> None:
        j, k = self.subspace
        if j == k:
            raise ValueError(
                f"SubspaceRX subspace requires distinct levels, got ({j}, {k})"
            )
        if j < 0 or k < 0:
            raise ValueError(
                f"SubspaceRX subspace levels must be non-negative, got ({j}, {k})"
            )

    def validate_targets(self, targets) -> None:
        dim = targets[0].register.dim
        j, k = self.subspace
        if j >= dim or k >= dim:
            raise ValueError(
                f"SubspaceRX subspace {self.subspace} invalid for target "
                f"dimension {dim}: indices must satisfy 0 <= j, k < dim"
            )


@dataclass(frozen=True)
class SubspaceRY(Operation):
    """Rotation around Y embedded in a 2-level subspace of a d-level qudit,
    identity on the complementary levels. subspace[0] plays the ``|0>`` role,
    subspace[1] the ``|1>`` role. Reduces to RY(theta) at dim=2, subspace=(0,1).

    .. math::

        \\mathrm{SubspaceRY}(\\theta, (j, k)) : \\begin{cases}
        |j\\rangle \\mapsto \\cos\\frac{\\theta}{2}|j\\rangle
            + \\sin\\frac{\\theta}{2}|k\\rangle \\\\
        |k\\rangle \\mapsto -\\sin\\frac{\\theta}{2}|j\\rangle
            + \\cos\\frac{\\theta}{2}|k\\rangle \\\\
        |m\\rangle \\mapsto |m\\rangle & (m \\neq j, k)
        \\end{cases}

    For example, at :math:`d = 3, (j, k) = (0, 1)`:

    .. math::

        \\mathrm{SubspaceRY}(\\theta, (0, 1))\\big|_{d=3} = \\begin{pmatrix}
        \\cos\\frac{\\theta}{2} & -\\sin\\frac{\\theta}{2} & 0 \\\\
        \\sin\\frac{\\theta}{2} & \\cos\\frac{\\theta}{2} & 0 \\\\
        0 & 0 & 1
        \\end{pmatrix}

    Attributes:
        theta: Rotation angle in radians.
        subspace: Pair of distinct, non-negative level indices (j, k).
    """

    theta: float | Parameter
    subspace: tuple[int, int]
    name: ClassVar[str] = "SubspaceRY"
    num_subsystems: ClassVar[int] = 1

    def __post_init__(self) -> None:
        j, k = self.subspace
        if j == k:
            raise ValueError(
                f"SubspaceRY subspace requires distinct levels, got ({j}, {k})"
            )
        if j < 0 or k < 0:
            raise ValueError(
                f"SubspaceRY subspace levels must be non-negative, got ({j}, {k})"
            )

    def validate_targets(self, targets) -> None:
        dim = targets[0].register.dim
        j, k = self.subspace
        if j >= dim or k >= dim:
            raise ValueError(
                f"SubspaceRY subspace {self.subspace} invalid for target "
                f"dimension {dim}: indices must satisfy 0 <= j, k < dim"
            )


@dataclass(frozen=True)
class SubspaceRZ(Operation):
    """Rotation around Z embedded in a 2-level subspace of a d-level qudit,
    identity on the complementary levels. subspace[0] plays the ``|0>`` role,
    subspace[1] the ``|1>`` role. Reduces to RZ(theta) at dim=2, subspace=(0,1).

    .. math::

        \\mathrm{SubspaceRZ}(\\theta, (j, k)) : \\begin{cases}
        |j\\rangle \\mapsto e^{-i\\theta/2}|j\\rangle \\\\
        |k\\rangle \\mapsto e^{i\\theta/2}|k\\rangle \\\\
        |m\\rangle \\mapsto |m\\rangle & (m \\neq j, k)
        \\end{cases}

    For example, at :math:`d = 3, (j, k) = (0, 1)`:

    .. math::

        \\mathrm{SubspaceRZ}(\\theta, (0, 1))\\big|_{d=3} = \\begin{pmatrix}
        e^{-i\\theta/2} & 0 & 0 \\\\
        0 & e^{i\\theta/2} & 0 \\\\
        0 & 0 & 1
        \\end{pmatrix}

    Attributes:
        theta: Rotation angle in radians.
        subspace: Pair of distinct, non-negative level indices (j, k).
    """

    theta: float | Parameter
    subspace: tuple[int, int]
    name: ClassVar[str] = "SubspaceRZ"
    num_subsystems: ClassVar[int] = 1

    def __post_init__(self) -> None:
        j, k = self.subspace
        if j == k:
            raise ValueError(
                f"SubspaceRZ subspace requires distinct levels, got ({j}, {k})"
            )
        if j < 0 or k < 0:
            raise ValueError(
                f"SubspaceRZ subspace levels must be non-negative, got ({j}, {k})"
            )

    def validate_targets(self, targets) -> None:
        dim = targets[0].register.dim
        j, k = self.subspace
        if j >= dim or k >= dim:
            raise ValueError(
                f"SubspaceRZ subspace {self.subspace} invalid for target "
                f"dimension {dim}: indices must satisfy 0 <= j, k < dim"
            )


@dataclass(frozen=True)
class CClock(Operation):
    """Generalized controlled-phase: applies Clock(i*power) to the target
    when the control is ``|i>``. targets = (control, target); operand 0 is the
    control. Unlike Sum, does not require equal control/target dimensions.
    power is reduced modulo the target dimension at lowering (a cyclic
    count, like Clock's power). Reduces to CZ at dim=2, power=1.

    .. math::

        C\\text{-}\\mathrm{Clock}(p) : |i, j\\rangle \\mapsto
        \\omega^{ipj} |i, j\\rangle, \\quad \\omega = e^{2\\pi i/d_t}

    where :math:`d_t` is the target dimension. At its smallest dimension,
    :math:`d_c = d_t = 2, p = 1`, this reduces to exactly :class:`CZGate`:

    .. math::

        C\\text{-}\\mathrm{Clock}(1)\\big|_{d=2} = \\begin{pmatrix}
        1&0&0&0\\\\ 0&1&0&0\\\\ 0&0&1&0\\\\ 0&0&0&-1
        \\end{pmatrix}

    Attributes:
        power: Phase power (reduced modulo the target dimension at lowering).
    """

    power: int
    name: ClassVar[str] = "CClock"
    num_subsystems: ClassVar[int] = 2
