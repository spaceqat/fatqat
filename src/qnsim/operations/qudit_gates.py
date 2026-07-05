"""Dimension-generic gates (generalized Pauli group + controlled add)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

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
    _num_subsystems: ClassVar[int] = 1


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
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class SumGate(Operation):
    """Generalized controlled add: ``|i, j> -> |i, (i + j) mod d>``.

    ``targets = (control, target)``; operand 0 is the control. The default
    implementation requires equal target dimensions.

    .. math::

        \\mathrm{Sum} : |i, j\\rangle \\mapsto |i, (i + j) \\bmod d\\rangle

    At its smallest dimension, :math:`d = 2`, this reduces to exactly
    :class:`CXGate`:

    .. math::

        \\mathrm{Sum}\\big|_{d=2} = \\begin{pmatrix}
        1&0&0&0\\\\ 0&1&0&0\\\\ 0&0&0&1\\\\ 0&0&1&0
        \\end{pmatrix}

    The class itself is not part of the ``qs.ops`` public surface (not in
    ``__all__``) but stays attribute-accessible for ``isinstance`` checks;
    ``Sum`` (the singleton) is the one users add to a program.
    """

    name: ClassVar[str] = "Sum"
    _num_subsystems: ClassVar[int] = 2


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

    Attributes:
        j: First level index (distinct from k, non-negative).
        k: Second level index (distinct from j, non-negative).
    """

    j: int
    k: int
    name: ClassVar[str] = "SwapLevels"
    _num_subsystems: ClassVar[int] = 1

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

    Internal only: unlike `SumGate` (attribute-accessible via `qs.ops.SumGate`
    though excluded from `__all__`), this class is not imported into
    `operations/__init__.py` at all, so it is not reachable as `qs.ops.
    FourierGate`. `Fourier` (the singleton) is the only public surface.
    """

    name: ClassVar[str] = "Fourier"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class FourierdgGate(Operation):
    """Inverse of FourierGate (conjugate transpose). Coincides with Fourier
    at dim=2 (H is self-adjoint) but differs for d > 2.

    Internal only: not imported into `operations/__init__.py`, so it is not
    reachable as `qs.ops.FourierdgGate`. `Fourierdg` (the singleton) is the
    only public surface.
    """

    name: ClassVar[str] = "Fourierdg"
    _num_subsystems: ClassVar[int] = 1


# Parameterless, so exported only as singleton values (see the "Public
# fixed-gate instances" convention already used for Sum). Unlike SumGate,
# the classes themselves are not imported into operations/__init__.py.
Fourier = FourierGate()
Fourierdg = FourierdgGate()


@dataclass(frozen=True)
class SubspaceRX(Operation):
    """Rotation around X embedded in a 2-level subspace of a d-level qudit,
    identity on the complementary levels. subspace[0] plays the |0> role,
    subspace[1] the |1> role. Reduces to RX(theta) at dim=2, subspace=(0,1).

    Attributes:
        theta: Rotation angle in radians.
        subspace: Pair of distinct, non-negative level indices (j, k).
    """

    theta: float
    subspace: tuple[int, int]
    name: ClassVar[str] = "SubspaceRX"
    _num_subsystems: ClassVar[int] = 1

    def __post_init__(self) -> None:
        j, k = self.subspace
        if j == k:
            raise ValueError(f"SubspaceRX subspace requires distinct levels, got ({j}, {k})")
        if j < 0 or k < 0:
            raise ValueError(f"SubspaceRX subspace levels must be non-negative, got ({j}, {k})")

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
    identity on the complementary levels. subspace[0] plays the |0> role,
    subspace[1] the |1> role. Reduces to RY(theta) at dim=2, subspace=(0,1).

    Attributes:
        theta: Rotation angle in radians.
        subspace: Pair of distinct, non-negative level indices (j, k).
    """

    theta: float
    subspace: tuple[int, int]
    name: ClassVar[str] = "SubspaceRY"
    _num_subsystems: ClassVar[int] = 1

    def __post_init__(self) -> None:
        j, k = self.subspace
        if j == k:
            raise ValueError(f"SubspaceRY subspace requires distinct levels, got ({j}, {k})")
        if j < 0 or k < 0:
            raise ValueError(f"SubspaceRY subspace levels must be non-negative, got ({j}, {k})")

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
    identity on the complementary levels. subspace[0] plays the |0> role,
    subspace[1] the |1> role. Reduces to RZ(theta) at dim=2, subspace=(0,1).

    Attributes:
        theta: Rotation angle in radians.
        subspace: Pair of distinct, non-negative level indices (j, k).
    """

    theta: float
    subspace: tuple[int, int]
    name: ClassVar[str] = "SubspaceRZ"
    _num_subsystems: ClassVar[int] = 1

    def __post_init__(self) -> None:
        j, k = self.subspace
        if j == k:
            raise ValueError(f"SubspaceRZ subspace requires distinct levels, got ({j}, {k})")
        if j < 0 or k < 0:
            raise ValueError(f"SubspaceRZ subspace levels must be non-negative, got ({j}, {k})")

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
    when the control is |i>. targets = (control, target); operand 0 is the
    control. Unlike Sum, does not require equal control/target dimensions.
    power is reduced modulo the target dimension at lowering (a cyclic
    count, like Clock's power). Reduces to CZ at dim=2, power=1.

    Attributes:
        power: Phase power (reduced modulo the target dimension at lowering).
    """

    power: int
    name: ClassVar[str] = "CClock"
    _num_subsystems: ClassVar[int] = 2
