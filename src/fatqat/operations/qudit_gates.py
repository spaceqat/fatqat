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
from .parametric_gates import _validate_angles
from ..registers import RegisterRef
from .base import Operation


@dataclass(frozen=True)
class Shift(Operation):
    """Cyclically shift one subsystem's computational-basis level.

    On a target of dimension ``d``, ``|k>`` maps to
    ``|(k + power) mod d>``. Negative and oversized integer powers are valid
    and equivalent modulo ``d``. ``Shift(1)`` on a qubit is ``X``.

    Args:
        power: Integer cyclic shift amount.
    """

    power: int
    name: ClassVar[str] = "Shift"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class Clock(Operation):
    """Apply a dimension-dependent phase to each basis level.

    On a target of dimension ``d``, ``|k>`` gains phase
    ``omega**(k*power)``, where ``omega = exp(2*pi*i/d)``. Negative and
    oversized integer powers are valid and equivalent modulo ``d``.
    ``Clock(1)`` on a qubit is ``Z``.

    Args:
        power: Integer phase power.
    """

    power: int
    name: ClassVar[str] = "Clock"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class SumGate(Operation):
    """Add a control level into a target level modulo their shared dimension.

    Targets are ``(control, target)``. For equal dimension ``d``, ``|i, j>``
    maps to ``|i, (i + j) mod d>``; for example, two qutrits map ``|2, 2>``
    to ``|2, 1>``. On qubits this is ``CX``.

    The operation is defined for equal-dimension targets. `fatqat.Program.add`
    records a mismatch, and the backend rejects it when the program runs.
    """

    name: ClassVar[str] = "Sum"
    num_subsystems: ClassVar[int] = 2
    _accepts_views: ClassVar[bool] = True


Sum = SumGate()


@dataclass(frozen=True)
class SwapLevels(Operation):
    """Exchange two basis levels and leave every other level unchanged.

    The gate is Hermitian and self-inverse. On a qubit,
    ``SwapLevels(0, 1)`` is ``X``.

    Args:
        j: First non-negative level index.
        k: Distinct second non-negative level index.

    Raises:
        ValueError: At construction if the indices are equal or negative, or
            from `fatqat.Program.add` if either index is outside the target's
            ``0 <= index < dim`` range.
    """

    j: int
    k: int
    name: ClassVar[str] = "SwapLevels"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.j == self.k:
            raise ValueError(f"SwapLevels requires j != k, got j=k={self.j}")
        if self.j < 0 or self.k < 0:
            raise ValueError(
                f"SwapLevels level indices must be non-negative, got ({self.j}, {self.k})"
            )

    def validate_targets(self, targets: tuple[RegisterRef, ...]) -> None:
        """Validate both selected levels against the resolved target.

        Args:
            targets: One resolved scalar `fatqat.RegisterRef`.

        Raises:
            ValueError: If ``j`` or ``k`` is not less than the target's local
                dimension.
        """
        dim = targets[0].register.dim
        if self.j >= dim or self.k >= dim:
            raise ValueError(
                f"SwapLevels({self.j}, {self.k}) invalid for target dimension "
                f"{dim}: level indices must satisfy 0 <= j, k < dim"
            )


@dataclass(frozen=True)
class FourierGate(Operation):
    """Apply the positive-exponent discrete Fourier transform to one qudit.

    For dimension ``d``, ``|j>`` maps to
    ``sum(exp(2*pi*i*j*k/d) * |k>) / sqrt(d)``. It is ``H`` for ``d=2``.
    """

    name: ClassVar[str] = "Fourier"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


@dataclass(frozen=True)
class FourierdgGate(Operation):
    """Apply the inverse discrete Fourier transform to one qudit.

    This is ``Fourier``'s conjugate transpose and uses the negative exponent.
    It is ``H`` for ``d=2`` but differs from ``Fourier`` at higher dimensions.
    """

    name: ClassVar[str] = "InverseFourier"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True


Fourier = FourierGate()
InverseFourier = FourierdgGate()


@dataclass(frozen=True)
class SubspaceRX(Operation):
    """Apply `RX` inside two selected levels and leave other levels unchanged.

    In the ordered pair ``(j, k)``, ``j`` has RX's ``|0>`` role and ``k`` has
    its ``|1>`` role. ``SubspaceRX(theta, (0, 1))`` on a qubit is ``RX`` with
    the same ``theta``.

    Args:
        theta: Numeric angle in radians, or a `fatqat.Parameter` to bind before
            execution.
        subspace: Tuple of exactly two distinct, non-negative integer level
            indices in ``(|0>, |1>)`` role order.

    Raises:
        ValueError: At construction if ``subspace`` does not contain exactly
            two values, its indices are equal, or an index is negative; or from
            `fatqat.Program.add` if an index is outside the target dimension.
    """

    theta: float | Parameter
    subspace: tuple[int, int]
    name: ClassVar[str] = "SubspaceRX"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True

    def __post_init__(self) -> None:
        _validate_angles(self, theta=self.theta)
        j, k = self.subspace
        if j == k:
            raise ValueError(
                f"SubspaceRX subspace requires distinct levels, got ({j}, {k})"
            )
        if j < 0 or k < 0:
            raise ValueError(
                f"SubspaceRX subspace levels must be non-negative, got ({j}, {k})"
            )

    def validate_targets(self, targets: tuple[RegisterRef, ...]) -> None:
        """Validate the selected subspace against the resolved target.

        Args:
            targets: One resolved scalar `fatqat.RegisterRef`.

        Raises:
            ValueError: If either subspace index is not less than the target's
                local dimension.
        """
        dim = targets[0].register.dim
        j, k = self.subspace
        if j >= dim or k >= dim:
            raise ValueError(
                f"SubspaceRX subspace {self.subspace} invalid for target "
                f"dimension {dim}: indices must satisfy 0 <= j, k < dim"
            )


@dataclass(frozen=True)
class SubspaceRY(Operation):
    """Apply `RY` inside two selected levels and leave other levels unchanged.

    In the ordered pair ``(j, k)``, ``j`` has RY's ``|0>`` role and ``k`` has
    its ``|1>`` role, so reversing the pair reverses the rotation direction.
    ``SubspaceRY(theta, (0, 1))`` on a qubit is ``RY`` with the same ``theta``.

    Args:
        theta: Numeric angle in radians, or a `fatqat.Parameter` to bind before
            execution.
        subspace: Tuple of exactly two distinct, non-negative integer level
            indices in ``(|0>, |1>)`` role order.

    Raises:
        ValueError: At construction if ``subspace`` does not contain exactly
            two values, its indices are equal, or an index is negative; or from
            `fatqat.Program.add` if an index is outside the target dimension.
    """

    theta: float | Parameter
    subspace: tuple[int, int]
    name: ClassVar[str] = "SubspaceRY"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True

    def __post_init__(self) -> None:
        _validate_angles(self, theta=self.theta)
        j, k = self.subspace
        if j == k:
            raise ValueError(
                f"SubspaceRY subspace requires distinct levels, got ({j}, {k})"
            )
        if j < 0 or k < 0:
            raise ValueError(
                f"SubspaceRY subspace levels must be non-negative, got ({j}, {k})"
            )

    def validate_targets(self, targets: tuple[RegisterRef, ...]) -> None:
        """Validate the selected subspace against the resolved target.

        Args:
            targets: One resolved scalar `fatqat.RegisterRef`.

        Raises:
            ValueError: If either subspace index is not less than the target's
                local dimension.
        """
        dim = targets[0].register.dim
        j, k = self.subspace
        if j >= dim or k >= dim:
            raise ValueError(
                f"SubspaceRY subspace {self.subspace} invalid for target "
                f"dimension {dim}: indices must satisfy 0 <= j, k < dim"
            )


@dataclass(frozen=True)
class SubspaceRZ(Operation):
    """Apply `RZ` inside two selected levels and leave other levels unchanged.

    In the ordered pair ``(j, k)``, ``j`` gains phase ``exp(-i*theta/2)`` and
    ``k`` gains ``exp(i*theta/2)``. Reversing the pair reverses the rotation.
    ``SubspaceRZ(theta, (0, 1))`` on a qubit is ``RZ`` with the same ``theta``.

    Args:
        theta: Numeric angle in radians, or a `fatqat.Parameter` to bind before
            execution.
        subspace: Tuple of exactly two distinct, non-negative integer level
            indices in ``(|0>, |1>)`` role order.

    Raises:
        ValueError: At construction if ``subspace`` does not contain exactly
            two values, its indices are equal, or an index is negative; or from
            `fatqat.Program.add` if an index is outside the target dimension.
    """

    theta: float | Parameter
    subspace: tuple[int, int]
    name: ClassVar[str] = "SubspaceRZ"
    num_subsystems: ClassVar[int] = 1
    _accepts_views: ClassVar[bool] = True

    def __post_init__(self) -> None:
        _validate_angles(self, theta=self.theta)
        j, k = self.subspace
        if j == k:
            raise ValueError(
                f"SubspaceRZ subspace requires distinct levels, got ({j}, {k})"
            )
        if j < 0 or k < 0:
            raise ValueError(
                f"SubspaceRZ subspace levels must be non-negative, got ({j}, {k})"
            )

    def validate_targets(self, targets: tuple[RegisterRef, ...]) -> None:
        """Validate the selected subspace against the resolved target.

        Args:
            targets: One resolved scalar `fatqat.RegisterRef`.

        Raises:
            ValueError: If either subspace index is not less than the target's
                local dimension.
        """
        dim = targets[0].register.dim
        j, k = self.subspace
        if j >= dim or k >= dim:
            raise ValueError(
                f"SubspaceRZ subspace {self.subspace} invalid for target "
                f"dimension {dim}: indices must satisfy 0 <= j, k < dim"
            )


@dataclass(frozen=True)
class CClock(Operation):
    """Apply a control-level-dependent `Clock` phase to a target qudit.

    Targets are ``(control, target)``. On basis state ``|i, j>``, the phase is
    ``omega**(i*j*power)``, where
    ``omega = exp(2*pi*i/target_dimension)``. Control and target dimensions
    may differ. Negative and oversized integer powers are valid and equivalent
    modulo the target dimension. On two qubits, ``CClock(1)`` is ``CZ``.

    Args:
        power: Integer phase power.
    """

    power: int
    name: ClassVar[str] = "CClock"
    num_subsystems: ClassVar[int] = 2
    _accepts_views: ClassVar[bool] = True
