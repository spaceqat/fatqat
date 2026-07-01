"""Operation base and the Phase 1 gate set, exposed as the `qs.ops` namespace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Operation:
    """Base class for immutable operation objects.

    Fixed gates are exposed as pre-built singleton values in `qnsim.ops`.
    Parametric gates are exposed as classes and should be instantiated, such as
    `RX(theta)`.

    Attributes:
        name: Public operation name.
        _num_qubits: Number of quantum targets required by the operation.
    """

    name: ClassVar[str] = "OP"
    _num_qubits: ClassVar[int] = 1

    def __post_init__(self) -> None:
        n = type(self)._num_qubits
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError(f"_num_qubits must be a positive int, got {n!r}")

    @property
    def num_qubits(self) -> int:
        """Number of quantum targets required by this operation."""
        return type(self)._num_qubits


@dataclass(frozen=True)
class HGate(Operation):
    """Hadamard gate operation."""

    name: ClassVar[str] = "H"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class TGate(Operation):
    """T phase gate operation."""

    name: ClassVar[str] = "T"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class XGate(Operation):
    """Pauli-X gate operation."""

    name: ClassVar[str] = "X"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class YGate(Operation):
    """Pauli-Y gate operation."""

    name: ClassVar[str] = "Y"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class ZGate(Operation):
    """Pauli-Z gate operation."""

    name: ClassVar[str] = "Z"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class CXGate(Operation):
    """Controlled-X gate operation."""

    name: ClassVar[str] = "CX"
    _num_qubits: ClassVar[int] = 2


@dataclass(frozen=True)
class CZGate(Operation):
    """Controlled-Z gate operation."""

    name: ClassVar[str] = "CZ"
    _num_qubits: ClassVar[int] = 2


@dataclass(frozen=True)
class RX(Operation):
    """Rotation around the X axis.

    Attributes:
        theta: Rotation angle in radians.
    """

    theta: float
    name: ClassVar[str] = "RX"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class RY(Operation):
    """Rotation around the Y axis.

    Attributes:
        theta: Rotation angle in radians.
    """

    theta: float
    name: ClassVar[str] = "RY"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class RZ(Operation):
    """Rotation around the Z axis.

    Attributes:
        theta: Rotation angle in radians.
    """

    theta: float
    name: ClassVar[str] = "RZ"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class ResetGate(Operation):
    """Reset operation: repreparation of a target qubit in ``|0>``.

    Has no matrix; the matrix-family backend resolves it to a boundary reset
    step by operation type.
    """

    name: ClassVar[str] = "Reset"
    _num_qubits: ClassVar[int] = 1


# Pre-built fixed-gate instances (parametric gates are used as classes: RX(theta)).
H = HGate()
T = TGate()
X = XGate()
Y = YGate()
Z = ZGate()
CX = CXGate()
CZ = CZGate()

# Reset takes no parameters but is used with call syntax: qs.ops.Reset().
Reset = ResetGate
