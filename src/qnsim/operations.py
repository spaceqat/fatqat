"""Operation base class and the built-in gate set, exposed as the `qs.ops` namespace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

__all__ = [
    "Operation",
    "I", "H", "S", "Sdg", "T", "Tdg", "X", "Y", "Z",
    "CX", "CZ",
    "RX", "RY", "RZ", "Phase",
    "Reset",
]


# ---------------------------------------------------------------------------
# Operation base class
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Operation:
    """Base class for immutable operation objects.

    Fixed gates are exposed as pre-built singleton values in `qnsim.ops`.
    Parametric gates are exposed as classes and should be instantiated, such as
    `RX(theta)`.

    Attributes:
        name: Public operation name.
        _num_qubits: Number of quantum targets required by the operation, or
            None for variable arity with at least one target.
    """

    name: ClassVar[str] = "OP"
    _num_qubits: ClassVar[int | None] = 1

    def __post_init__(self) -> None:
        n = type(self)._num_qubits
        if n is None:
            return
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError(f"_num_qubits must be a positive int or None, got {n!r}")

    @property
    def num_qubits(self) -> int | None:
        """Number of quantum targets required, or None for variable arity."""
        return type(self)._num_qubits


# ---------------------------------------------------------------------------
# Fixed single-qubit unitary gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HGate(Operation):
    """Hadamard gate operation."""

    name: ClassVar[str] = "H"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class IGate(Operation):
    """Identity gate operation."""

    name: ClassVar[str] = "I"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class SGate(Operation):
    """S phase gate operation (square root of Z)."""

    name: ClassVar[str] = "S"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class SdgGate(Operation):
    """Inverse S phase gate operation."""

    name: ClassVar[str] = "Sdg"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class TGate(Operation):
    """T phase gate operation."""

    name: ClassVar[str] = "T"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class TdgGate(Operation):
    """Inverse T phase gate operation."""

    name: ClassVar[str] = "Tdg"
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


# ---------------------------------------------------------------------------
# Fixed multi-qubit unitary gates (2+ qubits)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Parametric single-qubit unitary gates
# ---------------------------------------------------------------------------


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
class Phase(Operation):
    """General single-qubit phase gate: diag(1, e^{i theta}).

    Attributes:
        theta: Phase angle in radians.
    """

    theta: float
    name: ClassVar[str] = "Phase"
    _num_qubits: ClassVar[int] = 1


# ---------------------------------------------------------------------------
# Non-unitary frontend operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResetGate(Operation):
    """Reset operation: repreparation of one or more target qubits in ``|0>``.

    Has no matrix; the matrix-family backend resolves it to a boundary reset
    step by operation type.
    """

    name: ClassVar[str] = "Reset"
    _num_qubits: ClassVar[int | None] = None


# ---------------------------------------------------------------------------
# Public fixed-gate instances
# ---------------------------------------------------------------------------
# Fixed gates (no parameters) are exported as singleton values. Parametric
# gates are exported as classes above and instantiated by callers, e.g.
# RX(theta). `Reset` takes no parameters, so it follows the fixed-gate rule
# too: `qs.ops.Reset`, not `qs.ops.Reset()`.

H = HGate()
I = IGate()
S = SGate()
Sdg = SdgGate()
T = TGate()
Tdg = TdgGate()
X = XGate()
Y = YGate()
Z = ZGate()
CX = CXGate()
CZ = CZGate()
Reset = ResetGate()
