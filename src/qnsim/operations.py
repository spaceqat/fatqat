"""Operation base class and the built-in gate set, exposed as the `qs.ops` namespace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

__all__ = [
    "Operation",
    "I", "H", "S", "Sdg", "T", "Tdg", "X", "Y", "Z",
    "CX", "CZ", "Swap", "CY", "CS", "iSwap", "CCX", "CSwap",
    "RX", "RY", "RZ", "Phase",
    "CPhase",
    "Reset",
    "Shift", "Clock", "Sum", "SumGate",
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
        _num_subsystems: Number of quantum targets required by the operation, or
            None for variable arity with at least one target.
    """

    name: ClassVar[str] = "OP"
    _num_subsystems: ClassVar[int | None] = 1

    def __post_init__(self) -> None:
        n = type(self)._num_subsystems
        if n is None:
            return
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError(f"_num_subsystems must be a positive int or None, got {n!r}")

    @property
    def num_subsystems(self) -> int | None:
        """Number of quantum targets required, or None for variable arity."""
        return type(self)._num_subsystems


# ---------------------------------------------------------------------------
# Fixed single-qubit unitary gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HGate(Operation):
    """Hadamard gate operation."""

    name: ClassVar[str] = "H"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class IGate(Operation):
    """Identity gate operation."""

    name: ClassVar[str] = "I"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class SGate(Operation):
    """S phase gate operation (square root of Z)."""

    name: ClassVar[str] = "S"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class SdgGate(Operation):
    """Inverse S phase gate operation."""

    name: ClassVar[str] = "Sdg"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class TGate(Operation):
    """T phase gate operation."""

    name: ClassVar[str] = "T"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class TdgGate(Operation):
    """Inverse T phase gate operation."""

    name: ClassVar[str] = "Tdg"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class XGate(Operation):
    """Pauli-X gate operation."""

    name: ClassVar[str] = "X"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class YGate(Operation):
    """Pauli-Y gate operation."""

    name: ClassVar[str] = "Y"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class ZGate(Operation):
    """Pauli-Z gate operation."""

    name: ClassVar[str] = "Z"
    _num_subsystems: ClassVar[int] = 1


# ---------------------------------------------------------------------------
# Fixed multi-qubit unitary gates (2+ qubits)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CXGate(Operation):
    """Controlled-X gate operation."""

    name: ClassVar[str] = "CX"
    _num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class CZGate(Operation):
    """Controlled-Z gate operation."""

    name: ClassVar[str] = "CZ"
    _num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class SwapGate(Operation):
    """Swap gate operation: exchanges the state of its two targets."""

    name: ClassVar[str] = "Swap"
    _num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class CYGate(Operation):
    """Controlled-Y gate operation.

    ``targets = (control, target)``; operand 0 is the control.
    """

    name: ClassVar[str] = "CY"
    _num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class CSGate(Operation):
    """Controlled-S gate operation.

    ``targets = (control, target)``; operand 0 is the control.
    """

    name: ClassVar[str] = "CS"
    _num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class iSwapGate(Operation):
    """iSWAP gate operation: swaps its two targets, applying an i phase to
    the swapped amplitudes.
    """

    name: ClassVar[str] = "iSwap"
    _num_subsystems: ClassVar[int] = 2


@dataclass(frozen=True)
class CCXGate(Operation):
    """Doubly-controlled-X (Toffoli) gate operation.

    ``targets = (control0, control1, target)``; operands 0 and 1 are the
    controls.
    """

    name: ClassVar[str] = "CCX"
    _num_subsystems: ClassVar[int] = 3


@dataclass(frozen=True)
class CSwapGate(Operation):
    """Controlled-swap (Fredkin) gate operation.

    ``targets = (control, target0, target1)``; operand 0 is the control.
    """

    name: ClassVar[str] = "CSwap"
    _num_subsystems: ClassVar[int] = 3


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
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class RY(Operation):
    """Rotation around the Y axis.

    Attributes:
        theta: Rotation angle in radians.
    """

    theta: float
    name: ClassVar[str] = "RY"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class RZ(Operation):
    """Rotation around the Z axis.

    Attributes:
        theta: Rotation angle in radians.
    """

    theta: float
    name: ClassVar[str] = "RZ"
    _num_subsystems: ClassVar[int] = 1


@dataclass(frozen=True)
class Phase(Operation):
    """General single-qubit phase gate: diag(1, e^{i theta}).

    Attributes:
        theta: Phase angle in radians.
    """

    theta: float
    name: ClassVar[str] = "Phase"
    _num_subsystems: ClassVar[int] = 1


# ---------------------------------------------------------------------------
# Parametric controlled / multi-qubit unitary gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CPhase(Operation):
    """Controlled phase gate: applies diag(1, e^{i theta}) to the target when
    the control is |1>.

    ``targets = (control, target)``; operand 0 is the control.

    Attributes:
        theta: Phase angle in radians.
    """

    theta: float
    name: ClassVar[str] = "CPhase"
    _num_subsystems: ClassVar[int] = 2


# ---------------------------------------------------------------------------
# Non-unitary frontend operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResetGate(Operation):
    """Reset operation: repreparation of one or more target subsystems in ``|0>``.

    Has no matrix; the matrix-family backend resolves it to a boundary reset
    step by operation type.
    """

    name: ClassVar[str] = "Reset"
    _num_subsystems: ClassVar[int | None] = None


# ---------------------------------------------------------------------------
# Dimension-generic gates (generalized Pauli group + controlled add)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shift(Operation):
    """Generalized-Pauli cyclic shift: ``|k> -> |(k + power) mod d>``.

    Dimension-free: applies to a subsystem of any dimension; its matrix is
    built from the target dimension at backend lowering. Reduces to X at
    ``dim=2, power=1``.

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
    """

    name: ClassVar[str] = "Sum"
    _num_subsystems: ClassVar[int] = 2


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
Swap = SwapGate()
CY = CYGate()
CS = CSGate()
iSwap = iSwapGate()
CCX = CCXGate()
CSwap = CSwapGate()
Reset = ResetGate()
Sum = SumGate()
