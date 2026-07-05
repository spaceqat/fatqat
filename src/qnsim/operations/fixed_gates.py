"""Fixed (parameter-free) unitary gates: single-qubit and multi-qubit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import Operation

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
# Public fixed-gate instances
# ---------------------------------------------------------------------------
# These classes have no parameters, so each is exported only as a singleton
# value (e.g. `qs.ops.H`), not as a class - unlike parametric gates, there is
# no reason for a caller to ever name `HGate` itself.

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
