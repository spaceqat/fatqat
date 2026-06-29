"""Operation base and the Phase 1 gate set, exposed as the `qs.ops` namespace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Operation:
    name: ClassVar[str] = "OP"
    _num_qubits: ClassVar[int] = 1

    def __post_init__(self) -> None:
        n = type(self)._num_qubits
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError(f"_num_qubits must be a positive int, got {n!r}")

    @property
    def num_qubits(self) -> int:
        return type(self)._num_qubits


@dataclass(frozen=True)
class HGate(Operation):
    name: ClassVar[str] = "H"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class TGate(Operation):
    name: ClassVar[str] = "T"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class XGate(Operation):
    name: ClassVar[str] = "X"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class YGate(Operation):
    name: ClassVar[str] = "Y"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class ZGate(Operation):
    name: ClassVar[str] = "Z"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class CXGate(Operation):
    name: ClassVar[str] = "CX"
    _num_qubits: ClassVar[int] = 2


@dataclass(frozen=True)
class CZGate(Operation):
    name: ClassVar[str] = "CZ"
    _num_qubits: ClassVar[int] = 2


@dataclass(frozen=True)
class RX(Operation):
    theta: float
    name: ClassVar[str] = "RX"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class RY(Operation):
    theta: float
    name: ClassVar[str] = "RY"
    _num_qubits: ClassVar[int] = 1


@dataclass(frozen=True)
class RZ(Operation):
    theta: float
    name: ClassVar[str] = "RZ"
    _num_qubits: ClassVar[int] = 1


# Pre-built fixed-gate instances (parametric gates are used as classes: RX(theta)).
H = HGate()
T = TGate()
X = XGate()
Y = YGate()
Z = ZGate()
CX = CXGate()
CZ = CZGate()
