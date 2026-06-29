"""Program container plus AppliedOperation / Measurement value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .operations import Operation
from .registers import QuantumRegister, RegisterRef

ConditionTerm = tuple[RegisterRef, int]
Condition = tuple[ConditionTerm, ...] | None


@dataclass(frozen=True)
class AppliedOperation:
    operation: Operation
    targets: tuple[RegisterRef, ...]
    condition: Condition = None

    def __post_init__(self) -> None:
        if not isinstance(self.targets, tuple):
            raise TypeError("targets must be a tuple of RegisterRef")
        expected = self.operation.num_qubits
        if len(self.targets) != expected:
            raise ValueError(
                f"{self.operation.name} expects {expected} target(s), "
                f"got {len(self.targets)}"
            )
        for t in self.targets:
            if not isinstance(t, RegisterRef):
                raise TypeError(f"target must be RegisterRef, got {type(t)!r}")
            if not isinstance(t.register, QuantumRegister):
                raise TypeError("operation targets must reference a QuantumRegister")


@dataclass(frozen=True)
class Measurement:
    qreg: RegisterRef
    clreg: RegisterRef
    metadata: Mapping[str, Any] = field(default_factory=dict)
