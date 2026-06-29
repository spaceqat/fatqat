"""Program container plus AppliedOperation / Measurement value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .operations import Operation
from .registers import QuantumRegister, RegisterRef, ClassicalRegister, Register

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


class Program:
    def __init__(
        self,
        qreg: int | list[QuantumRegister],
        clreg: int | list[ClassicalRegister] = 0,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.qreg: list[QuantumRegister] = self._coerce_registers(
            qreg, QuantumRegister, "q"
        )
        self.creg: list[ClassicalRegister] = self._coerce_registers(
            clreg, ClassicalRegister, "c"
        )
        self.operations: list[AppliedOperation | Measurement] = []
        self.metadata: dict[str, Any] = dict(metadata) if metadata else {}

    @classmethod
    def registers(
        cls,
        *,
        qreg: list[QuantumRegister],
        clreg: list[ClassicalRegister] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Program":
        p = cls.__new__(cls)
        p.qreg = list(qreg)
        p.creg = list(clreg) if clreg is not None else []
        p.operations = []
        p.metadata = dict(metadata) if metadata else {}
        return p

    @staticmethod
    def _coerce_registers(spec, cls, default_name):
        # TODO: why bool? We should only allow int
        if isinstance(spec, int) and not isinstance(spec, bool):
            if spec < 0:
                raise ValueError(f"register count must be >= 0, got {spec}")
            return [cls(spec, name=default_name)] if spec > 0 else []
        return list(spec)

    @staticmethod
    def _flat_from_int(i: int, regs: list[Register]) -> RegisterRef:
        if not isinstance(i, int) or isinstance(i, bool):
            raise TypeError(f"operand must be int or RegisterRef, got {type(i)!r}")
        if i < 0:
            raise IndexError(i)
        remaining = i
        for reg in regs:
            if remaining < reg.size:
                return reg[remaining]
            remaining -= reg.size
        raise IndexError(i)

    def _resolve_ref(self, operand, regs, kind_cls, kind_name) -> RegisterRef:
        if isinstance(operand, RegisterRef):
            if not isinstance(operand.register, kind_cls):
                raise TypeError(f"expected a {kind_name} ref")
            if not any(operand.register is r for r in regs):
                raise ValueError(f"ref does not belong to this program's {kind_name}s")
            return operand
        return self._flat_from_int(operand, regs)

    def _resolve_qubit(self, operand) -> RegisterRef:
        return self._resolve_ref(operand, self.qreg, QuantumRegister, "quantum register")

    def _resolve_clbit(self, operand) -> RegisterRef:
        return self._resolve_ref(
            operand, self.creg, ClassicalRegister, "classical register"
        )

    def add(
        self,
        op: Operation,
        qreg: int | RegisterRef | tuple[int | RegisterRef, ...],
        *,
        condition=None,
    ) -> None:
        if not isinstance(op, Operation):
            raise TypeError(
                f"op must be an Operation instance, got {type(op)!r} "
                "(did you forget to call a parametric gate, e.g. ops.RX(0.2)?)"
            )
        operands = qreg if isinstance(qreg, tuple) else (qreg,)
        targets = tuple(self._resolve_qubit(o) for o in operands)
        normalized = self._normalize_condition(condition)
        self.operations.append(
            AppliedOperation(operation=op, targets=targets, condition=normalized)
        )

    def _normalize_condition(self, condition):
        # Full normalization arrives in Task 7; until then only None is supported.
        if condition is None:
            return None
        raise NotImplementedError("condition normalization not yet implemented")

    def add_measurement(
        self,
        qreg: int | RegisterRef,
        clreg: int | RegisterRef,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        q = self._resolve_qubit(qreg)
        c = self._resolve_clbit(clreg)
        self.operations.append(
            Measurement(qreg=q, clreg=c, metadata=dict(metadata) if metadata else {})
        )
