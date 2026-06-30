"""Program container plus AppliedOperation / Measurement value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .operations import Operation
from .registers import QuantumRegister, RegisterRef, ClassicalRegister

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

    def __post_init__(self) -> None:
        # Copy metadata so a caller's later mutation can't reach into this frozen
        # value object.
        object.__setattr__(self, "metadata", dict(self.metadata))


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

    def _resolve_ref(self, operand, regs, kind_cls, kind_name) -> RegisterRef:
        if isinstance(operand, RegisterRef):
            if not isinstance(operand.register, kind_cls):
                raise TypeError(f"expected a {kind_name} ref")
            if not any(operand.register is r for r in regs):
                raise ValueError(f"ref does not belong to this program's {kind_name}s")
            return operand
        if not isinstance(operand, int) or isinstance(operand, bool):
            raise TypeError(f"operand must be int or RegisterRef, got {type(operand)!r}")
        if len(regs) != 1:
            raise TypeError(
                "integer operands are only allowed when there is exactly one "
                "register of the relevant kind; pass an explicit RegisterRef "
                "(e.g. qreg[0] or creg[0]) instead"
            )
        # Bounds and negative-index checks are delegated to Register.__getitem__,
        # which raises IndexError.
        return regs[0][operand]

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
        if condition is None:
            return None
        # Discriminate single term `(slot, lit)` from a sequence of terms.
        terms = condition if isinstance(condition[0], tuple) else (condition,)
        return tuple(
            (self._resolve_classical_slot(slot), int(literal))
            for slot, literal in terms
        )

    def _resolve_classical_slot(self, slot) -> RegisterRef:
        if isinstance(slot, RegisterRef):
            if not isinstance(slot.register, ClassicalRegister):
                raise TypeError("condition slot ref must reference a ClassicalRegister")
            if not any(slot.register is r for r in self.creg):
                raise ValueError("condition slot ref not in this program")
            return slot
        return self._resolve_clbit(slot)

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

    def copy(self) -> "Program":
        new = Program.__new__(Program)
        new.qreg = list(self.qreg)
        new.creg = list(self.creg)
        new.operations = list(self.operations)
        new.metadata = dict(self.metadata)
        return new
