"""Target-independent immutable logical gate IR."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

from ...operations import Operation
from ...registers import ClassicalRegister, QuantumRegister, RegisterRef
from ..errors import ValidationError


@dataclass(frozen=True, slots=True)
class LogicalGate:
    operation_id: str
    operation: Operation
    operands: tuple[RegisterRef, ...]


@dataclass(frozen=True, slots=True)
class LogicalMeasure:
    operation_id: str
    qubit: RegisterRef
    clbit: RegisterRef


LogicalInstruction: TypeAlias = LogicalGate | LogicalMeasure


@dataclass(frozen=True, slots=True)
class LogicalProgram:
    IR_ID: ClassVar[str] = "gate.logical.v1"

    qubits: tuple[RegisterRef, ...]
    clbits: tuple[RegisterRef, ...]
    instructions: tuple[LogicalInstruction, ...]


def verify_logical_program(program: object) -> None:
    if type(program) is not LogicalProgram:
        raise ValidationError("expected LogicalProgram")
    _verify_refs(program.qubits, QuantumRegister, "qubit")
    _verify_refs(program.clbits, ClassicalRegister, "clbit")

    known_qubits = set(program.qubits)
    known_clbits = set(program.clbits)
    operation_ids: set[str] = set()
    measurement_started = False
    written_clbits: set[RegisterRef] = set()

    for instruction in program.instructions:
        operation_id = instruction.operation_id
        if not isinstance(operation_id, str) or not operation_id:
            raise ValidationError("logical operation ID must be a non-empty string")
        if operation_id in operation_ids:
            raise ValidationError(f"duplicate logical operation ID: {operation_id}")
        operation_ids.add(operation_id)

        if type(instruction) is LogicalGate:
            if measurement_started:
                raise ValidationError("logical measurements must be terminal")
            if not isinstance(instruction.operation, Operation):
                raise ValidationError("logical gate must contain a FatQat Operation")
            expected = instruction.operation.num_subsystems
            if expected is None and not instruction.operands:
                raise ValidationError(
                    f"{instruction.operation.name} expects at least one operand"
                )
            if expected is not None and len(instruction.operands) != expected:
                raise ValidationError(
                    f"{instruction.operation.name} expects {expected} operand(s), "
                    f"got {len(instruction.operands)}"
                )
            if any(item not in known_qubits for item in instruction.operands):
                raise ValidationError("logical gate references an undeclared qubit")
            if len(set(instruction.operands)) != len(instruction.operands):
                raise ValidationError("logical gate repeats a qubit operand")
            _verify_numeric_angle(instruction.operation)
            continue

        if type(instruction) is not LogicalMeasure:
            raise ValidationError(
                f"unsupported logical instruction: {type(instruction).__name__}"
            )
        measurement_started = True
        if instruction.qubit not in known_qubits:
            raise ValidationError("measurement references an undeclared qubit")
        if instruction.clbit not in known_clbits:
            raise ValidationError("measurement references an undeclared clbit")
        if instruction.clbit in written_clbits:
            raise ValidationError("logical measurement writes a clbit more than once")
        written_clbits.add(instruction.clbit)


def _verify_refs(
    refs: tuple[RegisterRef, ...],
    expected_register: type[QuantumRegister] | type[ClassicalRegister],
    label: str,
) -> None:
    if not isinstance(refs, tuple):
        raise ValidationError(f"{label} refs must be a tuple")
    if any(type(item) is not RegisterRef for item in refs):
        raise ValidationError(f"{label} refs must contain exact RegisterRef values")
    if any(not isinstance(item.register, expected_register) for item in refs):
        raise ValidationError(f"{label} ref has the wrong kind")
    if len(set(refs)) != len(refs):
        raise ValidationError(f"duplicate declared {label} ref")


def _verify_numeric_angle(operation: Operation) -> None:
    if not hasattr(operation, "theta"):
        return
    theta = operation.theta
    if (
        not isinstance(theta, numbers.Real)
        or isinstance(theta, bool)
        or not math.isfinite(float(theta))
    ):
        raise ValidationError(
            f"{type(operation).__name__}.theta must be a finite real number"
        )
