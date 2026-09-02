"""Closed neutral-atom gate IR."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

from ...operations.fixed_gates import CZGate
from ...operations.parametric_gates import RX, RY, RZ
from ...registers import ClassicalRegister, QuantumRegister, RegisterRef
from ..errors import ValidationError


@dataclass(frozen=True, slots=True)
class NAGate:
    operation_id: str
    origin_ids: tuple[str, ...]
    operation: RX | RY | RZ | CZGate
    atoms: tuple[RegisterRef, ...]


@dataclass(frozen=True, slots=True)
class NAMeasure:
    operation_id: str
    origin_ids: tuple[str, ...]
    atom: RegisterRef
    clbit: RegisterRef


NAInstruction: TypeAlias = NAGate | NAMeasure


@dataclass(frozen=True, slots=True)
class NAProgram:
    IR_ID: ClassVar[str] = "na.gate.v1"

    atoms: tuple[RegisterRef, ...]
    clbits: tuple[RegisterRef, ...]
    instructions: tuple[NAInstruction, ...]


def verify_na_program(program: object) -> None:
    """Validate facts fully contained in a standalone NAProgram."""

    if type(program) is not NAProgram:
        raise ValidationError("expected NAProgram")
    _verify_refs(program.atoms, QuantumRegister, "atom")
    _verify_refs(program.clbits, ClassicalRegister, "clbit")
    if not isinstance(program.instructions, tuple):
        raise ValidationError("NA instructions must be a tuple")

    known_atoms = set(program.atoms)
    known_clbits = set(program.clbits)
    operation_ids: set[str] = set()
    written_clbits: set[RegisterRef] = set()
    measurement_started = False

    for instruction in program.instructions:
        if type(instruction) not in (NAGate, NAMeasure):
            raise ValidationError(
                f"unsupported NA instruction: {type(instruction).__name__}"
            )
        _verify_operation_metadata(instruction, operation_ids)

        if type(instruction) is NAGate:
            if measurement_started:
                raise ValidationError("NA measurements must be terminal")
            if not isinstance(instruction.atoms, tuple):
                raise ValidationError("NA gate atoms must be a tuple")
            expected_arity = _GATE_ARITIES.get(type(instruction.operation))
            if expected_arity is None:
                raise ValidationError(
                    f"unsupported NA gate: {type(instruction.operation).__name__}"
                )
            if len(instruction.atoms) != expected_arity:
                raise ValidationError("invalid NA gate operand shape")
            if any(atom not in known_atoms for atom in instruction.atoms):
                raise ValidationError("NA gate references an undeclared atom")
            if len(set(instruction.atoms)) != len(instruction.atoms):
                raise ValidationError("NA gate repeats an atom operand")
            _verify_rotation(instruction.operation)
            continue

        measurement_started = True
        if instruction.atom not in known_atoms:
            raise ValidationError("NA measurement references an undeclared atom")
        if instruction.clbit not in known_clbits:
            raise ValidationError("NA measurement references an undeclared clbit")
        if instruction.clbit in written_clbits:
            raise ValidationError("NA measurement writes a clbit more than once")
        written_clbits.add(instruction.clbit)


_GATE_ARITIES: dict[type, int] = {RX: 1, RY: 1, RZ: 1, CZGate: 2}


def _verify_refs(
    refs: tuple[RegisterRef, ...],
    expected_register: type[QuantumRegister] | type[ClassicalRegister],
    label: str,
) -> None:
    if not isinstance(refs, tuple):
        raise ValidationError(f"NA {label} refs must be a tuple")
    if any(type(item) is not RegisterRef for item in refs):
        raise ValidationError(f"NA {label} refs must contain exact RegisterRef values")
    if any(not isinstance(item.register, expected_register) for item in refs):
        raise ValidationError(f"NA {label} ref has the wrong kind")
    if len(set(refs)) != len(refs):
        raise ValidationError(f"duplicate declared NA {label} ref")


def _verify_operation_metadata(
    instruction: NAInstruction, operation_ids: set[str]
) -> None:
    if not isinstance(instruction.operation_id, str) or not instruction.operation_id:
        raise ValidationError("NA operation ID must be a non-empty string")
    if instruction.operation_id in operation_ids:
        raise ValidationError(f"duplicate NA operation ID: {instruction.operation_id}")
    operation_ids.add(instruction.operation_id)
    if (
        not isinstance(instruction.origin_ids, tuple)
        or not instruction.origin_ids
        or any(not isinstance(item, str) or not item for item in instruction.origin_ids)
        or len(set(instruction.origin_ids)) != len(instruction.origin_ids)
    ):
        raise ValidationError("NA origin IDs must be non-empty and unique")


def _verify_rotation(operation: RX | RY | RZ | CZGate) -> None:
    if type(operation) not in (RX, RY, RZ):
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
