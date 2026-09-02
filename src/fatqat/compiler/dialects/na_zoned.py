"""Normalized neutral-atom physical-plan IR."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import ClassVar, Literal, TypeAlias

from ...operations.fixed_gates import CZGate
from ...operations.parametric_gates import RX, RY, RZ
from ...registers import ClassicalRegister, QuantumRegister, RegisterRef
from ..errors import ValidationError
from .na_gate import NAMeasure, NAProgram, verify_na_program

Position: TypeAlias = tuple[float, float]


@dataclass(frozen=True, slots=True)
class TransferEvent:
    kind: Literal["activate", "deactivate"]
    atoms: tuple[RegisterRef, ...]
    positions: tuple[Position, ...]
    durations: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MoveEvent:
    kind: Literal["big_move", "park"]
    atoms: tuple[RegisterRef, ...]
    starts: tuple[Position, ...]
    ends: tuple[Position, ...]
    distances: tuple[float, ...]
    durations: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ScheduledGate:
    operation_id: str
    origin_ids: tuple[str, ...]
    operation: RX | RY | RZ | CZGate
    atoms: tuple[RegisterRef, ...]
    positions: tuple[Position, ...]


@dataclass(frozen=True, slots=True)
class GateBatch:
    stage: int
    gates: tuple[ScheduledGate, ...]
    duration: float


@dataclass(frozen=True, slots=True)
class CrosstalkEvent:
    atoms: tuple[RegisterRef, ...]
    positions: tuple[Position, ...]
    durations: tuple[float, ...]


ZonedEvent: TypeAlias = TransferEvent | MoveEvent | GateBatch | CrosstalkEvent


@dataclass(frozen=True, slots=True)
class ZonedPlan:
    IR_ID: ClassVar[str] = "na.zoned.plan.v1"

    atoms: tuple[RegisterRef, ...]
    clbits: tuple[RegisterRef, ...]
    initial_placement: tuple[tuple[RegisterRef, Position], ...]
    events: tuple[ZonedEvent, ...]
    terminal_measurements: tuple[NAMeasure, ...]


def verify_zoned_plan(plan: object) -> None:
    """Validate a physical plan and replay its atom positions locally."""

    if type(plan) is not ZonedPlan:
        raise ValidationError("expected ZonedPlan")
    _verify_declared_refs(plan.atoms, QuantumRegister, "atom")
    _verify_declared_refs(plan.clbits, ClassicalRegister, "clbit")

    position_by_atom = _verify_initial_placement(plan)
    if not isinstance(plan.events, tuple):
        raise ValidationError("ZonedPlan events must be a tuple")

    known_atoms = set(plan.atoms)
    operation_ids: set[str] = set()
    for event in plan.events:
        if type(event) is TransferEvent:
            _verify_transfer(event, known_atoms, position_by_atom)
        elif type(event) is MoveEvent:
            _verify_move(event, known_atoms, position_by_atom)
        elif type(event) is GateBatch:
            _verify_gate_batch(event, known_atoms, position_by_atom, operation_ids)
        elif type(event) is CrosstalkEvent:
            _verify_crosstalk(event, known_atoms, position_by_atom)
        else:
            raise ValidationError(
                f"unsupported ZonedPlan event: {type(event).__name__}"
            )
    _verify_terminal_measurements(plan, operation_ids)


def _verify_declared_refs(
    refs: tuple[RegisterRef, ...],
    expected_register: type[QuantumRegister] | type[ClassicalRegister],
    label: str,
) -> None:
    if not isinstance(refs, tuple):
        raise ValidationError(f"ZonedPlan {label} refs must be a tuple")
    if any(type(ref) is not RegisterRef for ref in refs):
        raise ValidationError(
            f"ZonedPlan {label} refs must contain exact RegisterRef values"
        )
    if any(not isinstance(ref.register, expected_register) for ref in refs):
        raise ValidationError(f"ZonedPlan {label} ref has the wrong kind")
    if len(set(refs)) != len(refs):
        raise ValidationError(f"duplicate declared ZonedPlan {label} ref")


def _verify_initial_placement(plan: ZonedPlan) -> dict[RegisterRef, Position]:
    placement = plan.initial_placement
    if not isinstance(placement, tuple):
        raise ValidationError("initial placement must be a tuple")

    positions: dict[RegisterRef, Position] = {}
    for entry in placement:
        if type(entry) is not tuple or len(entry) != 2:
            raise ValidationError(
                "initial placement entries must be (RegisterRef, Position)"
            )
        atom, position = entry
        if type(atom) is not RegisterRef or atom not in plan.atoms:
            raise ValidationError("initial placement references an undeclared atom")
        if atom in positions:
            raise ValidationError("initial placement repeats an atom")
        _verify_position(position, "initial placement position")
        positions[atom] = position

    if set(positions) != set(plan.atoms):
        raise ValidationError(
            "initial placement must cover each declared atom exactly once"
        )
    return positions


def _verify_terminal_measurements(plan: ZonedPlan, operation_ids: set[str]) -> None:
    if not isinstance(plan.terminal_measurements, tuple):
        raise ValidationError("terminal measurements must be a tuple")
    if any(
        type(measurement) is not NAMeasure for measurement in plan.terminal_measurements
    ):
        raise ValidationError(
            "terminal measurements must contain exact NAMeasure values"
        )
    verify_na_program(NAProgram(plan.atoms, plan.clbits, plan.terminal_measurements))
    for measurement in plan.terminal_measurements:
        if measurement.operation_id in operation_ids:
            raise ValidationError(
                f"duplicate ZonedPlan operation ID: {measurement.operation_id}"
            )
        operation_ids.add(measurement.operation_id)


def _verify_transfer(
    event: TransferEvent,
    known_atoms: set[RegisterRef],
    position_by_atom: dict[RegisterRef, Position],
) -> None:
    if event.kind not in ("activate", "deactivate"):
        raise ValidationError("unsupported transfer kind")
    _verify_aligned_tuples("transfer", event.atoms, event.positions, event.durations)
    _verify_event_atoms(event.atoms, known_atoms, "transfer")
    for atom, position, duration in zip(
        event.atoms, event.positions, event.durations, strict=True
    ):
        _verify_position(position, "transfer position")
        _verify_non_negative_number(duration, "transfer duration")
        if position != position_by_atom[atom]:
            raise ValidationError(
                "transfer position does not match the current position"
            )


def _verify_move(
    event: MoveEvent,
    known_atoms: set[RegisterRef],
    position_by_atom: dict[RegisterRef, Position],
) -> None:
    if event.kind not in ("big_move", "park"):
        raise ValidationError("unsupported move kind")
    _verify_aligned_tuples(
        "move", event.atoms, event.starts, event.ends, event.distances, event.durations
    )
    _verify_event_atoms(event.atoms, known_atoms, "move")
    for atom, start, end, distance, duration in zip(
        event.atoms,
        event.starts,
        event.ends,
        event.distances,
        event.durations,
        strict=True,
    ):
        _verify_position(start, "move start")
        _verify_position(end, "move end")
        _verify_non_negative_number(distance, "move distance")
        _verify_non_negative_number(duration, "move duration")
        if start != position_by_atom[atom]:
            raise ValidationError("move start does not match the current position")
    for atom, end in zip(event.atoms, event.ends, strict=True):
        position_by_atom[atom] = end


def _verify_gate_batch(
    event: GateBatch,
    known_atoms: set[RegisterRef],
    position_by_atom: dict[RegisterRef, Position],
    operation_ids: set[str],
) -> None:
    if type(event.stage) is not int or event.stage < 0:
        raise ValidationError("gate batch stage must be a non-negative integer")
    if not isinstance(event.gates, tuple) or not event.gates:
        raise ValidationError("gate batch must contain at least one scheduled gate")
    _verify_non_negative_number(event.duration, "gate batch duration")

    batch_atoms: set[RegisterRef] = set()
    for gate in event.gates:
        if type(gate) is not ScheduledGate:
            raise ValidationError(f"unsupported scheduled gate: {type(gate).__name__}")
        _verify_scheduled_gate(gate, known_atoms, position_by_atom, operation_ids)
        for atom in gate.atoms:
            if atom in batch_atoms:
                raise ValidationError("an atom participates twice in one gate batch")
            batch_atoms.add(atom)


def _verify_scheduled_gate(
    gate: ScheduledGate,
    known_atoms: set[RegisterRef],
    position_by_atom: dict[RegisterRef, Position],
    operation_ids: set[str],
) -> None:
    _verify_operation_metadata(gate, operation_ids)
    if not isinstance(gate.atoms, tuple) or not isinstance(gate.positions, tuple):
        raise ValidationError("scheduled gate atoms and positions must be tuples")
    expected_arity = _GATE_ARITIES.get(type(gate.operation))
    if expected_arity is None:
        raise ValidationError(
            f"unsupported scheduled gate: {type(gate.operation).__name__}"
        )
    if len(gate.atoms) != expected_arity or len(gate.positions) != expected_arity:
        raise ValidationError("invalid scheduled gate operand shape")
    _verify_event_atoms(gate.atoms, known_atoms, "scheduled gate")
    _verify_rotation(gate.operation)

    for atom, position in zip(gate.atoms, gate.positions, strict=True):
        _verify_position(position, "scheduled gate position")
        if position != position_by_atom[atom]:
            raise ValidationError(
                "scheduled gate position does not match the current position"
            )


def _verify_crosstalk(
    event: CrosstalkEvent,
    known_atoms: set[RegisterRef],
    position_by_atom: dict[RegisterRef, Position],
) -> None:
    _verify_aligned_tuples("crosstalk", event.atoms, event.positions, event.durations)
    _verify_event_atoms(event.atoms, known_atoms, "crosstalk")
    for atom, position, duration in zip(
        event.atoms, event.positions, event.durations, strict=True
    ):
        _verify_position(position, "crosstalk position")
        _verify_non_negative_number(duration, "crosstalk duration")
        if position != position_by_atom[atom]:
            raise ValidationError(
                "crosstalk position does not match the current position"
            )


def _verify_aligned_tuples(label: str, *values: tuple[object, ...]) -> None:
    if any(not isinstance(value, tuple) for value in values):
        raise ValidationError(f"{label} fields must be tuples")
    if len({len(value) for value in values}) != 1:
        raise ValidationError(f"{label} fields must have aligned lengths")


def _verify_event_atoms(
    atoms: tuple[RegisterRef, ...], known_atoms: set[RegisterRef], label: str
) -> None:
    if any(type(atom) is not RegisterRef or atom not in known_atoms for atom in atoms):
        raise ValidationError(f"{label} references an undeclared atom")
    if len(set(atoms)) != len(atoms):
        raise ValidationError(f"{label} repeats an atom")


def _verify_operation_metadata(gate: ScheduledGate, operation_ids: set[str]) -> None:
    if not isinstance(gate.operation_id, str) or not gate.operation_id:
        raise ValidationError("scheduled operation ID must be a non-empty string")
    if gate.operation_id in operation_ids:
        raise ValidationError(f"duplicate scheduled operation ID: {gate.operation_id}")
    operation_ids.add(gate.operation_id)
    if (
        not isinstance(gate.origin_ids, tuple)
        or not gate.origin_ids
        or any(not isinstance(origin, str) or not origin for origin in gate.origin_ids)
        or len(set(gate.origin_ids)) != len(gate.origin_ids)
    ):
        raise ValidationError("scheduled gate origin IDs must be non-empty and unique")


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


def _verify_position(position: object, label: str) -> None:
    if not isinstance(position, tuple) or len(position) != 2:
        raise ValidationError(f"{label} must be a two-dimensional tuple")
    for coordinate in position:
        _verify_finite_number(coordinate, label)


def _verify_non_negative_number(value: object, label: str) -> None:
    _verify_finite_number(value, label)
    if value < 0:
        raise ValidationError(f"{label} must be non-negative")


def _verify_finite_number(value: object, label: str) -> None:
    if (
        not isinstance(value, numbers.Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValidationError(f"{label} must be a finite real number")


_GATE_ARITIES: dict[type, int] = {RX: 1, RY: 1, RZ: 1, CZGate: 2}
