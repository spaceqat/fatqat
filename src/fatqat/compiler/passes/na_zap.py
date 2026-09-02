"""Translation pass that schedules NA gates with the internal ZAP algorithm."""

from __future__ import annotations

from collections.abc import Mapping

from ...registers import RegisterRef
from ..algorithms.zap import ZapInteraction, ZapTrace, compile_interactions
from ..core import CompileContext
from ..dialects.na_gate import NAGate, NAMeasure, NAProgram, verify_na_program
from ..dialects.na_zoned import (
    CrosstalkEvent,
    GateBatch,
    MoveEvent,
    Position,
    ScheduledGate,
    TransferEvent,
    ZonedEvent,
    ZonedPlan,
    verify_zoned_plan,
)
from ..errors import ValidationError


def schedule_na_with_zap(
    source: NAProgram,
    architecture: Mapping[str, object],
) -> ZonedPlan:
    """Schedule an NAProgram with internal ZAP and restore atom identities."""
    verify_na_program(source)
    if not isinstance(architecture, Mapping):
        raise ValidationError("ZAP architecture must be a mapping")
    interactions = _project_interactions(source)
    trace = compile_interactions(
        interactions,
        architecture,
        atom_count=len(source.atoms),
    )
    return _translate_trace(source, trace)


def _project_interactions(source: NAProgram) -> tuple[ZapInteraction, ...]:
    index_by_atom = {atom: index for index, atom in enumerate(source.atoms)}
    return tuple(
        ZapInteraction(
            item.operation_id,
            tuple(index_by_atom[atom] for atom in item.atoms),
        )
        for item in source.instructions
        if type(item) is NAGate
    )


def _translate_trace(source: NAProgram, trace: ZapTrace) -> ZonedPlan:
    atom_by_index = source.atoms
    index_by_atom = {atom: index for index, atom in enumerate(atom_by_index)}
    gate_by_id = {
        item.operation_id: item for item in source.instructions if type(item) is NAGate
    }
    atom_count = _trace_field(trace, "atom_count")
    if type(atom_count) is not int or atom_count != len(atom_by_index):
        raise ValidationError("ZAP trace atom_count does not match the NAProgram")
    instructions = _trace_field(trace, "instructions")
    if not isinstance(instructions, tuple):
        raise ValidationError("ZAP trace instructions must be a tuple")

    initial_placement: tuple[tuple[RegisterRef, Position], ...] | None = None
    events: list[ZonedEvent] = []
    executed_ids: set[str] = set()
    for instruction in instructions:
        if not isinstance(instruction, Mapping):
            raise ValidationError("ZAP trace instructions must be mappings")
        instruction_type = _instruction_type(instruction)
        if instruction_type == "Init":
            if initial_placement is not None or events:
                raise ValidationError("ZAP Init must appear exactly once before events")
            initial_placement = _translate_init(instruction, atom_by_index)
        elif instruction_type in ("Activate", "Deactivate"):
            events.append(
                _translate_transfer(
                    instruction,
                    atom_by_index,
                    "activate" if instruction_type == "Activate" else "deactivate",
                )
            )
        elif instruction_type in ("BigMove", "Park"):
            events.append(
                _translate_move(
                    instruction,
                    atom_by_index,
                    "big_move" if instruction_type == "BigMove" else "park",
                )
            )
        elif instruction_type in ("1qGate", "2qGate"):
            events.append(
                _translate_gate_batch(
                    instruction,
                    atom_by_index,
                    index_by_atom,
                    gate_by_id,
                    executed_ids,
                    instruction_type,
                )
            )
        elif instruction_type == "Crosstalk":
            events.append(_translate_crosstalk(instruction, atom_by_index))
        else:
            raise ValidationError(
                f"unsupported ZAP instruction type: {instruction_type}"
            )

    if initial_placement is None:
        raise ValidationError("ZAP trace must begin with Init")
    if executed_ids != set(gate_by_id):
        missing = sorted(set(gate_by_id) - executed_ids)
        raise ValidationError(f"ZAP trace did not execute NA gate IDs: {missing}")

    measurements = tuple(
        item for item in source.instructions if type(item) is NAMeasure
    )
    plan = ZonedPlan(
        atoms=source.atoms,
        clbits=source.clbits,
        initial_placement=initial_placement,
        events=tuple(events),
        terminal_measurements=measurements,
    )
    verify_zoned_plan(plan)
    _verify_provenance(plan, gate_by_id, measurements)
    return plan


def _trace_field(trace: object, name: str) -> object:
    try:
        return getattr(trace, name)
    except AttributeError as exc:
        raise ValidationError(f"ZAP trace is missing {name!r}") from exc


def _instruction_type(instruction: Mapping[str, object]) -> str:
    value = _field(instruction, "type")
    if not isinstance(value, str):
        raise ValidationError("ZAP instruction type must be a string")
    return value


def _translate_init(
    instruction: Mapping[str, object], atom_by_index: tuple[RegisterRef, ...]
) -> tuple[tuple[RegisterRef, Position], ...]:
    locations = _sequence_field(instruction, "locs")
    durations = _sequence_field(instruction, "duration")
    if len(locations) != len(atom_by_index) or len(durations) != len(atom_by_index):
        raise ValidationError("ZAP Init fields must align with atom_count")
    positions = _positions_for_indices(locations, tuple(range(len(atom_by_index))))
    return tuple((atom, positions[index]) for index, atom in enumerate(atom_by_index))


def _translate_transfer(
    instruction: Mapping[str, object],
    atom_by_index: tuple[RegisterRef, ...],
    kind: str,
) -> TransferEvent:
    indices = _indices_field(instruction, "qs", len(atom_by_index))
    durations = _sequence_field(instruction, "duration")
    locations = _sequence_field(instruction, "locs")
    _require_aligned("ZAP transfer", indices, durations, locations)
    positions = _positions_for_indices(locations, indices)
    return TransferEvent(
        kind, _atoms(indices, atom_by_index), positions, tuple(durations)
    )


def _translate_move(
    instruction: Mapping[str, object],
    atom_by_index: tuple[RegisterRef, ...],
    kind: str,
) -> MoveEvent:
    indices = _indices_field(instruction, "qs", len(atom_by_index))
    distances = _sequence_field(instruction, "distance")
    durations = _sequence_field(instruction, "duration")
    locations = _sequence_field(instruction, "locs")
    _require_aligned("ZAP move", indices, distances, durations, locations)
    starts = _move_positions_for_indices(locations, indices, "begin")
    ends = _move_positions_for_indices(locations, indices, "end")
    return MoveEvent(
        kind,
        _atoms(indices, atom_by_index),
        starts,
        ends,
        tuple(distances),
        tuple(durations),
    )


def _translate_gate_batch(
    instruction: Mapping[str, object],
    atom_by_index: tuple[RegisterRef, ...],
    index_by_atom: dict[RegisterRef, int],
    gate_by_id: dict[str, NAGate],
    executed_ids: set[str],
    instruction_type: str,
) -> GateBatch:
    operation_ids = _sequence_field(instruction, "operation_ids")
    operands = _sequence_field(instruction, "gates")
    flattened = _indices_field(instruction, "qs", len(atom_by_index))
    durations = _sequence_field(instruction, "duration")
    locations = _sequence_field(instruction, "locs")
    if not operation_ids or len(operation_ids) != len(operands):
        raise ValidationError("ZAP gate operation_ids must align with gates")

    normalized_operands = tuple(
        _gate_operands(operand, instruction_type, len(atom_by_index))
        for operand in operands
    )
    expected_flattened = tuple(
        atom for operand in normalized_operands for atom in operand
    )
    if flattened != expected_flattened:
        raise ValidationError("ZAP gate qs do not align with reported operands")
    _require_aligned("ZAP gate", flattened, durations, locations)
    positions = _positions_for_indices(locations, flattened)

    gates: list[ScheduledGate] = []
    offset = 0
    for operation_id, operands_for_gate in zip(
        operation_ids, normalized_operands, strict=True
    ):
        if not isinstance(operation_id, str) or not operation_id:
            raise ValidationError("ZAP gate operation IDs must be non-empty strings")
        try:
            source_gate = gate_by_id[operation_id]
        except KeyError as exc:
            raise ValidationError(
                f"ZAP trace names unknown gate ID: {operation_id}"
            ) from exc
        if operation_id in executed_ids:
            raise ValidationError(f"ZAP trace executes gate ID twice: {operation_id}")
        expected_operands = tuple(index_by_atom[atom] for atom in source_gate.atoms)
        if operands_for_gate != expected_operands:
            raise ValidationError(
                f"ZAP gate operands do not match NA gate ID: {operation_id}"
            )
        arity = len(operands_for_gate)
        gate_positions = positions[offset : offset + arity]
        offset += arity
        gates.append(
            ScheduledGate(
                source_gate.operation_id,
                source_gate.origin_ids,
                source_gate.operation,
                source_gate.atoms,
                gate_positions,
            )
        )
        executed_ids.add(operation_id)

    stage = _field(instruction, "stage")
    return GateBatch(stage, tuple(gates), max(durations))


def _translate_crosstalk(
    instruction: Mapping[str, object], atom_by_index: tuple[RegisterRef, ...]
) -> CrosstalkEvent:
    indices = _indices_field(instruction, "qs", len(atom_by_index))
    durations = _sequence_field(instruction, "duration")
    locations = _sequence_field(instruction, "locs")
    _require_aligned("ZAP crosstalk", indices, durations, locations)
    return CrosstalkEvent(
        _atoms(indices, atom_by_index),
        _positions_for_indices(locations, indices),
        tuple(durations),
    )


def _field(instruction: Mapping[str, object], name: str) -> object:
    try:
        return instruction[name]
    except KeyError as exc:
        raise ValidationError(f"ZAP instruction is missing {name!r}") from exc


def _sequence_field(instruction: Mapping[str, object], name: str) -> tuple[object, ...]:
    value = _field(instruction, name)
    if not isinstance(value, (list, tuple)):
        raise ValidationError(f"ZAP instruction field {name!r} must be a sequence")
    return tuple(value)


def _indices_field(
    instruction: Mapping[str, object], name: str, atom_count: int
) -> tuple[int, ...]:
    values = _sequence_field(instruction, name)
    if any(
        type(value) is not int or value < 0 or value >= atom_count for value in values
    ):
        raise ValidationError(f"ZAP instruction field {name!r} has an invalid atom ID")
    if len(set(values)) != len(values):
        raise ValidationError(f"ZAP instruction field {name!r} repeats an atom ID")
    return values  # type: ignore[return-value]


def _gate_operands(
    value: object, instruction_type: str, atom_count: int
) -> tuple[int, ...]:
    if type(value) is int:
        operands = (value,)
    elif isinstance(value, (list, tuple)):
        operands = tuple(value)
    else:
        raise ValidationError("ZAP gate operands must be integers or sequences")
    if (
        instruction_type == "1qGate"
        and len(operands) == 2
        and operands[0] == operands[1]
    ):
        operands = (operands[0],)
    expected_arity = 1 if instruction_type == "1qGate" else 2
    if len(operands) != expected_arity:
        raise ValidationError(
            "ZAP gate operand arity does not match its instruction type"
        )
    if any(
        type(item) is not int or item < 0 or item >= atom_count for item in operands
    ):
        raise ValidationError("ZAP gate operands contain an invalid atom ID")
    if len(set(operands)) != len(operands):
        raise ValidationError("ZAP two-qubit gate repeats an atom ID")
    return operands  # type: ignore[return-value]


def _positions_for_indices(
    locations: tuple[object, ...], indices: tuple[int, ...]
) -> tuple[Position, ...]:
    positions: list[Position] = []
    for location, index in zip(locations, indices, strict=True):
        mapping = _location(location)
        if _location_id(mapping) != index:
            raise ValidationError("ZAP location IDs do not align with atom IDs")
        positions.append((_location_field(mapping, "x"), _location_field(mapping, "y")))
    return tuple(positions)


def _move_positions_for_indices(
    locations: tuple[object, ...], indices: tuple[int, ...], endpoint: str
) -> tuple[Position, ...]:
    positions: list[Position] = []
    for location, index in zip(locations, indices, strict=True):
        mapping = _location(location)
        if _location_id(mapping) != index:
            raise ValidationError("ZAP location IDs do not align with atom IDs")
        positions.append(
            (
                _location_field(mapping, f"x_{endpoint}"),
                _location_field(mapping, f"y_{endpoint}"),
            )
        )
    return tuple(positions)


def _location(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationError("ZAP locations must be mappings")
    if "id" not in value:
        raise ValidationError("ZAP location is missing 'id'")
    return value


def _location_field(location: Mapping[str, object], name: str) -> object:
    try:
        return location[name]
    except KeyError as exc:
        raise ValidationError(f"ZAP location is missing {name!r}") from exc


def _location_id(location: Mapping[str, object]) -> int:
    value = _location_field(location, "id")
    if type(value) is not int:
        raise ValidationError("ZAP location ID must be an integer")
    return value


def _atoms(
    indices: tuple[int, ...], atom_by_index: tuple[RegisterRef, ...]
) -> tuple[RegisterRef, ...]:
    return tuple(atom_by_index[index] for index in indices)


def _require_aligned(label: str, *values: tuple[object, ...]) -> None:
    if len({len(value) for value in values}) != 1:
        raise ValidationError(f"{label} fields must have aligned lengths")


def _verify_provenance(
    plan: ZonedPlan,
    gate_by_id: dict[str, NAGate],
    measurements: tuple[NAMeasure, ...],
) -> None:
    scheduled_ids = {
        gate.operation_id
        for event in plan.events
        if type(event) is GateBatch
        for gate in event.gates
    }
    if scheduled_ids != set(gate_by_id):
        raise ValidationError("ZAP plan gate IDs do not match the source NAProgram")
    if plan.terminal_measurements != measurements:
        raise ValidationError("ZAP plan terminal measurements do not match the source")


class ScheduleNaWithZapPass:
    name = "schedule-with-zap"
    source_type = NAProgram
    target_type = ZonedPlan

    def run(self, source: NAProgram, context: CompileContext) -> ZonedPlan:
        return schedule_na_with_zap(source, context.target)


schedule_with_zap = ScheduleNaWithZapPass()
