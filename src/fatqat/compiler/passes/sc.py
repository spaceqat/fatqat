"""Logical gate normalization into the unified SC wire-DAG."""

from __future__ import annotations

import math

from ... import operations as ops
from ...operations.fixed_gates import CZGate, SwapGate
from ...operations.parametric_gates import Phase, RX, RY, RZ
from ...operations.reset import ResetGate
from ...registers import RegisterRef
from ..core import CompileContext
from ..dialects.logical_gate import (
    LogicalGate,
    LogicalMeasure,
    LogicalProgram,
)
from ..dialects.sc_gate import (
    MEASURE,
    SCInstruction,
    SCNode,
    SCProgram,
    SCWire,
)
from ..errors import UnsupportedFeatureError, ValidationError

_ANGLE_TOLERANCE = 1e-12


class _SCBuilder:
    """Mutable pass-local builder that freezes and remaps NodeIds exactly once."""

    def __init__(
        self, qubits: tuple[RegisterRef, ...], clbits: tuple[RegisterRef, ...]
    ) -> None:
        self.qubits = qubits
        self.clbits = clbits
        self._nodes: list[SCNode | None] = []
        self._wires: dict[RegisterRef, list[int]] = {item: [] for item in qubits}
        self._merge_count = 0

    def add(
        self,
        instruction: SCInstruction,
        qubits: tuple[RegisterRef, ...],
        origin_ids: tuple[str, ...],
        clbits: tuple[RegisterRef, ...] = (),
    ) -> None:
        if type(instruction) in (RX, RZ):
            normalized = _normalize_angle(instruction.theta)
            if _is_zero(normalized):
                return
            instruction = type(instruction)(normalized)
            if self._merge_rotation(instruction, qubits, origin_ids):
                return

        if type(instruction) is CZGate and self._cancel_adjacent_cz(qubits):
            return

        node_id = len(self._nodes)
        self._nodes.append(
            SCNode(
                operation_id=f"sc.{node_id}",
                origin_ids=origin_ids,
                instruction=instruction,
                qubits=qubits,
                clbits=clbits,
            )
        )
        for qubit in qubits:
            self._wires[qubit].append(node_id)

    def _merge_rotation(
        self,
        instruction: RX | RZ,
        qubits: tuple[RegisterRef, ...],
        origin_ids: tuple[str, ...],
    ) -> bool:
        if len(qubits) != 1:
            return False
        wire = self._wires[qubits[0]]
        if not wire:
            return False
        previous_id = wire[-1]
        previous = self._nodes[previous_id]
        if previous is None or type(previous.instruction) is not type(instruction):
            return False
        if previous.qubits != qubits or previous.clbits:
            return False

        theta = _normalize_angle(previous.instruction.theta + instruction.theta)
        if _is_zero(theta):
            wire.pop()
            self._nodes[previous_id] = None
            return True

        merged_origins = tuple(dict.fromkeys(previous.origin_ids + origin_ids))
        self._nodes[previous_id] = SCNode(
            operation_id=f"sc.merge.{self._merge_count}",
            origin_ids=merged_origins,
            instruction=type(instruction)(theta),
            qubits=qubits,
        )
        self._merge_count += 1
        return True

    def _cancel_adjacent_cz(self, qubits: tuple[RegisterRef, ...]) -> bool:
        if len(qubits) != 2:
            return False
        first_wire = self._wires[qubits[0]]
        second_wire = self._wires[qubits[1]]
        if not first_wire or not second_wire or first_wire[-1] != second_wire[-1]:
            return False
        previous_id = first_wire[-1]
        previous = self._nodes[previous_id]
        if (
            previous is None
            or type(previous.instruction) is not CZGate
            or set(previous.qubits) != set(qubits)
        ):
            return False
        first_wire.pop()
        second_wire.pop()
        self._nodes[previous_id] = None
        return True

    def freeze(self) -> SCProgram:
        old_to_new: dict[int, int] = {}
        nodes: list[SCNode] = []
        for old_id, node in enumerate(self._nodes):
            if node is None:
                continue
            old_to_new[old_id] = len(nodes)
            nodes.append(node)
        wires = tuple(
            SCWire(
                qubit,
                tuple(old_to_new[node_id] for node_id in self._wires[qubit]),
            )
            for qubit in self.qubits
        )
        return SCProgram(self.qubits, self.clbits, tuple(nodes), wires)


def normalize_sc_program(source: LogicalProgram) -> SCProgram:
    """Lower numeric, static logical gates into the closed SC instruction set."""

    builder = _SCBuilder(source.qubits, source.clbits)
    known_origins = {item.operation_id for item in source.instructions}

    for instruction in source.instructions:
        if type(instruction) is LogicalMeasure:
            builder.add(
                MEASURE,
                (instruction.qubit,),
                (instruction.operation_id,),
                (instruction.clbit,),
            )
            continue
        if type(instruction) is not LogicalGate:
            raise UnsupportedFeatureError(
                f"unsupported logical instruction: {type(instruction).__name__}"
            )
        _lower_gate(builder, instruction)

    target = builder.freeze()
    output_origins = {origin for node in target.nodes for origin in node.origin_ids}
    unknown_origins = output_origins - known_origins
    if unknown_origins:
        raise ValidationError(
            f"SC output contains unknown origin IDs: {unknown_origins}"
        )
    return target


def _lower_gate(builder: _SCBuilder, instruction: LogicalGate) -> None:
    operation = instruction.operation
    qubits = instruction.operands
    origins = (instruction.operation_id,)

    if type(operation) in (RX, RZ, CZGate, SwapGate):
        builder.add(operation, qubits, origins)
        return
    if type(operation) is ResetGate:
        for qubit in qubits:
            builder.add(ops.Reset, (qubit,), origins)
        return
    if operation is ops.I:
        return
    if operation is ops.H:
        _add_h(builder, qubits[0], origins)
        return
    if operation is ops.X:
        builder.add(ops.RX(math.pi), qubits, origins)
        return
    if operation is ops.Y:
        builder.add(ops.RX(math.pi), qubits, origins)
        builder.add(ops.RZ(math.pi), qubits, origins)
        return
    if operation is ops.Z:
        builder.add(ops.RZ(math.pi), qubits, origins)
        return
    if operation is ops.S:
        builder.add(ops.RZ(math.pi / 2), qubits, origins)
        return
    if operation is ops.Sdg:
        builder.add(ops.RZ(-math.pi / 2), qubits, origins)
        return
    if operation is ops.T:
        builder.add(ops.RZ(math.pi / 4), qubits, origins)
        return
    if operation is ops.Tdg:
        builder.add(ops.RZ(-math.pi / 4), qubits, origins)
        return
    if operation is ops.SX:
        builder.add(ops.RX(math.pi / 2), qubits, origins)
        return
    if type(operation) is RY:
        builder.add(ops.RZ(-math.pi / 2), qubits, origins)
        builder.add(ops.RX(operation.theta), qubits, origins)
        builder.add(ops.RZ(math.pi / 2), qubits, origins)
        return
    if type(operation) is Phase:
        builder.add(ops.RZ(operation.theta), qubits, origins)
        return
    if operation is ops.CX:
        control, target = qubits
        _add_h(builder, target, origins)
        builder.add(ops.CZ, (control, target), origins)
        _add_h(builder, target, origins)
        return
    raise UnsupportedFeatureError(
        f"operation {operation.name} is not supported by sc.gate.v1 normalization"
    )


def _add_h(builder: _SCBuilder, qubit: RegisterRef, origins: tuple[str, ...]) -> None:
    builder.add(ops.RZ(math.pi / 2), (qubit,), origins)
    builder.add(ops.RX(math.pi / 2), (qubit,), origins)
    builder.add(ops.RZ(math.pi / 2), (qubit,), origins)


def _normalize_angle(theta: float) -> float:
    value = math.remainder(float(theta), 2 * math.pi)
    return 0.0 if _is_zero(value) else value


def _is_zero(theta: float) -> bool:
    return abs(theta) <= _ANGLE_TOLERANCE


class NormalizeScPass:
    name = "normalize-sc"
    source_type = LogicalProgram
    target_type = SCProgram

    def run(self, source: LogicalProgram, context: CompileContext) -> SCProgram:
        del context
        return normalize_sc_program(source)


normalize_sc = NormalizeScPass()
