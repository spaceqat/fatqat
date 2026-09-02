"""Unified, topology-independent superconducting gate IR."""

from __future__ import annotations

import math
import numbers
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Mapping, TypeAlias

from ...operations.fixed_gates import CZGate, SwapGate
from ...operations.parametric_gates import RX, RZ

# Required here as a member of the closed SC instruction type alias.
from ...operations.reset import ResetGate
from ...registers import ClassicalRegister, QuantumRegister, RegisterRef
from ..errors import ValidationError


@dataclass(frozen=True, slots=True)
class MeasureOp:
    """Stateless SC measurement marker; occurrence operands live on SCNode."""

    name: ClassVar[str] = "Measure"


MEASURE = MeasureOp()

SCGate: TypeAlias = RX | RZ | CZGate | SwapGate
SCInstruction: TypeAlias = SCGate | ResetGate | MeasureOp
NodeId: TypeAlias = int

SC_INSTRUCTION_RULES: Mapping[type, tuple[int, int]] = MappingProxyType(
    {
        RX: (1, 0),
        RZ: (1, 0),
        CZGate: (2, 0),
        SwapGate: (2, 0),
        ResetGate: (1, 0),
        MeasureOp: (1, 1),
    }
)


@dataclass(frozen=True, slots=True)
class SCNode:
    operation_id: str
    origin_ids: tuple[str, ...]
    instruction: SCInstruction
    qubits: tuple[RegisterRef, ...]
    clbits: tuple[RegisterRef, ...] = ()


@dataclass(frozen=True, slots=True)
class SCWire:
    qubit: RegisterRef
    nodes: tuple[NodeId, ...]


@dataclass(frozen=True, slots=True)
class SCProgram:
    IR_ID: ClassVar[str] = "sc.gate.v1"

    qubits: tuple[RegisterRef, ...]
    clbits: tuple[RegisterRef, ...]
    nodes: tuple[SCNode, ...]
    wires: tuple[SCWire, ...]


def verify_sc_program(program: object) -> None:
    """Validate facts fully contained in a standalone SCProgram."""

    if type(program) is not SCProgram:
        raise ValidationError("expected SCProgram")
    _verify_declared_refs(program.qubits, QuantumRegister, "qubit")
    _verify_declared_refs(program.clbits, ClassicalRegister, "clbit")
    if not isinstance(program.nodes, tuple):
        raise ValidationError("SC nodes must be a tuple")
    if not isinstance(program.wires, tuple):
        raise ValidationError("SC wires must be a tuple")

    known_qubits = set(program.qubits)
    known_clbits = set(program.clbits)
    operation_ids: set[str] = set()
    written_clbits: set[RegisterRef] = set()
    measurement_nodes: set[NodeId] = set()

    for node_id, node in enumerate(program.nodes):
        if type(node) is not SCNode:
            raise ValidationError("SC nodes must contain exact SCNode values")
        if not isinstance(node.operation_id, str) or not node.operation_id:
            raise ValidationError("SC operation ID must be a non-empty string")
        if node.operation_id in operation_ids:
            raise ValidationError(f"duplicate SC operation ID: {node.operation_id}")
        operation_ids.add(node.operation_id)
        if (
            not isinstance(node.origin_ids, tuple)
            or not node.origin_ids
            or any(not isinstance(item, str) or not item for item in node.origin_ids)
            or len(set(node.origin_ids)) != len(node.origin_ids)
        ):
            raise ValidationError("SC origin IDs must be non-empty and unique")
        if not isinstance(node.qubits, tuple) or not isinstance(node.clbits, tuple):
            raise ValidationError("SC operands must be tuples")

        rule = SC_INSTRUCTION_RULES.get(type(node.instruction))
        if rule is None:
            raise ValidationError(
                f"unsupported SC instruction: {type(node.instruction).__name__}"
            )
        if (len(node.qubits), len(node.clbits)) != rule:
            raise ValidationError("invalid SC operand shape")
        if any(item not in known_qubits for item in node.qubits):
            raise ValidationError("SC node references an undeclared qubit")
        if any(item not in known_clbits for item in node.clbits):
            raise ValidationError("SC node references an undeclared clbit")
        if len(set(node.qubits)) != len(node.qubits):
            raise ValidationError("SC node repeats a qubit operand")
        _verify_rotation(node.instruction)

        if type(node.instruction) is MeasureOp:
            measurement_nodes.add(node_id)
            clbit = node.clbits[0]
            if clbit in written_clbits:
                raise ValidationError("SC measurement writes a clbit more than once")
            written_clbits.add(clbit)

    wire_by_qubit: dict[RegisterRef, SCWire] = {}
    coverage: list[set[RegisterRef]] = [set() for _ in program.nodes]
    successors: list[set[NodeId]] = [set() for _ in program.nodes]
    indegree = [0] * len(program.nodes)

    for wire in program.wires:
        if type(wire) is not SCWire:
            raise ValidationError("SC wires must contain exact SCWire values")
        if wire.qubit not in known_qubits:
            raise ValidationError("SC wire references an undeclared qubit")
        if wire.qubit in wire_by_qubit:
            raise ValidationError("each qubit must have exactly one SC wire")
        if not isinstance(wire.nodes, tuple):
            raise ValidationError("SC wire nodes must be a tuple")
        if len(set(wire.nodes)) != len(wire.nodes):
            raise ValidationError("SC wire repeats a NodeId")
        wire_by_qubit[wire.qubit] = wire

        for node_id in wire.nodes:
            if (
                not isinstance(node_id, int)
                or isinstance(node_id, bool)
                or node_id < 0
                or node_id >= len(program.nodes)
            ):
                raise ValidationError(f"invalid NodeId on SC wire: {node_id!r}")
            coverage[node_id].add(wire.qubit)
        for source, target in zip(wire.nodes, wire.nodes[1:]):
            if target not in successors[source]:
                successors[source].add(target)
                indegree[target] += 1

    if set(wire_by_qubit) != known_qubits:
        raise ValidationError("each declared qubit must have exactly one SC wire")
    for node_id, node in enumerate(program.nodes):
        if coverage[node_id] != set(node.qubits):
            raise ValidationError(
                f"SC wire coverage does not match operands for NodeId {node_id}"
            )
    for node_id in measurement_nodes:
        qubit = program.nodes[node_id].qubits[0]
        if wire_by_qubit[qubit].nodes[-1] != node_id:
            raise ValidationError("SC measurement must be terminal on its qubit wire")

    ready = deque(index for index, count in enumerate(indegree) if count == 0)
    visited = 0
    while ready:
        source = ready.popleft()
        visited += 1
        for target in successors[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(program.nodes):
        raise ValidationError("SC wire dependencies contain a cycle")


def _verify_declared_refs(
    refs: tuple[RegisterRef, ...],
    expected_register: type[QuantumRegister] | type[ClassicalRegister],
    label: str,
) -> None:
    if not isinstance(refs, tuple):
        raise ValidationError(f"SC {label} refs must be a tuple")
    if any(type(item) is not RegisterRef for item in refs):
        raise ValidationError(f"SC {label} refs must contain exact RegisterRef values")
    if any(not isinstance(item.register, expected_register) for item in refs):
        raise ValidationError(f"SC {label} ref has the wrong kind")
    if len(set(refs)) != len(refs):
        raise ValidationError(f"duplicate declared SC {label} ref")


def _verify_rotation(instruction: SCInstruction) -> None:
    if type(instruction) not in (RX, RZ):
        return
    theta = instruction.theta
    if (
        not isinstance(theta, numbers.Real)
        or isinstance(theta, bool)
        or not math.isfinite(float(theta))
    ):
        raise ValidationError(
            f"{type(instruction).__name__}.theta must be a finite real number"
        )
