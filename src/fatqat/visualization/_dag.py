"""Private instruction dependency DAG extraction for visualization."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from ..operations import BarrierGate, Measurement, PairGate, PutGate, UnpairGate
from ..registers import RegisterRef, RegisterView, _view_members
from ._viewmodels import (
    _DependencyEdge,
    _InstructionDAG,
    _InstructionNode,
    _InteractionFrequencyEdge,
    _InteractionFrequencyGraph,
)

if TYPE_CHECKING:
    from ..program import Program


def _ref_key(ref: RegisterRef) -> tuple[int, int]:
    """Return an identity key that does not depend on register names."""
    return id(ref.register), ref.index


def _is_hardware_instruction(step: Any) -> bool:
    """Return whether an instruction belongs to a backend-specific timeline."""
    if isinstance(step, Measurement):
        return False
    operation = step.operation
    return isinstance(operation, (PutGate, PairGate, UnpairGate)) or bool(
        getattr(operation, "_is_direct_control", False)
    )


def _expanded_targets(targets: Iterable[Any]) -> tuple[RegisterRef, ...]:
    """Flatten grouped targets into the scalar refs they touch."""
    values = tuple(targets)
    if not any(isinstance(target, RegisterView) for target in values):
        return tuple(values)

    groups = tuple(
        _view_members(target) if isinstance(target, RegisterView) else (target,)
        for target in values
    )
    if len(groups) == 1:
        occurrences = ((member,) for member in groups[0])
    else:
        occurrences = zip(*groups, strict=True)

    flattened: list[RegisterRef] = []
    seen: set[tuple[int, int]] = set()
    for occurrence in occurrences:
        for ref in occurrence:
            key = _ref_key(ref)
            if key not in seen:
                flattened.append(ref)
                seen.add(key)
    return tuple(flattened)


def _node_info(
    step: Any,
) -> tuple[
    str,
    tuple[RegisterRef, ...],
    tuple[tuple[RegisterRef, int], ...] | None,
    str,
    tuple[RegisterRef, ...],
]:
    """Extract normalized display and dependency facts from one instruction."""
    if isinstance(step, Measurement):
        return (
            "Measurement",
            tuple(step.targets),
            None,
            "measurement",
            tuple(step.outputs),
        )

    operation = step.operation
    targets = _expanded_targets(step.targets)
    is_pair = isinstance(operation, PairGate)
    is_unpair = isinstance(operation, UnpairGate)
    is_barrier = isinstance(operation, BarrierGate)
    if is_barrier:
        node_type = "barrier"
    elif isinstance(operation, PutGate):
        node_type = "put"
    elif is_pair:
        node_type = "pair"
    elif is_unpair:
        node_type = "unpair"
    else:
        node_type = "operation"
    return (
        operation.name,
        targets,
        step.condition,
        node_type,
        (),
    )


def _build_instruction_dag(program: Program) -> _InstructionDAG:
    """Build the private dependency DAG for a program.

    The builder deliberately consumes ``Program._instructions`` rather than adding a public extraction method. This is a circuit-level graph: it knows about logical quantum resources, classical predicates, and barriers, but it does not inspect a device topology or decide whether a two-qubit gate is physically legal. Hardware-specific connectivity is evaluated by the backend-specific evolution timeline used for later animation views.

    Every dependency points from an earlier source position to a later one, so a malformed or cyclic graph cannot be introduced by this extraction strategy. Multiple reasons between the same pair of nodes are retained because they are useful to future renderers and diagnostics.
    """
    nodes: list[_InstructionNode] = []
    edges: list[_DependencyEdge] = []
    edge_keys: set[tuple[int, int, str]] = set()

    last_quantum: dict[tuple[int, int], int] = {}
    last_classical_write: dict[tuple[int, int], int] = {}
    last_barrier: dict[tuple[int, int], int] = {}

    def add_edge(source: int, target: int, reason: str) -> None:
        if source == target:
            return
        key = (source, target, reason)
        if key not in edge_keys:
            edges.append(_DependencyEdge(source, target, reason))
            edge_keys.add(key)

    for program_index, step in enumerate(program._instructions):
        if _is_hardware_instruction(step):
            # Hardware directives remain in Program._instructions for the
            # backend-specific connectivity/evolution pass. They are not
            # logical circuit nodes and must not affect the generic DAG.
            continue
        (
            operation_name,
            targets,
            condition,
            node_type,
            outputs,
        ) = _node_info(step)
        node_id = program_index

        target_keys = tuple(_ref_key(ref) for ref in targets)
        for key in target_keys:
            previous = last_quantum.get(key)
            if previous is not None:
                add_edge(previous, node_id, "quantum")
            previous_barrier = last_barrier.get(key)
            if previous_barrier is not None:
                add_edge(previous_barrier, node_id, "barrier")

        if condition:
            for ref, _ in condition:
                previous = last_classical_write.get(_ref_key(ref))
                if previous is not None:
                    add_edge(previous, node_id, "classical")

        node = _InstructionNode(
            node_id=node_id,
            program_index=program_index,
            operation_name=operation_name,
            targets=targets,
            condition=condition,
            node_type=node_type,
            outputs=outputs,
            instruction=step,
        )
        nodes.append(node)

        for key in target_keys:
            previous_quantum = last_quantum.get(key)
            if node_type == "barrier" and previous_quantum is not None:
                add_edge(previous_quantum, node_id, "barrier")
            last_quantum[key] = node_id
            if node_type == "barrier":
                last_barrier[key] = node_id

        for ref in outputs:
            key = _ref_key(ref)
            previous = last_classical_write.get(key)
            if previous is not None:
                add_edge(previous, node_id, "classical")
            last_classical_write[key] = node_id

    return _InstructionDAG(
        nodes=tuple(nodes),
        edges=tuple(edges),
        position_to_node={node.program_index: node for node in nodes},
    )


def _interaction_pairs(step: Any) -> tuple[tuple[RegisterRef, RegisterRef], ...]:
    """Return every logical pair occurrence in one exact two-target operation."""
    operation = getattr(step, "operation", None)
    if operation is None or operation.num_subsystems != 2:
        return ()
    raw_targets = tuple(step.targets)
    if not any(isinstance(target, RegisterView) for target in raw_targets):
        return ((raw_targets[0], raw_targets[1]),)

    groups = tuple(
        _view_members(target) if isinstance(target, RegisterView) else (target,)
        for target in raw_targets
    )
    if len(groups) != 2:
        return ()
    return tuple(zip(*groups, strict=True))


def _build_interaction_frequency_graph(program: Program) -> _InteractionFrequencyGraph:
    """Build a static logical-qubit interaction frequency graph.

    The graph counts source-level two-target logical operations, including conditionally executed operations as potential interactions. It excludes backend directives such as Put, Pair, Unpair, and direct pulse controls; those are not logical circuit interactions.
    """
    dag = _build_instruction_dag(program)
    nodes = tuple(
        ref
        for register in program.quantum_registers
        for ref in (register[index] for index in range(register.size))
    )
    node_keys = {_ref_key(ref): ref for ref in nodes}
    node_order = {key: index for index, key in enumerate(node_keys)}
    counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}

    for node in dag.nodes:
        for first, second in _interaction_pairs(node.instruction):
            first_key = _ref_key(first)
            second_key = _ref_key(second)
            if first_key not in node_keys or second_key not in node_keys:
                raise ValueError(
                    "interaction references a quantum resource outside the program"
                )
            key = tuple(sorted((first_key, second_key), key=node_order.__getitem__))
            counts[key] = counts.get(key, 0) + 1

    edges = tuple(
        _InteractionFrequencyEdge(
            source=node_keys[first_key],
            target=node_keys[second_key],
            count=count,
        )
        for (first_key, second_key), count in sorted(
            counts.items(),
            key=lambda item: (
                node_order[item[0][0]],
                node_order[item[0][1]],
            ),
        )
    )
    return _InteractionFrequencyGraph(nodes=nodes, edges=edges)


__all__ = ["_build_instruction_dag", "_build_interaction_frequency_graph"]
