"""Derived dependency indexes and per-search state for SC wire-DAGs."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from ..dialects.sc_gate import NodeId, SCProgram


@dataclass(frozen=True, slots=True)
class DAGIndex:
    """Immutable predecessor/successor projection derived from an SCProgram."""

    predecessors: tuple[tuple[NodeId, ...], ...]
    successors: tuple[tuple[NodeId, ...], ...]

    def reversed(self) -> "DAGIndex":
        return DAGIndex(self.successors, self.predecessors)


def build_dag_index(program: SCProgram) -> DAGIndex:
    predecessors = [set() for _ in program.nodes]
    successors = [set() for _ in program.nodes]
    for wire in program.wires:
        for source, target in zip(wire.nodes, wire.nodes[1:]):
            successors[source].add(target)
            predecessors[target].add(source)
    return DAGIndex(
        predecessors=tuple(tuple(sorted(items)) for items in predecessors),
        successors=tuple(tuple(sorted(items)) for items in successors),
    )


class DAGCursor:
    """Mutable ready/remaining-indegree state for one DAG traversal."""

    def __init__(self, index: DAGIndex) -> None:
        if len(index.predecessors) != len(index.successors):
            raise ValueError("DAG index predecessor/successor sizes differ")
        self._index = index
        self._remaining = [len(items) for items in index.predecessors]
        self._ready = {
            node_id for node_id, count in enumerate(self._remaining) if count == 0
        }
        self._consumed = 0

    @property
    def ready(self) -> tuple[NodeId, ...]:
        return tuple(sorted(self._ready))

    @property
    def complete(self) -> bool:
        return self._consumed == len(self._remaining)

    def consume(self, node_id: NodeId) -> None:
        if node_id not in self._ready:
            raise ValueError(f"NodeId {node_id} is not ready")
        self._ready.remove(node_id)
        self._remaining[node_id] = -1
        self._consumed += 1
        for successor in self._index.successors[node_id]:
            self._remaining[successor] -= 1
            if self._remaining[successor] == 0:
                self._ready.add(successor)


def topological_order(
    program: SCProgram, index: DAGIndex | None = None
) -> tuple[NodeId, ...]:
    """Return a stable topological order without adding order to the IR itself."""

    dag = build_dag_index(program) if index is None else index
    remaining = [len(items) for items in dag.predecessors]
    qubit_index = {qubit: position for position, qubit in enumerate(program.qubits)}

    def key(node_id: NodeId):
        node = program.nodes[node_id]
        return (
            tuple(qubit_index[item] for item in node.qubits),
            node.operation_id,
            node_id,
        )

    ready = [
        (key(node_id), node_id) for node_id, count in enumerate(remaining) if count == 0
    ]
    heapq.heapify(ready)
    ordered: list[NodeId] = []
    while ready:
        _, source = heapq.heappop(ready)
        ordered.append(source)
        for target in dag.successors[source]:
            remaining[target] -= 1
            if remaining[target] == 0:
                heapq.heappush(ready, (key(target), target))
    if len(ordered) != len(program.nodes):
        raise ValueError("SC dependency graph contains a cycle")
    return tuple(ordered)
