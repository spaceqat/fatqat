"""Private data-only view models used by visualization renderers."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from ..registers import RegisterRef


@dataclass(frozen=True)
class _CountsView:
    """Prepared counts data independent of Matplotlib."""

    labels: tuple[str, ...]
    values: tuple[int | float, ...]
    total: int
    stat: str
    has_other: bool = False


@dataclass(frozen=True, slots=True)
class _InstructionNode:
    """One immutable source-program instruction in the dependency DAG.

    ``node_id`` is the original source position of the logical instruction in ``Program._instructions``. Keeping that identity stable makes diagnostics and future renderers reproducible even when backend-specific instructions are omitted or several logical instructions share a later execution step.
    """

    node_id: int
    program_index: int
    operation_name: str
    targets: tuple[RegisterRef, ...]
    condition: tuple[tuple[RegisterRef, int], ...] | None
    node_type: str
    outputs: tuple[RegisterRef, ...] = ()
    instruction: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(
            self,
            "condition",
            None if self.condition is None else tuple(self.condition),
        )
        object.__setattr__(self, "outputs", tuple(self.outputs))

    # Short aliases keep the view model convenient for renderers without
    # duplicating state or weakening its immutable contract.
    @property
    def id(self) -> int:
        return self.node_id

    @property
    def index(self) -> int:
        return self.program_index

    @property
    def name(self) -> str:
        return self.operation_name

    @property
    def kind(self) -> str:
        return self.node_type


@dataclass(frozen=True, slots=True)
class _DependencyEdge:
    """One directed dependency between two instruction node IDs."""

    source: int
    target: int
    reason: str

    @property
    def source_id(self) -> int:
        return self.source

    @property
    def target_id(self) -> int:
        return self.target

    @property
    def from_node(self) -> int:
        return self.source

    @property
    def to_node(self) -> int:
        return self.target


@dataclass(frozen=True, slots=True)
class _InstructionDAG:
    """Immutable, hardware-independent logical instruction dependency graph."""

    nodes: tuple[_InstructionNode, ...]
    edges: tuple[_DependencyEdge, ...]
    position_to_node: Mapping[int, _InstructionNode]

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(
            self,
            "position_to_node",
            MappingProxyType(dict(self.position_to_node)),
        )

    @property
    def node_ids(self) -> tuple[int, ...]:
        """Return node IDs in source-program order."""
        return tuple(node.node_id for node in self.nodes)

    @property
    def node_by_id(self) -> Mapping[int, _InstructionNode]:
        """Return a read-only lookup by stable source node ID."""
        return MappingProxyType({node.node_id: node for node in self.nodes})

    @property
    def by_program_index(self) -> Mapping[int, _InstructionNode]:
        """Alias for the source-position lookup."""
        return self.position_to_node

    def predecessors(self, node: int | _InstructionNode) -> tuple[int, ...]:
        """Return direct predecessor IDs in deterministic edge order."""
        node_id = node.node_id if isinstance(node, _InstructionNode) else node
        return tuple(edge.source for edge in self.edges if edge.target == node_id)

    def successors(self, node: int | _InstructionNode) -> tuple[int, ...]:
        """Return direct successor IDs in deterministic edge order."""
        node_id = node.node_id if isinstance(node, _InstructionNode) else node
        return tuple(edge.target for edge in self.edges if edge.source == node_id)


@dataclass(frozen=True, slots=True)
class _InteractionFrequencyEdge:
    """One undirected logical-qubit edge weighted by static interaction count."""

    source: RegisterRef
    target: RegisterRef
    count: int

    @property
    def weight(self) -> int:
        """Alias used by graph renderers for the edge width/weight."""
        return self.count

    @property
    def frequency(self) -> int:
        """Alias for the number of logical interaction occurrences."""
        return self.count


@dataclass(frozen=True, slots=True)
class _InteractionFrequencyGraph:
    """Immutable, hardware-independent logical-qubit interaction graph."""

    nodes: tuple[RegisterRef, ...]
    edges: tuple[_InteractionFrequencyEdge, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))

    def edge_between(
        self,
        first: RegisterRef,
        second: RegisterRef,
    ) -> _InteractionFrequencyEdge | None:
        """Return the edge for two logical refs, independent of operand order."""
        for edge in self.edges:
            if {edge.source, edge.target} == {first, second}:
                return edge
        return None

    def draw(self, renderer: str = "matplotlib", **kwargs: Any):
        """Render this interaction graph with the selected renderer."""
        if renderer != "matplotlib":
            raise ValueError(
                "interaction frequency drawing only supports the matplotlib renderer"
            )
        from ._render_mpl import _render_interaction_frequency

        return _render_interaction_frequency(self, **kwargs)
