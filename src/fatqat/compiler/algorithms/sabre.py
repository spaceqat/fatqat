"""SABRE initial mapping, SWAP routing, and routed-event contracts."""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import TypeAlias

from ...registers import QuantumRegister, RegisterRef
from ..dialects.sc_gate import NodeId, SCProgram
from .sc_dag import DAGCursor, DAGIndex, build_dag_index

SiteId: TypeAlias = int | str


@dataclass(frozen=True, slots=True)
class ExecuteNode:
    """Execute a semantic SC node on its currently mapped physical sites."""

    node_id: NodeId
    sites: tuple[SiteId, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.node_id, int)
            or isinstance(self.node_id, bool)
            or self.node_id < 0
        ):
            raise ValueError("ExecuteNode node_id must be a non-negative integer")
        _verify_sites(self.sites, allow_empty=False)
        if len(set(self.sites)) != len(self.sites):
            raise ValueError("ExecuteNode sites must be distinct")


@dataclass(frozen=True, slots=True)
class RouteSwap:
    """Compiler-inserted physical SWAP that updates the logical layout."""

    swap_id: str
    sites: tuple[SiteId, SiteId]

    def __post_init__(self) -> None:
        if not isinstance(self.swap_id, str) or not self.swap_id:
            raise ValueError("RouteSwap swap_id must be a non-empty string")
        _verify_sites(self.sites, allow_empty=False)
        if len(self.sites) != 2 or self.sites[0] == self.sites[1]:
            raise ValueError("RouteSwap requires two distinct physical sites")


RoutedEvent: TypeAlias = ExecuteNode | RouteSwap
LayoutSnapshot: TypeAlias = tuple[tuple[RegisterRef, SiteId], ...]


@dataclass(frozen=True, slots=True)
class SabreResult:
    """Ordered route events and immutable layout snapshots produced by SABRE."""

    events: tuple[RoutedEvent, ...]
    initial_layout: LayoutSnapshot
    final_layout: LayoutSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple) or any(
            type(item) not in (ExecuteNode, RouteSwap) for item in self.events
        ):
            raise TypeError(
                "SabreResult events must be ExecuteNode or RouteSwap values"
            )
        initial_refs = _verify_layout(self.initial_layout, "initial")
        final_refs = _verify_layout(self.final_layout, "final")
        if initial_refs != final_refs:
            raise ValueError("initial and final layouts must contain the same refs")


def sabre_map(
    program: SCProgram,
    sites: tuple[SiteId, ...],
    couplings: frozenset[tuple[SiteId, SiteId]],
    *,
    seed: int = 0,
) -> SabreResult:
    """Choose an initial layout and route one SC wire-DAG onto a coupling graph."""

    _verify_sites(sites, allow_empty=False)
    if len(set(sites)) != len(sites):
        raise ValueError("physical sites must be distinct")
    if len(program.qubits) > len(sites):
        raise ValueError("not enough physical sites for SC program qubits")
    if type(seed) is not int:
        raise TypeError("SABRE seed must be an integer")

    adjacency, edges = _build_topology(sites, couplings)
    distances = _all_pairs_distances(sites, adjacency)
    index = build_dag_index(program)
    rng = random.Random(seed)
    shuffled_sites = list(sites)
    rng.shuffle(shuffled_sites)
    initial = dict(zip(program.qubits, shuffled_sites))

    for _ in range(3):
        _, forward = _route(
            program,
            index,
            initial,
            sites,
            adjacency,
            edges,
            distances,
            rng,
            emit_events=False,
        )
        _, initial = _route(
            program,
            index.reversed(),
            forward,
            sites,
            adjacency,
            edges,
            distances,
            rng,
            emit_events=False,
        )

    events, final = _route(
        program,
        index,
        initial,
        sites,
        adjacency,
        edges,
        distances,
        rng,
        emit_events=True,
    )
    return SabreResult(
        events=events,
        initial_layout=_layout_snapshot(program, initial),
        final_layout=_layout_snapshot(program, final),
    )


def _build_topology(
    sites: tuple[SiteId, ...],
    couplings: frozenset[tuple[SiteId, SiteId]],
) -> tuple[dict[SiteId, set[SiteId]], tuple[tuple[SiteId, SiteId], ...]]:
    site_set = set(sites)
    site_rank = {site: index for index, site in enumerate(sites)}
    adjacency = {site: set() for site in sites}
    undirected: set[tuple[SiteId, SiteId]] = set()
    for edge in couplings:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise TypeError("couplings must contain two-site tuples")
        first, second = edge
        if first not in site_set or second not in site_set:
            raise ValueError("coupling endpoint is outside physical sites")
        if first == second:
            raise ValueError("coupling endpoints must be distinct")
        if site_rank[first] < site_rank[second]:
            canonical = (first, second)
        else:
            canonical = (second, first)
        undirected.add(canonical)
        adjacency[first].add(second)
        adjacency[second].add(first)
    edges = tuple(
        sorted(
            undirected,
            key=lambda edge: (site_rank[edge[0]], site_rank[edge[1]]),
        )
    )
    return adjacency, edges


def _all_pairs_distances(
    sites: tuple[SiteId, ...], adjacency: dict[SiteId, set[SiteId]]
) -> dict[SiteId, dict[SiteId, float]]:
    distances: dict[SiteId, dict[SiteId, float]] = {}
    for source in sites:
        source_distances = {site: math.inf for site in sites}
        source_distances[source] = 0.0
        queue = deque((source,))
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if math.isinf(source_distances[neighbor]):
                    source_distances[neighbor] = source_distances[current] + 1.0
                    queue.append(neighbor)
        distances[source] = source_distances
    return distances


def _route(
    program: SCProgram,
    index: DAGIndex,
    initial: dict[RegisterRef, SiteId],
    sites: tuple[SiteId, ...],
    adjacency: dict[SiteId, set[SiteId]],
    edges: tuple[tuple[SiteId, SiteId], ...],
    distances: dict[SiteId, dict[SiteId, float]],
    rng: random.Random,
    *,
    emit_events: bool,
) -> tuple[tuple[RoutedEvent, ...], dict[RegisterRef, SiteId]]:
    cursor = DAGCursor(index)
    logical_to_site = dict(initial)
    site_to_logical: dict[SiteId, RegisterRef | None] = {site: None for site in sites}
    for logical, site in logical_to_site.items():
        site_to_logical[site] = logical

    events: list[RoutedEvent] = []
    decay = {site: 1.0 for site in sites}
    swap_count = 0
    max_swaps = max(1, len(program.nodes) * max(1, len(sites)) * 10)

    while not cursor.complete:
        node_id = _next_executable(program, cursor, logical_to_site, adjacency)
        if node_id is not None:
            node_sites = tuple(
                logical_to_site[qubit] for qubit in program.nodes[node_id].qubits
            )
            if emit_events:
                events.append(ExecuteNode(node_id, node_sites))
            cursor.consume(node_id)
            continue

        front = tuple(
            node_id
            for node_id in cursor.ready
            if len(program.nodes[node_id].qubits) == 2
        )
        if not front:
            raise ValueError("SABRE front layer cannot make progress")

        front_sites = {
            logical_to_site[qubit]
            for node_id in front
            for qubit in program.nodes[node_id].qubits
        }
        candidates = tuple(
            edge for edge in edges if edge[0] in front_sites or edge[1] in front_sites
        )
        if not candidates:
            raise ValueError("SC interaction is unreachable in coupling graph")

        extended = _extended_set(program, index, front)
        scored = [
            (
                _swap_score(
                    candidate,
                    program,
                    front,
                    extended,
                    logical_to_site,
                    site_to_logical,
                    distances,
                    decay,
                ),
                candidate,
            )
            for candidate in candidates
        ]
        best_score = min(score for score, _ in scored)
        if math.isinf(best_score):
            raise ValueError("SC interaction is unreachable in coupling graph")
        best = [
            candidate
            for score, candidate in scored
            if math.isclose(score, best_score, rel_tol=0.0, abs_tol=1e-12)
        ]
        selected = rng.choice(best)
        _apply_swap(selected, logical_to_site, site_to_logical)
        if emit_events:
            events.append(RouteSwap(f"route.swap.{swap_count}", selected))
        decay[selected[0]] += 0.001
        decay[selected[1]] += 0.001
        swap_count += 1
        if swap_count > max_swaps:
            raise ValueError("SABRE routing did not converge")

    return tuple(events), logical_to_site


def _next_executable(
    program: SCProgram,
    cursor: DAGCursor,
    layout: dict[RegisterRef, SiteId],
    adjacency: dict[SiteId, set[SiteId]],
) -> NodeId | None:
    for node_id in cursor.ready:
        qubits = program.nodes[node_id].qubits
        if len(qubits) < 2:
            return node_id
        first, second = (layout[qubit] for qubit in qubits)
        if second in adjacency[first]:
            return node_id
    return None


def _extended_set(
    program: SCProgram,
    index: DAGIndex,
    front: tuple[NodeId, ...],
    limit: int = 20,
) -> tuple[NodeId, ...]:
    queue = deque(
        successor for node_id in front for successor in index.successors[node_id]
    )
    seen = set(front)
    extended: list[NodeId] = []
    while queue and len(extended) < limit:
        node_id = queue.popleft()
        if node_id in seen:
            continue
        seen.add(node_id)
        if len(program.nodes[node_id].qubits) == 2:
            extended.append(node_id)
        queue.extend(index.successors[node_id])
    return tuple(extended)


def _swap_score(
    candidate: tuple[SiteId, SiteId],
    program: SCProgram,
    front: tuple[NodeId, ...],
    extended: tuple[NodeId, ...],
    logical_to_site: dict[RegisterRef, SiteId],
    site_to_logical: dict[SiteId, RegisterRef | None],
    distances: dict[SiteId, dict[SiteId, float]],
    decay: dict[SiteId, float],
) -> float:
    swapped = dict(logical_to_site)
    first, second = candidate
    first_logical = site_to_logical[first]
    second_logical = site_to_logical[second]
    if first_logical is not None:
        swapped[first_logical] = second
    if second_logical is not None:
        swapped[second_logical] = first

    front_cost = _node_distance(program, front, swapped, distances)
    lookahead_cost = _node_distance(program, extended, swapped, distances)
    return (front_cost + 0.5 * lookahead_cost) * max(decay[first], decay[second])


def _node_distance(
    program: SCProgram,
    node_ids: tuple[NodeId, ...],
    layout: dict[RegisterRef, SiteId],
    distances: dict[SiteId, dict[SiteId, float]],
) -> float:
    if not node_ids:
        return 0.0
    total = 0.0
    for node_id in node_ids:
        first, second = program.nodes[node_id].qubits
        total += distances[layout[first]][layout[second]]
    return total / len(node_ids)


def _apply_swap(
    edge: tuple[SiteId, SiteId],
    logical_to_site: dict[RegisterRef, SiteId],
    site_to_logical: dict[SiteId, RegisterRef | None],
) -> None:
    first, second = edge
    first_logical = site_to_logical[first]
    second_logical = site_to_logical[second]
    site_to_logical[first], site_to_logical[second] = second_logical, first_logical
    if first_logical is not None:
        logical_to_site[first_logical] = second
    if second_logical is not None:
        logical_to_site[second_logical] = first


def _layout_snapshot(
    program: SCProgram, layout: dict[RegisterRef, SiteId]
) -> LayoutSnapshot:
    return tuple((qubit, layout[qubit]) for qubit in program.qubits)


def _verify_layout(layout: LayoutSnapshot, label: str) -> set[RegisterRef]:
    if not isinstance(layout, tuple):
        raise TypeError(f"{label} layout must be a tuple")
    refs: set[RegisterRef] = set()
    sites: set[SiteId] = set()
    for entry in layout:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(
                f"{label} layout entries must be (RegisterRef, SiteId) tuples"
            )
        ref, site = entry
        if type(ref) is not RegisterRef or not isinstance(
            ref.register, QuantumRegister
        ):
            raise TypeError(f"{label} layout refs must be qubits")
        _verify_site(site)
        if ref in refs:
            raise ValueError(f"ref appears more than once in {label} layout")
        if site in sites:
            raise ValueError(f"site appears more than once in {label} layout")
        refs.add(ref)
        sites.add(site)
    return refs


def _verify_sites(sites: tuple[SiteId, ...], *, allow_empty: bool) -> None:
    if not isinstance(sites, tuple):
        raise TypeError("physical sites must be a tuple")
    if not allow_empty and not sites:
        raise ValueError("physical sites must not be empty")
    for site in sites:
        _verify_site(site)


def _verify_site(site: SiteId) -> None:
    if isinstance(site, bool) or not isinstance(site, (int, str)):
        raise TypeError("physical site ID must be an integer or string")
    if isinstance(site, str) and not site:
        raise ValueError("physical site string must not be empty")
