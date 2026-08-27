"""Private dynamic-pairing graph for the atom-array simulator."""

from __future__ import annotations

from ..registers import RegisterRef


def _make_edge(a: RegisterRef, b: RegisterRef) -> frozenset[RegisterRef]:
    """Validate and normalize one unordered pair of distinct atom refs."""
    for atom in (a, b):
        if not isinstance(atom, RegisterRef):
            raise TypeError(f"atom must be a RegisterRef, got {type(atom)!r}")
    if a == b:
        raise ValueError("an atom cannot be paired with itself (no self-loops)")
    return frozenset((a, b))


class _AtomConnectivity:
    """Immutable set of atom pairs used while lowering one program."""

    def __init__(self) -> None:
        self._edges: frozenset[frozenset[RegisterRef]] = frozenset()

    @classmethod
    def _from_edges(cls, edges: frozenset[frozenset[RegisterRef]]) -> _AtomConnectivity:
        graph = cls.__new__(cls)
        graph._edges = edges
        return graph

    def pair(self, a: RegisterRef, b: RegisterRef) -> _AtomConnectivity:
        """Return a graph with exactly the requested pair added."""
        edge = _make_edge(a, b)
        if edge in self._edges:
            return self
        return type(self)._from_edges(self._edges | {edge})

    def unpair(self, a: RegisterRef, b: RegisterRef) -> _AtomConnectivity:
        """Return a graph with exactly the requested pair removed."""
        edge = _make_edge(a, b)
        if edge not in self._edges:
            return self
        return type(self)._from_edges(self._edges - {edge})

    def are_paired(self, a: RegisterRef, b: RegisterRef) -> bool:
        """Return whether two distinct atom refs are currently paired."""
        return _make_edge(a, b) in self._edges
