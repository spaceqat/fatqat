"""AtomConnectivity: backend-neutral record of which atoms are paired.

A standalone, undirected connectivity graph over ``RegisterRef`` atoms. It is
the source of truth for two-qubit-gate legality in the neutral-atom model: a
gate on a pair is *structurally* legal iff that pair is currently paired here
(per-shot presence and loss are decided separately by the engine's occupancy
state).

Edges are independent: ``pair(A, B)`` adds exactly one edge and never touches
any third atom, so a configuration like "A-B and A-C paired but B-C not" (a
"V") is exactly representable. The graph carries no coordinates, no occupancy,
and no atom-existence registry; it depends only on ``registers.RegisterRef``,
so the pulse layer can reuse it directly.

Values are immutable: ``pair``/``unpair`` return a new ``AtomConnectivity``
and leave the original unchanged. Atoms are keyed by ref identity, matching
``ResourceLayout``.
"""

from __future__ import annotations

from collections.abc import Iterable

from .registers import RegisterRef


def _make_edge(a: RegisterRef, b: RegisterRef) -> frozenset[RegisterRef]:
    """Validate an unordered atom pair and normalize it to a canonical edge.

    Raises:
        TypeError: If either endpoint is not a ``RegisterRef``.
        ValueError: If both endpoints are the same atom (no self-loops).
    """
    for atom in (a, b):
        if not isinstance(atom, RegisterRef):
            raise TypeError(f"atom must be a RegisterRef, got {type(atom)!r}")
    if a == b:
        raise ValueError("an atom cannot be paired with itself (no self-loops)")
    return frozenset((a, b))


class AtomConnectivity:
    """Immutable undirected graph of paired atoms, keyed by ref identity.

    Examples:
        The flagship "V" case -- A paired with both B and C, while B and C stay
        unpaired -- is representable, because each ``pair`` touches one edge:

        >>> import fatqat as fq
        >>> from fatqat.connectivity import AtomConnectivity
        >>> atoms = fq.QuantumRegister(3, name="atoms")
        >>> A, B, C = atoms[0], atoms[1], atoms[2]
        >>> conn = AtomConnectivity().pair(A, B).pair(A, C)
        >>> conn.are_paired(A, B), conn.are_paired(A, C)
        (True, True)
        >>> conn.are_paired(B, C)
        False
        >>> sorted(r.index for r in conn.neighbors(A))
        [1, 2]

        Mutation returns a new value; the original is untouched:

        >>> conn.pair(B, C) is conn
        False
        >>> conn.are_paired(B, C)
        False
    """

    def __init__(self, edges: Iterable[tuple[RegisterRef, RegisterRef]] = ()) -> None:
        """Create a connectivity graph from an optional iterable of atom pairs.

        Args:
            edges: Iterable of ``(a, b)`` atom pairs to pair initially.
                Duplicates and reversed pairs collapse to one edge.
        """
        self._edges: frozenset[frozenset[RegisterRef]] = frozenset(
            _make_edge(a, b) for a, b in edges
        )

    @classmethod
    def _from_edges(
        cls, edges: frozenset[frozenset[RegisterRef]]
    ) -> "AtomConnectivity":
        """Build directly from already-validated canonical edges (no re-check)."""
        obj = cls.__new__(cls)
        obj._edges = frozenset(edges)
        return obj

    def pair(self, a: RegisterRef, b: RegisterRef) -> "AtomConnectivity":
        """Return a new graph with the ``a``-``b`` edge added (idempotent).

        Raises:
            TypeError: If either endpoint is not a ``RegisterRef``.
            ValueError: If ``a`` and ``b`` are the same atom.
        """
        edge = _make_edge(a, b)
        if edge in self._edges:
            return self
        return AtomConnectivity._from_edges(self._edges | {edge})

    def unpair(self, a: RegisterRef, b: RegisterRef) -> "AtomConnectivity":
        """Return a new graph with the ``a``-``b`` edge removed.

        Removing an edge that is not present is a silent no-op (returns an
        equal graph).

        Raises:
            TypeError: If either endpoint is not a ``RegisterRef``.
            ValueError: If ``a`` and ``b`` are the same atom.
        """
        edge = _make_edge(a, b)
        if edge not in self._edges:
            return self
        return AtomConnectivity._from_edges(self._edges - {edge})

    def are_paired(self, a: RegisterRef, b: RegisterRef) -> bool:
        """Whether ``a`` and ``b`` are currently paired (symmetric).

        Querying an atom against itself is always ``False`` (there are no
        self-loops); it does not raise.

        Raises:
            TypeError: If either endpoint is not a ``RegisterRef``.
        """
        for atom in (a, b):
            if not isinstance(atom, RegisterRef):
                raise TypeError(f"atom must be a RegisterRef, got {type(atom)!r}")
        if a == b:
            return False
        return frozenset((a, b)) in self._edges

    def neighbors(self, ref: RegisterRef) -> frozenset[RegisterRef]:
        """Return every atom paired with ``ref`` (empty if it has no edges).

        Raises:
            TypeError: If ``ref`` is not a ``RegisterRef``.
        """
        if not isinstance(ref, RegisterRef):
            raise TypeError(f"atom must be a RegisterRef, got {type(ref)!r}")
        found: set[RegisterRef] = set()
        for edge in self._edges:
            if ref in edge:
                found.update(edge - {ref})
        return frozenset(found)

    @property
    def edges(self) -> frozenset[frozenset[RegisterRef]]:
        """The set of paired edges, each a two-atom ``frozenset``."""
        return self._edges

    @property
    def atoms(self) -> frozenset[RegisterRef]:
        """Every atom that appears in at least one edge.

        Note this is derived from edges only: an isolated atom that was never
        paired does not appear here, since connectivity records no separate
        atom registry (existence/occupancy lives in the engine).
        """
        found: set[RegisterRef] = set()
        for edge in self._edges:
            found.update(edge)
        return frozenset(found)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AtomConnectivity):
            return NotImplemented
        return self._edges == other._edges

    def __hash__(self) -> int:
        return hash(self._edges)

    def __repr__(self) -> str:
        return f"AtomConnectivity({len(self._edges)} edge(s))"
