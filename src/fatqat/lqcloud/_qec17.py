"""Fixed physical constraints for LQCloud's QZ01-surface_code processor."""

NUM_QUBITS = 17

# Undirected physical CZ couplings; no routing is performed by the adapter.
CZ_EDGES = frozenset(
    frozenset(edge)
    for edge in (
        (6, 15),
        (16, 3),
        (3, 11),
        (11, 7),
        (0, 9),
        (9, 4),
        (4, 12),
        (12, 8),
        (1, 10),
        (10, 5),
        (5, 14),
        (13, 2),
        (6, 11),
        (15, 7),
        (16, 0),
        (3, 9),
        (11, 4),
        (7, 12),
        (9, 1),
        (4, 10),
        (12, 5),
        (8, 14),
        (1, 13),
        (10, 2),
    )
)
