"""Multi-site interaction gates and minimum-width parallel layering.

This module turns a product of single-site operators - a "restricted"
multi-site interaction such as the PXP-model terms ``PXP``, ``QXQ``, ``PXQ`` or
``QXP`` - into single- and two-qubit gates acting only on the participating
sites, and schedules a collection of such terms into the *minimum* number of
parallel layers.

Operator strings
----------------
An operator string is one character per site, chosen from:

    ``"P"``   projector onto ``|0>``   (``|0><0|``, the blockade / ground state)
    ``"Q"``   projector onto ``|1>``   (``|1><1|``, the excited state)
    ``"X"``   Pauli-X                  (``"Y"`` and ``"Z"`` likewise)
    ``"I"``   identity

A term such as ``exp(-i*theta * P X Q)`` on sites ``(n, m, k)`` therefore reads
as: *rotate site ``m`` about ``X`` by ``2*theta``, but only when site ``n`` is
``|0>`` and site ``k`` is ``|1>``*. The projector sites are the conditions and
the Pauli sites carry the rotation - exactly a multi-controlled rotation. Any
number and mixture of projector and Pauli sites is allowed (so ``PXP``,
``QXQ``, ``PXQ``, ``QXP``, ``XX``, ``XYZ``, ... all work), though the
Z-string expansion costs ``2**k`` terms and is intended for small ``k``
(roughly 3 to 5 sites).

Layering
--------
A layer may hold several terms only when they share no site (so they commute
and run in parallel). :func:`layer_schedule` returns a schedule using the
*minimum* number of layers: it colors the term-conflict graph exactly via
branch-and-bound (DSATUR upper bound + a clique lower bound), which is fast for
the small term counts of constrained models.

The central object is an :class:`Instruction`: a ``(operation, targets)`` pair.
Decomposition functions return instruction lists; :func:`to_program` and
:func:`append_instructions` turn them into a runnable :class:`fatqat.Program`.
All decompositions are exact up to a global phase, which is dropped. Angles are
numeric (derived sub-angles cannot be expressed symbolically).

Example
-------
Decompose ``PXP`` and stack an open PXP chain into three layers::

    >>> import fatqat as fq
    >>> import fatqat.multisite as ms
    >>> instr = ms.interaction((0, 1, 2), "PXP", 0.5)
    >>> prog = ms.to_program(instr, 3)
    >>> len(ms.layer_schedule([(i, i + 1, i + 2) for i in range(3)]))
    3
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

from .operations import CX, H, RZ, S, Sdg, Operation
from .program import Program

__all__ = [
    "Instruction",
    "z_string_rotation",
    "pauli_string_rotation",
    "product_interaction",
    "interaction",
    "controlled_rotation",
    "layer_schedule",
    "append_instructions",
    "to_program",
    "build_layered_programs",
]

# A decomposed gate: (operation, operand-order targets).
Instruction = tuple[Operation, tuple[int, ...]]

# Single-site operator labels accepted by `product_interaction` / `interaction`.
_OPERATOR_LABELS = frozenset({"I", "X", "Y", "Z", "P", "Q", "P0", "P1"})

# Single-qubit frame for each Pauli axis: how to rotate the axis to Z.
#
# For an axis ``p`` we emit ``before`` first, then a Z rotation, then ``after``,
# realizing ``exp(-i*theta*sigma_p)``. ``before``/``after`` are gate lists
# applied in the given order.
#
#   exp(-i*theta*X) = H RZ(2*theta) H
#   exp(-i*theta*Y) = (S H) RZ(2*theta) (H Sdg)  == Sdg, H, RZ, H, S in time order
#   exp(-i*theta*Z) = RZ(2*theta)
_PAULI_FRAME: dict[str, tuple[tuple[Operation, ...], tuple[Operation, ...]]] = {
    "X": ((H,), (H,)),
    "Y": ((Sdg, H), (H, S)),
    "Z": ((), ()),
}

# Eigen-data per operator label: ``(before, after, d0, d1)`` where the operator
# ``A`` is ``U diag(d0, d1) U^dagger``, ``before`` is ``U^dagger`` and ``after``
# is ``U``. ``before``/``after`` are gate-value lists in time order.
_LABEL_EIG: dict[
    str, tuple[tuple[Operation, ...], tuple[Operation, ...], float, float]
] = {
    "I": ((), (), 1.0, 1.0),
    "Z": ((), (), 1.0, -1.0),
    "X": ((H,), (H,), 1.0, -1.0),
    "Y": ((Sdg, H), (H, S), 1.0, -1.0),
    # P = |0><0|, Q = |1><1|; P0/P1 are retained as aliases.
    "P": ((), (), 1.0, 0.0),
    "Q": ((), (), 0.0, 1.0),
    "P0": ((), (), 1.0, 0.0),
    "P1": ((), (), 0.0, 1.0),
}


def _validate_sites(sites: Sequence[int], *, name: str) -> tuple[int, ...]:
    """Normalize and validate a sequence of qubit indices.

    Returns:
        A tuple of non-negative ``int`` qubit indices.

    Raises:
        TypeError: If an entry is not an integer.
        ValueError: If an entry is negative or duplicated.
    """
    result = tuple(sites)
    if not result:
        raise ValueError(f"{name} must not be empty")
    for site in result:
        if type(site) is not int:
            raise TypeError(f"{name} entries must be int, got {type(site).__name__!r}")
        if site < 0:
            raise ValueError(f"{name} entries must be non-negative, got {site}")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be distinct, got {result}")
    return result


def z_string_rotation(sites: Sequence[int], theta: float) -> list[Instruction]:
    """Decompose ``exp(-i*theta*Z^tensor k)`` into CNOTs and one RZ rotation.

    The operator is the tensor product of Pauli-Z on every site in ``sites``.
    It is realized by an ``(k-1)``-CNOT ladder onto the last site, one
    ``RZ(2*theta)`` there, and the ladder in reverse. This is exact and uses
    only two-qubit ``CX`` and single-qubit ``RZ`` gates.

    Args:
        sites: Qubit indices carrying the ``Z`` factors. The last entry is the
            rotation carrier. Order among the others is arbitrary.
        theta: Rotation angle in radians.

    Returns:
        The instruction list implementing the rotation. A global phase is
        dropped.

    Raises:
        TypeError: If an entry of ``sites`` is not an ``int``.
        ValueError: If ``sites`` is empty or has a duplicate or negative entry.

    Examples:
        ``exp(-i*theta*Z0 Z1)`` needs one ladder CNOT and one rotation::

            >>> import fatqat.multisite as ms
            >>> len(ms.z_string_rotation((0, 1), 0.3))
            3
    """
    sites = _validate_sites(sites, name="sites")
    theta = float(theta)
    if len(sites) == 1:
        return [(RZ(2.0 * theta), (sites[0],))]
    last = sites[-1]
    instructions: list[Instruction] = []
    for site in sites[:-1]:
        instructions.append((CX, (site, last)))
    instructions.append((RZ(2.0 * theta), (last,)))
    for site in reversed(sites[:-1]):
        instructions.append((CX, (site, last)))
    return instructions


def pauli_string_rotation(
    sites: Sequence[int],
    paulis: str | Sequence[str],
    theta: float,
) -> list[Instruction]:
    """Decompose ``exp(-i*theta * tensor(sigma_p))`` into 1- and 2-qubit gates.

    Each factor is a Pauli ``"X"``, ``"Y"``, or ``"Z"`` (``"I"`` factors are
    ignored). Every non-identity factor is rotated to ``Z`` with single-qubit
    frame gates, the resulting Z-string is applied with
    :func:`z_string_rotation`, and the frames are undone.

    Args:
        sites: Qubit indices, one per Pauli factor, in operand order.
        paulis: A string (e.g. ``"XYZ"``) or sequence of ``"X"``/``"Y"``/
            ``"Z"``/``"I"`` labels of the same length as ``sites``.
        theta: Rotation angle in radians.

    Returns:
        The instruction list. A global phase is dropped.

    Raises:
        TypeError: If a label is not a string.
        ValueError: If ``sites`` and ``paulis`` have different lengths, a label
            is unknown, or a site is invalid.

    Examples:
        ``exp(-i*theta*X0 X1)`` wraps a ZZ rotation in Hadamards::

            >>> import fatqat.multisite as ms
            >>> instr = ms.pauli_string_rotation((0, 1), "XX", 0.5)
            >>> [op.name for op, _ in instr]
            ['H', 'H', 'CX', 'RZ', 'CX', 'H', 'H']
    """
    sites = _validate_sites(sites, name="sites")
    if isinstance(paulis, str):
        paulis = tuple(paulis)
    else:
        paulis = tuple(paulis)
    if len(paulis) != len(sites):
        raise ValueError(
            f"paulis has length {len(paulis)} but sites has length {len(sites)}"
        )
    for label in paulis:
        if label not in _PAULI_FRAME and label != "I":
            raise ValueError(
                f"unknown Pauli label {label!r}; expected one of 'X', 'Y', 'Z', 'I'"
            )

    active = [(s, p) for s, p in zip(sites, paulis) if p != "I"]
    if not active:
        return []

    instructions: list[Instruction] = []
    for site, label in active:
        before, _after = _PAULI_FRAME[label]
        for gate in before:
            instructions.append((gate, (site,)))
    instructions.extend(z_string_rotation([s for s, _ in active], theta))
    for site, label in reversed(active):
        _before, after = _PAULI_FRAME[label]
        for gate in after:
            instructions.append((gate, (site,)))
    return instructions


def product_interaction(
    operators: Sequence[tuple[int, str]],
    theta: float,
) -> list[Instruction]:
    """Decompose ``exp(-i*theta * tensor(A_j))`` into 1- and 2-qubit gates.

    Each ``(site, label)`` pair names a single-site Hermitian operator: ``"I"``
    (identity), ``"X"``/``"Y"``/``"Z"`` (Pauli), or ``"P"``/``"Q"`` (the
    projectors ``|0><0|`` and ``|1><1|``; ``"P0"``/``"P1"`` are aliases).
    Every operator is diagonalized, the tensor product of diagonals is expanded
    into commuting Z-strings, and each Z-string is realized with
    :func:`z_string_rotation`. The expansion has ``2**k`` terms, so this is
    intended for small ``k`` (roughly 3 to 5 sites).

    Args:
        operators: ``(site, label)`` pairs in tensor order.
        theta: Rotation angle in radians.

    Returns:
        The instruction list. A global phase is dropped.

    Raises:
        TypeError: If ``site`` is not an ``int`` or ``label`` is not a string.
        ValueError: If a site is invalid, duplicated, or a label is unknown.
    """
    operators = tuple(operators)
    if not operators:
        raise ValueError("operators must not be empty")

    sites = tuple(site for site, _label in operators)
    sites = _validate_sites(sites, name="sites")
    entries: list[tuple[tuple[Operation, ...], tuple[Operation, ...], float, float]] = (
        []
    )
    for _site, label in operators:
        if label not in _LABEL_EIG:
            raise ValueError(
                f"unknown operator label {label!r}; expected one of "
                f"{sorted(_OPERATOR_LABELS)}"
            )
        before, after, d0, d1 = _LABEL_EIG[label]
        entries.append((before, after, float(d0), float(d1)))

    instructions: list[Instruction] = []
    # 1. Diagonalizing frame: apply U^dagger for every site.
    for site, (before, _after, _d0, _d1) in zip(sites, entries):
        for gate in before:
            instructions.append((gate, (site,)))

    # 2. Tensor of diagonals -> sum over Z-strings.
    #    diag(d0, d1) = m*I + s*Z with m=(d0+d1)/2, s=(d0-d1)/2.
    means = [(d0 + d1) / 2.0 for _b, _a, d0, d1 in entries]
    spreads = [(d0 - d1) / 2.0 for _b, _a, d0, d1 in entries]
    k = len(sites)
    for size in range(1, k + 1):
        for subset in combinations(range(k), size):
            coefficient = 1.0
            for j in range(k):
                coefficient *= spreads[j] if j in subset else means[j]
            if coefficient == 0.0:
                continue
            subset_sites = tuple(sites[j] for j in subset)
            instructions.extend(z_string_rotation(subset_sites, theta * coefficient))

    # 3. Undo the frames: apply U for every site, in reverse.
    for site, (_before, after, _d0, _d1) in reversed(list(zip(sites, entries))):
        for gate in after:
            instructions.append((gate, (site,)))

    return instructions


def interaction(
    sites: Sequence[int],
    opstring: str,
    theta: float,
) -> list[Instruction]:
    """Decompose ``exp(-i*theta * tensor(op_j))`` described by an operator string.

    ``opstring`` is one character per site, chosen from ``"P"``, ``"Q"``,
    ``"X"``, ``"Y"``, ``"Z"``, ``"I"``. This is the natural spelling of a
    restricted multi-site term: ``"PXP"`` is the PXP blockade term,
    ``"QXQ"``/``"PXQ"``/``"QXP"`` place the conditions and rotation on
    different sites, and ``"XYZ"`` is a generic three-body rotation.

    Args:
        sites: Qubit indices, one per character, in tensor order.
        opstring: Operator string of the same length as ``sites``.
        theta: Rotation angle in radians.

    Returns:
        The instruction list. A global phase is dropped.

    Raises:
        TypeError: If ``opstring`` is not a string.
        ValueError: If ``sites`` and ``opstring`` have different lengths, a
            character is unknown, or a site is invalid.

    Examples:
        The three-site PXP term::

            >>> import fatqat.multisite as ms
            >>> len(ms.interaction((0, 1, 2), "PXP", 0.4)) > 0
            True
    """
    sites = _validate_sites(sites, name="sites")
    if not isinstance(opstring, str):
        raise TypeError(f"opstring must be a string, got {type(opstring).__name__!r}")
    labels = tuple(opstring)
    if len(labels) != len(sites):
        raise ValueError(
            f"opstring has length {len(labels)} but sites has length {len(sites)}"
        )
    for label in labels:
        if label not in _OPERATOR_LABELS:
            raise ValueError(
                f"unknown operator character {label!r}; expected one of "
                f"{sorted(_OPERATOR_LABELS)}"
            )
    return product_interaction(tuple(zip(sites, labels)), theta)


def _normalize_control_states(
    control_state: int | Sequence[int], count: int
) -> tuple[int, ...]:
    """Normalize a control-state spec to one 0/1 flag per control site."""
    if type(control_state) is int:
        if control_state not in (0, 1):
            raise ValueError(f"control_state must be 0 or 1, got {control_state!r}")
        return (control_state,) * count
    states = tuple(control_state)
    if len(states) != count:
        raise ValueError(
            f"control_state has length {len(states)} but there are {count} controls"
        )
    for state in states:
        if type(state) is not int or state not in (0, 1):
            raise ValueError(f"control states must be 0 or 1, got {state!r}")
    return states


def controlled_rotation(
    controls: Sequence[int],
    control_states: int | Sequence[int],
    target: int,
    theta: float,
    *,
    axis: str = "X",
) -> list[Instruction]:
    """Decompose a multi-controlled rotation with per-site conditions.

    The operator is ``exp(-i*theta * (tensor of P_c) tensor axis(target))``
    where each control site ``c`` contributes the projector ``|0><0|`` when its
    state is ``0`` or ``|1><1|`` when its state is ``1``, and ``axis`` is the
    Pauli operator ``"X"``, ``"Y"``, or ``"Z"`` on the target. The rotation
    therefore acts only when every control sits in its required state.

    Args:
        controls: The condition (projector) sites.
        control_states: One ``0``/``1`` per control (``0`` = ``|0><0|``, ``1``
            = ``|1><1|``), or a single ``0``/``1`` applied to every control.
        target: The single rotation site. Must differ from every control.
        theta: Rotation angle in radians.
        axis: Pauli axis of the target.

    Returns:
        The instruction list. A global phase is dropped.

    Raises:
        TypeError: If ``target`` or a control is not an ``int``.
        ValueError: If the target overlaps a control, ``axis`` is unknown, or a
            control state is not ``0``/``1``.

    Examples:
        ``P X Q`` on sites ``(0, 1, 2)`` has controls ``(0, 2)`` in states
        ``(0, 1)``::

            >>> import fatqat.multisite as ms
            >>> len(ms.controlled_rotation((0, 2), (0, 1), 1, 0.4)) > 0
            True
    """
    controls = _validate_sites(controls, name="controls")
    if type(target) is not int:
        raise TypeError(f"target must be int, got {type(target).__name__!r}")
    if target < 0:
        raise ValueError(f"target must be non-negative, got {target}")
    if target in controls:
        raise ValueError(f"target {target} must not overlap controls {controls}")
    if axis not in _PAULI_FRAME:
        raise ValueError(f"axis must be 'X', 'Y', or 'Z', got {axis!r}")
    states = _normalize_control_states(control_states, len(controls))

    labels = ("P" if state == 0 else "Q" for state in states)
    return product_interaction(
        tuple((site, label) for site, label in zip(controls, labels))
        + ((target, axis),),
        theta,
    )


# ---------------------------------------------------------------------------
# Minimum-width parallel layering
# ---------------------------------------------------------------------------


def _conflict_graph(terms: Sequence[tuple[int, ...]]) -> list[set[int]]:
    """Adjacency list of the term-conflict graph (edge iff terms share a site)."""
    n = len(terms)
    adjacency: list[set[int]] = [set() for _ in range(n)]
    site_sets = [set(term) for term in terms]
    for i in range(n):
        for j in range(i + 1, n):
            if site_sets[i] & site_sets[j]:
                adjacency[i].add(j)
                adjacency[j].add(i)
    return adjacency


def _clique_lower_bound(terms: Sequence[tuple[int, ...]]) -> int:
    """Largest number of terms containing any single site (a clique bound)."""
    counts: dict[int, int] = {}
    for term in terms:
        for site in term:
            counts[site] = counts.get(site, 0) + 1
    return max(counts.values(), default=0)


def _dsatur_coloring(adjacency: Sequence[set[int]]) -> list[int]:
    """Greedy DSATUR coloring; returns a valid (not necessarily optimal) color list."""
    n = len(adjacency)
    color = [-1] * n

    def saturation(node: int) -> int:
        return len({color[nbr] for nbr in adjacency[node] if color[nbr] != -1})

    for _ in range(n):
        chosen = -1
        chosen_key = (-1, -1)
        for node in range(n):
            if color[node] != -1:
                continue
            key = (saturation(node), len(adjacency[node]))
            if key > chosen_key:
                chosen_key = key
                chosen = node
        if chosen == -1:
            break
        used = {color[nbr] for nbr in adjacency[chosen] if color[nbr] != -1}
        assigned = 0
        while assigned in used:
            assigned += 1
        color[chosen] = assigned
    return color


def _minimum_coloring(adjacency: Sequence[set[int]]) -> list[int]:
    """Exact minimum graph coloring by branch-and-bound (DSATUR ordering)."""
    n = len(adjacency)
    if n == 0:
        return []

    greedy = _dsatur_coloring(adjacency)
    upper = max(greedy) + 1
    best = upper
    best_coloring = list(greedy)
    color = [-1] * n

    def saturation(node: int) -> int:
        return len({color[nbr] for nbr in adjacency[node] if color[nbr] != -1})

    def choose_next() -> int:
        chosen = -1
        chosen_key = (-1, -1)
        for node in range(n):
            if color[node] != -1:
                continue
            key = (saturation(node), len(adjacency[node]))
            if key > chosen_key:
                chosen_key = key
                chosen = node
        return chosen

    def search(colored: int, used: int) -> None:
        nonlocal best, best_coloring
        if used >= best:
            return
        if colored == n:
            best = used
            best_coloring = color[:]
            return
        node = choose_next()
        if node == -1:
            return
        for candidate in range(used):
            if all(color[nbr] != candidate for nbr in adjacency[node]):
                color[node] = candidate
                search(colored + 1, used)
                color[node] = -1
        if used + 1 < best:
            color[node] = used
            search(colored + 1, used + 1)
            color[node] = -1

    search(0, 0)
    return best_coloring


def layer_schedule(
    terms: Sequence[Sequence[int]], *, optimal: bool = True
) -> list[list[tuple[int, ...]]]:
    """Partition site tuples into the minimum number of parallel layers.

    Two terms may share a layer only when they share no site. With
    ``optimal=True`` (the default) the schedule uses the fewest possible
    layers: the term-conflict graph is colored exactly by branch-and-bound.
    This is fast for the small term counts of constrained models but is
    exponential in the worst case. Set ``optimal=False`` for a linear-time
    greedy schedule instead.

    Args:
        terms: Site tuples (triples, 4-tuples, ...) to schedule.
        optimal: Whether to search for the minimum layer count.

    Returns:
        A list of layers; each layer is a list of the original tuples (kept in
        input order) that can run in parallel.

    Raises:
        TypeError: If a term is not a sequence of ``int``.
        ValueError: If a term is empty or contains a duplicate or negative site.

    Examples:
        Disjoint triples share one layer; three mutually overlapping triples
        need three::

            >>> import fatqat.multisite as ms
            >>> ms.layer_schedule([(0, 1, 2), (3, 4, 5)])
            [[(0, 1, 2), (3, 4, 5)]]
            >>> len(ms.layer_schedule([(0, 1, 2), (1, 2, 3), (2, 3, 4)]))
            3
    """
    normalized = [_validate_sites(term, name="term") for term in terms]
    if not normalized:
        return []
    if optimal:
        coloring = _minimum_coloring(_conflict_graph(normalized))
    else:
        coloring = _dsatur_coloring(_conflict_graph(normalized))

    num_layers = max(coloring) + 1
    layers: list[list[tuple[int, ...]]] = [[] for _ in range(num_layers)]
    for term, color in zip(normalized, coloring):
        layers[color].append(term)
    return layers


# ---------------------------------------------------------------------------
# Program assembly
# ---------------------------------------------------------------------------


def append_instructions(
    program: Program, instructions: Sequence[Instruction]
) -> Program:
    """Append decomposed instructions to an existing program in place.

    Args:
        program: The program to extend.
        instructions: ``(operation, targets)`` pairs to append.

    Returns:
        ``program``, for chaining.
    """
    for operation, targets in instructions:
        program.add(operation, targets)
    return program


def to_program(instructions: Sequence[Instruction], num_qubits: int) -> Program:
    """Assemble decomposed instructions into a single program.

    Args:
        instructions: ``(operation, targets)`` pairs.
        num_qubits: Number of qubits to allocate. Every target index must be
            below this bound.

    Returns:
        A new :class:`fatqat.Program` containing the instructions in order.

    Raises:
        IndexError: If an instruction targets a qubit at or beyond ``num_qubits``.
    """
    if type(num_qubits) is not int or num_qubits < 0:
        raise ValueError(f"num_qubits must be a non-negative int, got {num_qubits!r}")
    program = Program(num_qubits)
    return append_instructions(program, instructions)


def _normalize_terms(
    terms: Sequence[Sequence[int] | tuple[Sequence[int], str]],
    opstring: str | None,
) -> list[tuple[tuple[int, ...], str]]:
    """Normalize ``terms`` to ``(sites_tuple, opstring)`` pairs.

    Each item is either a site tuple (used with the uniform ``opstring``) or a
    ``(sites, opstring)`` pair (when the second element is a string).
    """
    if opstring is not None and not isinstance(opstring, str):
        raise TypeError(
            f"opstring must be a string or None, got {type(opstring).__name__!r}"
        )

    normalized: list[tuple[tuple[int, ...], str]] = []
    for term in terms:
        if (
            isinstance(term, tuple)
            and len(term) == 2
            and isinstance(term[1], str)
            and not isinstance(term[0], int)
        ):
            sites, ops = tuple(term[0]), term[1]
        else:
            if opstring is None:
                raise ValueError(
                    "opstring is required when terms are plain site tuples"
                )
            sites, ops = tuple(term), opstring
        normalized.append((sites, ops))
    return normalized


def build_layered_programs(
    terms: Sequence[Sequence[int] | tuple[Sequence[int], str]],
    theta: float,
    *,
    opstring: str | None = "PXP",
    num_qubits: int | None = None,
    optimal: bool = True,
) -> list[Program]:
    """Build one parallel program per layer for a collection of interaction terms.

    Each term is decomposed with :func:`interaction`, the terms are packed into
    the minimum number of disjoint layers with :func:`layer_schedule`, and every
    layer is assembled into its own :class:`fatqat.Program`. Running the
    returned programs in order applies the full constrained-model Trotter step.

    Args:
        terms: Site tuples (with a uniform ``opstring``), or ``(sites,
            opstring)`` pairs for per-term operators.
        theta: Rotation angle in radians for every term.
        opstring: Uniform operator string applied to every site tuple. Ignored
            when ``terms`` entries already carry their own operator string;
            required in that case only when a plain site tuple appears.
        num_qubits: Total qubit count. Defaults to ``max(site) + 1`` over all
            terms.
        optimal: Whether to minimize the number of layers.

    Returns:
        One program per layer, in schedule order.

    Raises:
        ValueError: If ``terms`` is empty or ``num_qubits`` is too small.
    """
    normalized = _normalize_terms(terms, opstring)
    if not normalized:
        raise ValueError("terms must not be empty")

    layers = layer_schedule([sites for sites, _ops in normalized], optimal=optimal)
    if num_qubits is None:
        num_qubits = max(site for term in terms for site in _sites_of(term)) + 1
    if type(num_qubits) is not int or num_qubits < 0:
        raise ValueError(f"num_qubits must be a non-negative int, got {num_qubits!r}")

    op_by_sites = dict(normalized)
    programs: list[Program] = []
    for layer in layers:
        program = Program(num_qubits)
        for sites in layer:
            append_instructions(program, interaction(sites, op_by_sites[sites], theta))
        programs.append(program)
    return programs


def _sites_of(term: Sequence[int] | tuple[Sequence[int], str]) -> Sequence[int]:
    """Return the site sequence of a raw term item (site tuple or pair)."""
    if (
        isinstance(term, tuple)
        and len(term) == 2
        and isinstance(term[1], str)
        and not isinstance(term[0], int)
    ):
        return term[0]
    return term
