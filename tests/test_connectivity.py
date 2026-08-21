"""Tests AtomConnectivity: exhaustive small-topology coverage plus invariants."""

from itertools import chain, combinations

import pytest

import fatqat as fq
from fatqat.connectivity import AtomConnectivity
from fatqat.registers import QuantumRegister


def _powerset(items):
    items = list(items)
    return chain.from_iterable(combinations(items, r) for r in range(len(items) + 1))


def _build(atoms, edge_subset):
    conn = AtomConnectivity()
    for i, j in edge_subset:
        conn = conn.pair(atoms[i], atoms[j])
    return conn


def _configs(n):
    """All 2**c(n,2) edge subsets over n atoms, as ((reg, atoms), subset)."""
    reg = QuantumRegister(n, name="atoms")
    atoms = [reg[i] for i in range(n)]
    all_pairs = list(combinations(range(n), 2))
    for subset in _powerset(all_pairs):
        yield atoms, all_pairs, subset


# --- Exhaustive: every topology over 2 and 3 atoms behaves exactly as constructed ---


@pytest.mark.parametrize("n", [2, 3])
def test_exhaustive_topologies(n):
    for atoms, all_pairs, subset in _configs(n):
        conn = _build(atoms, subset)
        present = {frozenset(p) for p in subset}

        # are_paired matches the constructed edges, and is symmetric
        for i, j in all_pairs:
            want = frozenset((i, j)) in present
            assert conn.are_paired(atoms[i], atoms[j]) is want
            assert conn.are_paired(atoms[j], atoms[i]) is want

        # querying an atom against itself is always False (no self-loops)
        for a in atoms:
            assert conn.are_paired(a, a) is False

        # edges / atoms / neighbors all agree with the construction
        assert conn.edges == {frozenset((atoms[i], atoms[j])) for i, j in subset}
        assert conn.atoms == frozenset(atoms[i] for p in subset for i in p)
        for k in range(n):
            want_nb = set()
            for i, j in subset:
                if i == k:
                    want_nb.add(atoms[j])
                if j == k:
                    want_nb.add(atoms[i])
            assert conn.neighbors(atoms[k]) == frozenset(want_nb)


@pytest.mark.parametrize("n", [2, 3])
def test_config_count(n):
    assert len(list(_configs(n))) == 2 ** (n * (n - 1) // 2)


# --- Flagship: the "V" (a-b, a-c paired; b-c not) is distinct from the triangle ---


def test_v_shape_is_representable_and_distinct_from_triangle():
    atoms = fq.QuantumRegister(3, name="atoms")
    a, b, c = atoms[0], atoms[1], atoms[2]

    v = AtomConnectivity().pair(a, b).pair(a, c)
    assert v.are_paired(a, b) and v.are_paired(a, c)
    assert not v.are_paired(b, c)

    triangle = v.pair(b, c)
    assert triangle.are_paired(b, c)
    assert v != triangle


# --- Single-edge locality: one pair/unpair only ever touches that one edge ---


def test_pair_only_affects_that_edge():
    atoms = fq.QuantumRegister(3, name="atoms")
    a, b, c = atoms[0], atoms[1], atoms[2]

    conn = AtomConnectivity().pair(a, b)
    after = conn.pair(a, c)  # adding a-c must not disturb a-b
    assert after.are_paired(a, b)
    assert after.are_paired(a, c)
    assert not after.are_paired(b, c)


def test_unpair_only_affects_that_edge():
    atoms = fq.QuantumRegister(3, name="atoms")
    a, b, c = atoms[0], atoms[1], atoms[2]

    triangle = AtomConnectivity().pair(a, b).pair(a, c).pair(b, c)
    minus = triangle.unpair(a, b)
    assert not minus.are_paired(a, b)
    assert minus.are_paired(a, c)
    assert minus.are_paired(b, c)


# --- Immutability, idempotency, order-independence ----


def test_pair_is_idempotent_and_returns_self_when_unchanged():
    atoms = fq.QuantumRegister(2, name="atoms")
    a, b = atoms[0], atoms[1]
    base = AtomConnectivity().pair(a, b)
    assert base.pair(a, b) is base
    assert base.pair(a, b).pair(a, b) == base


def test_unpair_absent_edge_is_noop():
    atoms = fq.QuantumRegister(3, name="atoms")
    a, b, c = atoms[0], atoms[1], atoms[2]
    base = AtomConnectivity().pair(a, b)
    assert base.unpair(a, c) is base
    assert base.unpair(a, c) == base


def test_mutation_leaves_original_unchanged():
    atoms = fq.QuantumRegister(3, name="atoms")
    a, b, c = atoms[0], atoms[1], atoms[2]
    base = AtomConnectivity().pair(a, b)
    edges_before = base.edges
    _ = base.pair(a, c)
    assert base.edges == edges_before
    assert not base.are_paired(a, c)


def test_construction_order_independent():
    atoms = fq.QuantumRegister(3, name="atoms")
    a, b, c = atoms[0], atoms[1], atoms[2]
    # pylint: disable-next=arguments-out-of-order  # deliberately swapped: tests order-independence
    assert AtomConnectivity().pair(a, b).pair(b, c) == AtomConnectivity().pair(
        c, b
    ).pair(b, a)


# --- Validation ---


def test_self_loop_rejected():
    atoms = fq.QuantumRegister(1, name="atoms")
    a = atoms[0]
    with pytest.raises(ValueError, match="itself"):
        AtomConnectivity().pair(a, a)


@pytest.mark.parametrize("bad", [0, "x", None, 1.5])
def test_non_ref_endpoint_rejected(bad):
    atoms = fq.QuantumRegister(1, name="atoms")
    a = atoms[0]
    with pytest.raises(TypeError):
        AtomConnectivity().pair(a, bad)


def test_identity_keying_lookalike_register_is_a_different_atom():
    # Two registers built with identical args are distinct entities (eq=False),
    # so their refs are different atoms and are never found paired.
    reg1 = fq.QuantumRegister(2, name="atoms")
    reg2 = fq.QuantumRegister(2, name="atoms")
    conn = AtomConnectivity().pair(reg1[0], reg1[1])
    assert conn.are_paired(reg1[0], reg1[1])
    assert not conn.are_paired(reg2[0], reg2[1])
