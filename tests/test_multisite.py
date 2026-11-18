"""Tests for the multisite (restricted interaction + minimum layering) module.

Every decomposition is verified numerically: the emitted instructions are
assembled into a program and its unitary is compared - up to a global phase -
against the exact exponential of the target operator built with NumPy/SciPy.
The minimum-layering algorithm is checked against a brute-force minimum.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

import fatqat as fq
import fatqat.multisite as ms

# Single-site operator matrices (|0>, |1> basis order).
_SINGLE = {
    "I": np.eye(2),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "Y": np.array([[0.0, -1j], [1j, 0.0]]),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]]),
    "P": np.array([[1.0, 0.0], [0.0, 0.0]]),  # |0><0|
    "Q": np.array([[0.0, 0.0], [0.0, 1.0]]),  # |1><1|
}


def _full_operator(num_qubits: int, site_ops: dict[int, np.ndarray]) -> np.ndarray:
    """Kron single-site operators in most-significant-first (fatqat) order."""
    operator: np.ndarray = np.array([[1.0 + 0.0j]])
    for site in range(num_qubits):
        operator = np.kron(operator, site_ops.get(site, _SINGLE["I"]))
    return operator


def _target_unitary(
    num_qubits: int, site_ops: dict[int, np.ndarray], theta: float
) -> np.ndarray:
    """Exact ``exp(-i * theta * tensor(A_j))`` in fatqat's public basis."""
    return expm(-1j * theta * _full_operator(num_qubits, site_ops))


def _target_from_opstring(
    num_qubits: int, sites, opstring: str, theta: float
) -> np.ndarray:
    site_ops = {s: _SINGLE[ch] for s, ch in zip(sites, opstring)}
    return _target_unitary(num_qubits, site_ops, theta)


def _unitary(instructions, num_qubits: int) -> np.ndarray:
    """Assemble and run instructions, returning the fatqat-computed unitary."""
    program = ms.to_program(instructions, num_qubits)
    backend = fq.simulator.Simulator(method="unitary", runtime="numpy")
    result = backend.run(
        program, result_config={"counts": False, "final_state": True}
    ).result()
    return result.get_unitary()


def _assert_same_unitary(actual: np.ndarray, expected: np.ndarray, atol: float = 1e-8):
    """Assert ``actual == exp(i*phi) * expected`` for some global phase."""
    inner = np.vdot(actual, expected)  # trace(actual^dag @ expected)
    assert abs(inner) > 1e-9, "unitaries are orthogonal, not equal up to phase"
    phase = inner / abs(inner)
    np.testing.assert_allclose(expected, phase * actual, atol=atol, rtol=atol)


def _can_color(terms, k: int) -> bool:
    """Whether ``terms`` can be scheduled into ``k`` disjoint layers."""
    n = len(terms)
    sets = [set(term) for term in terms]
    assign = [-1] * n

    def rec(i: int) -> bool:
        if i == n:
            return True
        for color in range(k):
            if all(not (sets[i] & sets[j]) for j in range(i) if assign[j] == color):
                assign[i] = color
                if rec(i + 1):
                    return True
                assign[i] = -1
        return False

    return rec(0)


def _brute_min_layers(terms) -> int:
    """Exact minimum layer count by backtracking over layer assignments."""
    for k in range(1, len(terms) + 1):
        if _can_color(terms, k):
            return k
    return len(terms)


# ---------------------------------------------------------------------------
# z_string_rotation / pauli_string_rotation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sites", [(0, 1), (1, 0), (0, 1, 2), (0, 2, 1)])
def test_z_string_rotation(sites):
    theta = 0.7
    instructions = ms.z_string_rotation(sites, theta)
    expected = _target_unitary(max(sites) + 1, {s: _SINGLE["Z"] for s in sites}, theta)
    _assert_same_unitary(_unitary(instructions, max(sites) + 1), expected)


@pytest.mark.parametrize("paulis", ["X", "Y", "Z", "XX", "YY", "XY", "XZ", "XYZ"])
def test_pauli_string_rotation(paulis):
    sites = tuple(range(len(paulis)))
    active = {i: _SINGLE[p] for i, p in enumerate(paulis) if p != "I"}
    theta = 0.6
    instructions = ms.pauli_string_rotation(sites, paulis, theta)
    expected = _target_unitary(len(paulis), active, theta)
    _assert_same_unitary(_unitary(instructions, len(paulis)), expected)


# ---------------------------------------------------------------------------
# product_interaction (with P/Q labels)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operators",
    [
        [(0, "P"), (1, "X"), (2, "P")],  # PXP
        [(0, "Q"), (1, "X"), (2, "Q")],  # QXQ
        [(0, "P"), (1, "X"), (2, "Q")],  # PXQ
        [(0, "Q"), (1, "X"), (2, "P")],  # QXP
        [(0, "X"), (1, "Y"), (2, "Z")],
        [(0, "I"), (1, "P"), (2, "Z")],
    ],
)
def test_product_interaction(operators):
    num_qubits = max(site for site, _ in operators) + 1
    site_ops = {site: _SINGLE[label] for site, label in operators}
    theta = 0.55
    instructions = ms.product_interaction(operators, theta)
    expected = _target_unitary(num_qubits, site_ops, theta)
    _assert_same_unitary(_unitary(instructions, num_qubits), expected)


# ---------------------------------------------------------------------------
# interaction (operator strings)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "opstring",
    ["PXP", "QXQ", "PXQ", "QXP", "XYZ", "PYP", "XX", "PQP", "PZQ"],
)
def test_interaction(opstring):
    sites = tuple(range(len(opstring)))
    theta = 0.45
    instructions = ms.interaction(sites, opstring, theta)
    expected = _target_from_opstring(len(opstring), sites, opstring, theta)
    _assert_same_unitary(_unitary(instructions, len(opstring)), expected)


def test_interaction_on_arbitrary_sites():
    # Opstring applies to the given (not necessarily 0..k-1) sites.
    sites = (3, 0, 5)
    opstring = "PXQ"
    theta = 0.5
    instructions = ms.interaction(sites, opstring, theta)
    expected = _target_from_opstring(6, sites, opstring, theta)
    _assert_same_unitary(_unitary(instructions, 6), expected)


# ---------------------------------------------------------------------------
# controlled_rotation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
@pytest.mark.parametrize(
    "control_states",
    [(0, 0), (1, 1), (0, 1), (1, 0)],
)
def test_controlled_rotation_mixed_states(axis, control_states):
    theta = 0.4
    controls = (0, 2)
    target = 1
    instructions = ms.controlled_rotation(
        controls, control_states, target, theta, axis=axis
    )
    site_ops = {
        0: _SINGLE["P" if control_states[0] == 0 else "Q"],
        2: _SINGLE["P" if control_states[1] == 0 else "Q"],
        1: _SINGLE[axis],
    }
    expected = _target_unitary(3, site_ops, theta)
    _assert_same_unitary(_unitary(instructions, 3), expected)


def test_controlled_rotation_scalar_state():
    theta = 0.4
    instructions = ms.controlled_rotation((0, 2), 1, 1, theta, axis="X")
    expected = _target_unitary(
        3, {0: _SINGLE["Q"], 2: _SINGLE["Q"], 1: _SINGLE["X"]}, theta
    )
    _assert_same_unitary(_unitary(instructions, 3), expected)


def test_controlled_rotation_four_body():
    theta = 0.35
    controls = (0, 1, 3)
    target = 2
    instructions = ms.controlled_rotation(controls, (0, 1, 0), target, theta)
    site_ops = {0: _SINGLE["P"], 1: _SINGLE["Q"], 3: _SINGLE["P"], 2: _SINGLE["X"]}
    expected = _target_unitary(4, site_ops, theta)
    _assert_same_unitary(_unitary(instructions, 4), expected)


# ---------------------------------------------------------------------------
# layer_schedule (minimum-width)
# ---------------------------------------------------------------------------


def test_layer_schedule_disjoint_share_layer():
    assert ms.layer_schedule([(0, 1, 2), (3, 4, 5)]) == [[(0, 1, 2), (3, 4, 5)]]


def test_layer_schedule_pxp_chain_is_three_layers():
    terms = [(i, i + 1, i + 2) for i in range(6)]
    layers = ms.layer_schedule(terms)
    assert len(layers) == 3
    for layer in layers:
        seen: set[int] = set()
        for term in layer:
            assert seen.isdisjoint(term)
            seen.update(term)


def test_layer_schedule_mutually_overlapping_triples():
    # Three triples that all pairwise overlap need three layers.
    layers = ms.layer_schedule([(0, 1, 2), (1, 2, 3), (2, 3, 4)])
    assert len(layers) == 3


def test_layer_schedule_matches_bruteforce_random():
    rng = np.random.RandomState(7)
    for _ in range(200):
        n = rng.randint(1, 7)
        terms = [
            tuple(sorted(rng.choice(6, size=rng.randint(2, 4), replace=False).tolist()))
            for _ in range(n)
        ]
        layers = ms.layer_schedule(terms, optimal=True)
        # Valid schedule.
        for layer in layers:
            seen: set[int] = set()
            for term in layer:
                assert seen.isdisjoint(term)
                seen.update(term)
        assert len(layers) == _brute_min_layers(terms)


def test_layer_schedule_optimal_fewer_than_greedy():
    # A 5-cycle conflict graph: exact minimum is 3, a naive first-fit in a bad
    # order can use more. Build 5 terms forming a cycle A-B-C-D-E-A.
    terms = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    assert _brute_min_layers(terms) == 3
    assert len(ms.layer_schedule(terms, optimal=True)) == 3


def test_layer_schedule_greedy_mode():
    terms = [(0, 1, 2), (3, 4, 5)]
    assert len(ms.layer_schedule(terms, optimal=False)) == 1


def test_layer_schedule_validates():
    with pytest.raises(ValueError):
        ms.layer_schedule([(0, 1, 1)])
    with pytest.raises(ValueError):
        ms.layer_schedule([(0, -1, 2)])
    with pytest.raises(ValueError):
        ms.layer_schedule([()])


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_interaction_validation():
    with pytest.raises(ValueError):
        ms.interaction((0, 1, 2), "PX", 0.5)  # length mismatch
    with pytest.raises(ValueError):
        ms.interaction((0, 1, 2), "PRX", 0.5)  # unknown char
    with pytest.raises(TypeError):
        ms.interaction((0, 1, 2), 123, 0.5)


def test_controlled_rotation_validation():
    with pytest.raises(ValueError):
        ms.controlled_rotation((0, 1), (0, 0), 1, 0.5)  # target overlaps control
    with pytest.raises(ValueError):
        ms.controlled_rotation((0, 2), (0, 1, 0), 1, 0.5)  # wrong state count
    with pytest.raises(ValueError):
        ms.controlled_rotation((0, 2), (0, 2), 1, 0.5)  # bad state value
    with pytest.raises(ValueError):
        ms.controlled_rotation((0, 2), (0, 0), 1, 0.5, axis="W")


def test_product_interaction_validation():
    with pytest.raises(ValueError):
        ms.product_interaction([(0, "R")], 0.5)
    with pytest.raises(ValueError):
        ms.product_interaction([], 0.5)


# ---------------------------------------------------------------------------
# end-to-end layering
# ---------------------------------------------------------------------------


def test_build_layered_programs_uniform_opstring():
    terms = [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5)]
    num_qubits = 6
    theta = 0.3

    layers = ms.layer_schedule(terms)
    flat = [term for layer in layers for term in layer]
    reference = np.eye(2**num_qubits)
    for term in flat:
        reference = _target_from_opstring(num_qubits, term, "PXP", theta) @ reference

    programs = ms.build_layered_programs(terms, theta, num_qubits=num_qubits)
    assert len(programs) == len(layers)
    backend = fq.simulator.Simulator(method="unitary", runtime="numpy")
    combined = np.eye(2**num_qubits)
    for program in programs:
        result = backend.run(
            program, result_config={"counts": False, "final_state": True}
        ).result()
        combined = result.get_unitary() @ combined
    _assert_same_unitary(combined, reference)


def test_build_layered_programs_per_term_opstrings():
    # Non-uniform operators: PXP and QXQ terms in one schedule.
    terms = [((0, 1, 2), "PXP"), ((3, 4, 5), "QXQ"), ((1, 2, 3), "PXQ")]
    num_qubits = 6
    theta = 0.3

    programs = ms.build_layered_programs(terms, theta, num_qubits=num_qubits)
    layers = ms.layer_schedule([s for s, _ in terms])
    assert len(programs) == len(layers)

    flat = [item for layer in layers for item in layer]
    op_by_sites = dict(terms)
    reference = np.eye(2**num_qubits)
    for sites in flat:
        reference = (
            _target_from_opstring(num_qubits, sites, op_by_sites[sites], theta)
            @ reference
        )

    backend = fq.simulator.Simulator(method="unitary", runtime="numpy")
    combined = np.eye(2**num_qubits)
    for program in programs:
        result = backend.run(
            program, result_config={"counts": False, "final_state": True}
        ).result()
        combined = result.get_unitary() @ combined
    _assert_same_unitary(combined, reference)


def test_build_layered_programs_num_qubits_default():
    programs = ms.build_layered_programs([(0, 1, 2), (4, 5, 6)], 0.1)
    assert len(programs) == 1
    assert sum(reg.size for reg in programs[0].quantum_registers) == 7
