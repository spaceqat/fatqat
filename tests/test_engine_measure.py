import numpy as np

from qnsim.engine import zero_state, apply, probabilities, sample_indices, collapse


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def test_probabilities():
    s = apply(zero_state(1), _H, (0,), 1)
    assert np.allclose(probabilities(s), [0.5, 0.5])


def test_sample_indices_deterministic_state():
    s = zero_state(2)  # always |00> -> index 0
    rng = np.random.default_rng(0)
    idx = sample_indices(s, 100, rng)
    assert idx.shape == (100,)
    assert np.all(idx == 0)


def test_sample_indices_balanced_with_seed():
    s = apply(zero_state(1), _H, (0,), 1)
    rng = np.random.default_rng(42)
    idx = sample_indices(s, 2000, rng)
    frac_one = np.mean(idx == 1)
    assert 0.45 < frac_one < 0.55


def test_collapse_projects_to_basis_state():
    s = apply(zero_state(1), _H, (0,), 1)
    rng = np.random.default_rng(1)
    collapsed, bits = collapse(s, 1, [0], rng)
    outcome = bits[0]
    expected = np.zeros(2, dtype=complex)
    expected[outcome] = 1.0
    assert np.allclose(collapsed, expected)
    assert np.isclose(np.linalg.norm(collapsed), 1.0)
