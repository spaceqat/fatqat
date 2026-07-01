"""Tests statevector engine probabilities, sampling, and collapse."""

import numpy as np

from qnsim.engine import StateVectorEngine
from qnsim.implementation import MatrixImplementation


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _h_engine(n_qubits, target):
    eng = StateVectorEngine()
    eng.initialize(n_qubits)
    eng.apply(MatrixImplementation(matrix=_H, target_indices=(target,)))
    return eng


def test_probabilities():
    eng = _h_engine(1, 0)
    assert np.allclose(eng.probabilities(), [0.5, 0.5])


def test_sample_indices_deterministic_state():
    eng = StateVectorEngine()
    eng.initialize(2)  # always |00> -> index 0
    rng = np.random.default_rng(0)
    idx = eng.sample_indices(100, rng)
    assert idx.shape == (100,)
    assert np.all(idx == 0)


def test_sample_indices_balanced_with_seed():
    eng = _h_engine(1, 0)
    rng = np.random.default_rng(42)
    idx = eng.sample_indices(2000, rng)
    frac_one = np.mean(idx == 1)
    assert 0.45 < frac_one < 0.55


def test_collapse_returns_flat_index_and_projects():
    eng = _h_engine(1, 0)
    rng = np.random.default_rng(1)
    idx = eng.collapse([0], rng)
    assert idx in (0, 1)
    expected = np.zeros(2, dtype=complex)
    expected[idx] = 1.0
    assert np.allclose(eng.export_state(), expected)
    assert np.isclose(np.linalg.norm(eng.export_state()), 1.0)
