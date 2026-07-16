"""Tests statevector simulator probabilities, sampling, and collapse."""

import numpy as np

from fatqat.simulator.np import NumpySVSimulator
from fatqat.backends.steps import ApplyMatrixStep

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _h_engine(n_qubits, target):
    eng = NumpySVSimulator()
    eng.initialize((2,) * n_qubits)
    eng.apply(ApplyMatrixStep(matrix=_H, target_indices=(target,)))
    return eng


def test_probabilities():
    eng = _h_engine(1, 0)
    assert np.allclose(eng.probabilities(), [0.5, 0.5])


def test_sample_indices_deterministic_state():
    eng = NumpySVSimulator()
    eng.initialize((2, 2))  # always |00> -> index 0
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


def test_collapse_partial_measurement_preserves_compatible_superposition():
    eng = _h_engine(2, 0)
    rng = np.random.default_rng(1)
    idx = eng.collapse([1], rng)

    bit = (idx >> 1) & 1
    expected = np.zeros(4, dtype=complex)
    expected[bit << 1] = 1 / np.sqrt(2)
    expected[(bit << 1) | 1] = 1 / np.sqrt(2)

    assert np.allclose(eng.export_state(), expected)
    assert np.isclose(np.linalg.norm(eng.export_state()), 1.0)


def test_collapse_all_qubits_projects_to_single_basis_state():
    eng = _h_engine(2, 0)
    eng.apply(ApplyMatrixStep(matrix=_H, target_indices=(1,)))
    rng = np.random.default_rng(2)
    idx = eng.collapse([0, 1], rng)

    expected = np.zeros(4, dtype=complex)
    expected[idx] = 1.0

    assert np.allclose(eng.export_state(), expected)
    assert np.isclose(np.linalg.norm(eng.export_state()), 1.0)


def test_collapse_returns_index_and_projects_without_mutating_prior_buffer():
    eng = NumpySVSimulator()
    eng.initialize((2, 2))
    eng.state = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
    prior = eng.state  # collapse must not mutate this buffer in place
    original = prior.copy()
    rng = np.random.default_rng(1)

    idx = eng.collapse([1], rng)

    bit = (idx >> 1) & 1
    expected = np.zeros(4, dtype=complex)
    expected[bit << 1] = 1 / np.sqrt(2)
    expected[(bit << 1) | 1] = 1 / np.sqrt(2)

    assert np.allclose(prior, original)  # prior buffer left untouched
    assert np.allclose(eng.export_state(), expected)
    assert np.isclose(np.linalg.norm(eng.export_state()), 1.0)


def test_measure_qutrit_digit_extraction():
    from fatqat.implementation.matrices import shift_matrix

    eng = NumpySVSimulator()
    eng.initialize((3,))
    eng._state = shift_matrix(3, 2) @ eng.export_state()
    (digit,) = eng.measure_subsystems((0,), np.random.default_rng(0))
    assert digit == 2
