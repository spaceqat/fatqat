import numpy as np

from qnsim.engine import StateVectorEngine
from qnsim.implementation import ApplyMatrixStep

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def test_measure_qubit_deterministic_one():
    eng = StateVectorEngine()
    eng.initialize(1)
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))  # |1>
    bit = eng.measure_qubit(0, np.random.default_rng(0))
    assert bit == 1
    assert np.allclose(eng.export_state(), np.array([0, 1], dtype=complex))


def test_measure_qubit_collapses_and_is_repeatable():
    eng = StateVectorEngine()
    eng.initialize(1)
    eng.apply(ApplyMatrixStep(matrix=_H, target_indices=(0,)))  # (|0>+|1>)/sqrt2
    first = eng.measure_qubit(0, np.random.default_rng(0))
    # after collapse the state is a basis state; re-measuring returns the same bit
    second = eng.measure_qubit(0, np.random.default_rng(123))
    assert first == second


def test_reset_qubit_from_one_returns_zero():
    eng = StateVectorEngine()
    eng.initialize(1)
    eng.apply(ApplyMatrixStep(matrix=_X, target_indices=(0,)))  # |1>
    eng.reset_qubit(0, np.random.default_rng(0))
    assert np.allclose(eng.export_state(), np.array([1, 0], dtype=complex))


def test_reset_qubit_on_entangled_pair_conditions_the_partner():
    # Bell pair (|00>+|11>)/sqrt2; reset qubit 0 -> partner is |0> or |1>, 50/50.
    outcomes = []
    for s in range(200):
        eng = StateVectorEngine()
        eng.initialize(2)
        eng.apply(ApplyMatrixStep(matrix=_H, target_indices=(0,)))
        cx = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            dtype=complex,
        )
        eng.apply(ApplyMatrixStep(matrix=cx, target_indices=(0, 1)))
        eng.reset_qubit(0, np.random.default_rng(s))
        st = eng.export_state()
        # qubit 0 must be |0>: only indices with bit0 == 0 are allowed.
        # Amplitude is on index 0 (partner 0) or 2 (partner 1).
        nz = np.flatnonzero(np.round(np.abs(st), 6))
        assert nz.size == 1 and nz[0] in (0, 2)
        outcomes.append(int(nz[0] == 2))  # partner==1
    frac = sum(outcomes) / len(outcomes)
    assert 0.35 < frac < 0.65
