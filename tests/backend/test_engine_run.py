import numpy as np
import pytest

from fatqat.backends.engine_contract import _ResultRequest
from fatqat.backends.statevectorengine import StateVectorEngine
from fatqat.backends.steps import ApplyMatrixStep, MeasurementStep


def test_engine_run_requires_initialize():
    engine = StateVectorEngine()
    with pytest.raises(RuntimeError, match="engine not initialized"):
        engine.run(
            [], shots=1, seed=0, request=_ResultRequest(counts=False, statevector=False)
        )


def test_engine_run_fast_counts_returns_arrays():
    engine = StateVectorEngine()
    engine.initialize((2,), n_clbits=1)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    result = engine.run(
        [ApplyMatrixStep(matrix=x, target_indices=(0,)), MeasurementStep((0,), (0,))],
        shots=4,
        seed=0,
        request=_ResultRequest(counts=True, statevector=False),
    )
    assert result.state is None
    assert result.outcome_keys.shape == (1, 1)
    assert result.outcome_counts.shape == (1,)
    assert result.outcome_keys.tolist() == [[1]]
    assert result.outcome_counts.tolist() == [4]


def test_engine_run_fast_state_copies_only_when_requested():
    engine = StateVectorEngine()
    engine.initialize((2,), n_clbits=0)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    without_state = engine.run(
        [ApplyMatrixStep(matrix=x, target_indices=(0,))],
        shots=1,
        seed=0,
        request=_ResultRequest(counts=False, statevector=False),
    )
    assert without_state.state is None

    engine.initialize((2,), n_clbits=0)
    with_state = engine.run(
        [ApplyMatrixStep(matrix=x, target_indices=(0,))],
        shots=1,
        seed=0,
        request=_ResultRequest(counts=False, statevector=True),
    )
    assert np.allclose(with_state.state, [0, 1])


def test_engine_run_fast_counts_and_state_share_collapse_event():
    engine = StateVectorEngine()
    engine.initialize((2,), n_clbits=1)
    h = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
    result = engine.run(
        [ApplyMatrixStep(matrix=h, target_indices=(0,)), MeasurementStep((0,), (0,))],
        shots=1,
        seed=2026,
        request=_ResultRequest(counts=True, statevector=True),
    )
    measured = int(result.outcome_keys[0, 0])
    assert result.outcome_counts.tolist() == [1]
    assert np.isclose(abs(result.state[measured]), 1.0)
    assert np.count_nonzero(np.abs(result.state) > 1e-12) == 1
