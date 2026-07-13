import numpy as np
import pytest

from fatqat.backends.engine_contract import _EngineConfig, _ResultRequest
from fatqat.backends.numpy_engine import NumpyEngine
from fatqat.backends.steps import ApplyMatrixStep, MeasurementStep, ResetStep


def test_engine_run_requires_initialize():
    engine = NumpyEngine(state_semantics="statevector")
    with pytest.raises(RuntimeError, match="engine not initialized"):
        engine.run(
            [], shots=1, seed=0, request=_ResultRequest(counts=False, statevector=False)
        )


def test_engine_run_fast_counts_returns_arrays():
    engine = NumpyEngine(state_semantics="statevector")
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
    engine = NumpyEngine(state_semantics="statevector")
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
    engine = NumpyEngine(state_semantics="statevector")
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


def test_engine_run_dynamic_counts_use_clbit_snapshots():
    engine = NumpyEngine(_EngineConfig(max_workers=1), state_semantics="statevector")
    engine.initialize((2,), n_clbits=1)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    result = engine.run(
        [
            ApplyMatrixStep(matrix=x, target_indices=(0,)),
            MeasurementStep((0,), (0,)),
            ResetStep((0,)),
        ],
        shots=5,
        seed=0,
        request=_ResultRequest(counts=True, statevector=False),
    )
    assert result.outcome_keys.tolist() == [[1]]
    assert result.outcome_counts.tolist() == [5]


def test_engine_run_dynamic_statevector_only_runs_one_trajectory():
    engine = NumpyEngine(state_semantics="statevector")
    engine.initialize((2, 2), n_clbits=2)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    result = engine.run(
        [ApplyMatrixStep(matrix=x, target_indices=(1,), condition=((0, 0),))],
        shots=0,
        seed=0,
        request=_ResultRequest(counts=False, statevector=True),
    )
    assert result.outcome_keys is None
    assert result.outcome_counts is None
    assert np.allclose(result.state, [0, 0, 1, 0])


def test_engine_run_dynamic_reinitializes_each_shot():
    engine = NumpyEngine(_EngineConfig(max_workers=1), state_semantics="statevector")
    engine.initialize((2,), n_clbits=1)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    result = engine.run(
        [
            ApplyMatrixStep(matrix=x, target_indices=(0,)),
            ResetStep((0,)),
            MeasurementStep((0,), (0,)),
        ],
        shots=4,
        seed=123,
        request=_ResultRequest(counts=True, statevector=False),
    )
    assert result.outcome_keys.tolist() == [[0]]
    assert result.outcome_counts.tolist() == [4]
