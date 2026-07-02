"""Tests statevector result availability and measurement-related backend behavior."""

import warnings

import numpy as np
import pytest

import qnsim as qs
from qnsim.backends import StateVectorBackend
from qnsim.errors import BackendValidationError, NoMeasurementWarning
from qnsim import operations as ops
from qnsim.program import Program


def test_statevector_default_attached_when_no_measurement():
    p = Program(1)
    p.add(ops.H, 0)
    job = StateVectorBackend().run(p, result_config={"counts": False})
    sv = job.result().get_statevector()
    assert np.allclose(sv, np.array([1, 1]) / np.sqrt(2))


def test_statevector_not_attached_by_default_with_measurement():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    result = StateVectorBackend().run(p, shots=10, seed=0).result()
    with pytest.raises(qs.ResultFieldUnavailableError):
        result.get_statevector()


def test_result_metadata_records_backend_shots_and_config():
    p = Program(1, 1)
    p.add(ops.X, 0)
    p.add_measurement(0, 0)
    config = {"counts": True, "statevector": False}

    result = StateVectorBackend().run(
        p, shots=7, seed=0, result_config=config
    ).result()

    assert result.metadata["shots"] == 7
    assert result.metadata["backend_name"] == "StateVectorBackend"
    assert result.metadata["result_config"] == config


def test_run_accepts_result_config_as_dict():
    p = Program(1, 1)
    p.add(ops.X, 0)
    p.add_measurement(0, 0)

    result = StateVectorBackend().run(
        p,
        shots=7,
        seed=0,
        result_config={"counts": True, "statevector": False},
    ).result()

    assert result.get_counts() == {"1": 7}
    assert result.metadata["result_config"] == {"counts": True, "statevector": False}


def test_run_warns_and_ignores_unknown_result_config_keys():
    p = Program(1)
    p.add(ops.H, 0)

    with pytest.warns(UserWarning, match="ignored unsupported result_config options"):
        result = StateVectorBackend().run(
            p,
            result_config={"counts": False, "gpu": True},
        ).result()

    assert result.metadata["result_config"] == {
        "counts": False,
        "statevector": None,
    }


def test_run_rejects_non_dict_result_config():
    p = Program(1)
    p.add(ops.H, 0)

    with pytest.raises(TypeError, match="dict or None"):
        StateVectorBackend().run(p, result_config=object())


def test_projected_statevector_shots_one():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    sv = (
        StateVectorBackend()
        .run(
            p,
            shots=1,
            seed=0,
            result_config={"counts": True, "statevector": True},
        )
        .result()
        .get_statevector()
    )
    # collapsed to a basis state
    assert np.isclose(np.linalg.norm(sv), 1.0)
    assert np.count_nonzero(np.round(np.abs(sv), 6)) == 1


def test_statevector_with_measurement_and_many_shots_rejected():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    with pytest.raises(BackendValidationError):
        StateVectorBackend().run(p, shots=10, result_config={"counts": True, "statevector": True})


def test_no_measurement_warning_when_counts_only_and_no_state():
    p = Program(1, 1)  # has a clbit, never measured
    p.add(ops.H, 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        StateVectorBackend().run(
            p,
            shots=10,
            seed=0,
            result_config={"counts": True, "statevector": False},
        ).result()
    assert any(issubclass(w.category, NoMeasurementWarning) for w in caught)
