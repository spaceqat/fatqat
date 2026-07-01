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
    job = StateVectorBackend().run(p, result_config=qs.ResultConfig(counts=False))
    sv = job.result().get_statevector()
    assert np.allclose(sv, np.array([1, 1]) / np.sqrt(2))


def test_statevector_not_attached_by_default_with_measurement():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    result = StateVectorBackend(seed=0).run(p, shots=10).result()
    with pytest.raises(qs.ResultFieldUnavailableError):
        result.get_statevector()


def test_projected_statevector_shots_one():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    sv = (
        StateVectorBackend(seed=0)
        .run(p, shots=1, result_config=qs.ResultConfig(counts=True, statevector=True))
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
        StateVectorBackend().run(
            p, shots=10, result_config=qs.ResultConfig(counts=True, statevector=True)
        )


def test_no_measurement_warning_when_counts_only_and_no_state():
    p = Program(1, 1)  # has a clbit, never measured
    p.add(ops.H, 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        StateVectorBackend(seed=0).run(
            p, shots=10, result_config=qs.ResultConfig(counts=True, statevector=False)
        ).result()
    assert any(issubclass(w.category, NoMeasurementWarning) for w in caught)
