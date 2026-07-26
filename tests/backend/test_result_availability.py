"""Tests statevector result availability and measurement-related backend behavior."""

import warnings

import numpy as np
import pytest

import fatqat as fq
from fatqat.backends import SimulatorBackend
from fatqat.errors import BackendValidationError, NoMeasurementWarning
from fatqat import operations as ops
from fatqat.program import Program


def test_statevector_default_attached_when_no_measurement():
    p = Program(1)
    p.add(ops.H, 0)
    job = SimulatorBackend("SV").run(p, result_config={"counts": False})
    sv = job.result().get_statevector()
    assert np.allclose(sv, np.array([1, 1]) / np.sqrt(2))


def test_statevector_not_attached_by_default_with_measurement():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    result = (
        SimulatorBackend("SV").run(p, shots=10, simulation_config={"seed": 0}).result()
    )
    with pytest.raises(fq.errors.ResultFieldUnavailableError):
        result.get_statevector()


def test_result_metadata_records_backend_shots_and_config():
    p = Program(1, 1)
    p.add(ops.X, 0)
    p.add_measurement(0, 0)
    config = {"counts": True, "final_state": False}

    result = (
        SimulatorBackend("SV")
        .run(p, shots=7, simulation_config={"seed": 0}, result_config=config)
        .result()
    )

    assert result.metadata["shots"] == 7
    assert result.metadata["backend_name"] == "SimulatorBackend"
    assert result.metadata["result_config"] == config


def test_run_accepts_result_config_as_dict():
    p = Program(1, 1)
    p.add(ops.X, 0)
    p.add_measurement(0, 0)

    result = (
        SimulatorBackend("SV")
        .run(
            p,
            shots=7,
            simulation_config={"seed": 0},
            result_config={"counts": True, "final_state": False},
        )
        .result()
    )

    assert result.get_counts() == {"1": 7}
    assert result.metadata["result_config"] == {"counts": True, "final_state": False}


def test_run_rejects_unknown_result_config_keys():
    p = Program(1)
    p.add(ops.H, 0)

    with pytest.raises(BackendValidationError, match="does not support result_config"):
        SimulatorBackend("SV").run(p, result_config={"counts": False, "gpu": True})


def test_run_rejects_non_dict_result_config():
    p = Program(1)
    p.add(ops.H, 0)

    with pytest.raises(TypeError, match="dict or None"):
        SimulatorBackend("SV").run(p, result_config=object())


def test_projected_statevector_shots_one():
    p = Program(1, 1)
    p.add(ops.H, 0)
    p.add_measurement(0, 0)
    sv = (
        SimulatorBackend("SV")
        .run(
            p,
            shots=1,
            simulation_config={"seed": 0},
            result_config={"counts": True, "final_state": True},
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
        SimulatorBackend("SV").run(
            p, shots=10, result_config={"counts": True, "final_state": True}
        )


def test_no_measurement_warning_when_counts_only_and_no_state():
    p = Program(1, 1)  # has a clbit, never measured
    p.add(ops.H, 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SimulatorBackend("SV").run(
            p,
            shots=10,
            simulation_config={"seed": 0},
            result_config={"counts": True, "final_state": False},
        ).result()
    assert any(issubclass(w.category, NoMeasurementWarning) for w in caught)


def test_dim2_gate_on_qutrit_raises_at_lowering_not_frontend():
    qt = fq.QuantumRegister(1, dim=3)
    program = fq.Program([qt])
    program.add(fq.ops.H, qt[0])  # frontend must NOT raise here
    with pytest.raises(BackendValidationError) as exc:
        fq.backends.SimulatorBackend("SV").run(
            program, result_config={"counts": False, "final_state": True}
        ).result()
    msg = str(exc.value)
    assert "H" in msg and "3" in msg  # names the op and the target dimension


def test_dim2_gate_frontend_add_does_not_raise():
    qt = fq.QuantumRegister(1, dim=3)
    program = fq.Program([qt])
    program.add(fq.ops.X, qt[0])  # no exception at add-time
