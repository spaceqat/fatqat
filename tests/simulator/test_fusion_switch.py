"""Fusion controls and unsupported state-evolution routes."""

import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.errors import BackendValidationError
from fatqat.simulator import Simulator

pytest.importorskip("numba")

_STATE_ONLY = {"counts": False, "final_state": True}


def _noisy_program():
    """A noisy circuit shared by configuration and numerical checks."""
    program = fq.Program(3)
    for qubit in range(3):
        program.add(ops.H, qubit)
    for qubit in range(2):
        program.add(ops.CX, (qubit, qubit + 1))
    return program


def test_fusion_is_off_by_default():
    result = Simulator(method="DM", runtime="numba").run(
        _noisy_program(), shots=0, result_config=_STATE_ONLY
    )
    assert result.result().metadata["execution_config"]["fusion"] is False


def test_numpy_runtime_rejects_enabling_fusion():
    with pytest.raises(BackendValidationError, match="not supported.*matrix engine"):
        Simulator(method="DM", runtime="numpy").run(
            _noisy_program(), execution_config={"fusion": True}
        )


def test_numpy_runtime_accepts_fusion_off():
    result = (
        Simulator(method="DM", runtime="numpy")
        .run(_noisy_program(), execution_config={"fusion": False})
        .result()
    )
    assert result.metadata["execution_config"]["fusion"] is False


@pytest.mark.parametrize("method", ["SV", "DM"])
def test_numba_state_evolution_rejects_enabling_fusion(method):
    with pytest.raises(BackendValidationError, match="fusion=True is not supported"):
        Simulator(method=method, runtime="numba").run(
            _noisy_program(), execution_config={"fusion": True}
        )


@pytest.mark.parametrize("bad", ["yes", 1, 0])
def test_a_non_boolean_is_rejected(bad):
    with pytest.raises(BackendValidationError, match="fusion must be a bool"):
        Simulator(method="DM", runtime="numba").run(
            _noisy_program(), execution_config={"fusion": bad}
        )
