"""Fusion controls on the Numba runtime."""

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
    assert result.result().metadata["simulation_config"]["fusion"] is False


def test_numpy_runtime_accepts_fusion_off():
    result = (
        Simulator(method="DM", runtime="numpy")
        .run(_noisy_program(), simulation_config={"fusion": False})
        .result()
    )
    assert result.metadata["simulation_config"]["fusion"] is False


@pytest.mark.parametrize("method", ["SV", "DM"])
def test_numba_state_evolution_accepts_enabling_fusion(method):
    result = (
        Simulator(method=method, runtime="numba")
        .run(
            _noisy_program(),
            shots=0,
            result_config=_STATE_ONLY,
            simulation_config={"fusion": True},
        )
        .result()
    )
    assert result.metadata["simulation_config"]["fusion"] is True


def test_numba_fixed_photon_engine_declares_no_fusion():
    from fatqat.simulator._engine.engines.numba_engine import NumbaFixedPhotonEngine

    kernels = NumbaFixedPhotonEngine._capabilities.kernels

    assert kernels.gate_fusion is None
    assert kernels.supports_fusion is False


@pytest.mark.parametrize("bad", ["yes", 1, 0])
def test_a_non_boolean_is_rejected(bad):
    with pytest.raises(BackendValidationError, match="fusion must be a bool"):
        Simulator(method="DM", runtime="numba").run(
            _noisy_program(), simulation_config={"fusion": bad}
        )
