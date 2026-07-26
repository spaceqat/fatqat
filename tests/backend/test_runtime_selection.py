"""Runtime selection on SimulatorBackend: dispatch, errors, and equivalence."""

import pytest

import fatqat as fq
from fatqat.backends import SimulatorBackend
from fatqat.errors import BackendValidationError
from fatqat.simulator.np import NumpyDMSimulator, NumpySVSimulator


def _bell_program():
    program = fq.Program(2, 2)
    program.add(fq.ops.H, 0)
    program.add(fq.ops.CX, (0, 1))
    program.add_measurement((0, 1), (0, 1))
    return program


def test_default_runtime_is_numpy():
    assert type(SimulatorBackend()._simulator) is NumpySVSimulator
    assert type(SimulatorBackend(method="DM")._simulator) is NumpyDMSimulator


def test_numba_runtime_selects_the_numba_simulator():
    pytest.importorskip("numba")
    from fatqat.simulator.nb import NumbaSVSimulator

    backend = SimulatorBackend(method="SV", runtime="numba")
    assert type(backend._simulator) is NumbaSVSimulator
    # Case-insensitive, like method.
    backend = SimulatorBackend(runtime="NUMBA")
    assert type(backend._simulator) is NumbaSVSimulator


def test_unknown_runtime_rejected_at_construction():
    with pytest.raises(BackendValidationError, match="runtime"):
        SimulatorBackend(runtime="jax")


def test_density_matrix_numba_selects_the_numba_dm_simulator():
    pytest.importorskip("numba")
    from fatqat.simulator.nb import NumbaDMSimulator

    backend = SimulatorBackend(method="DM", runtime="numba")
    assert type(backend._simulator) is NumbaDMSimulator


def test_metadata_echoes_the_runtime():
    result = (
        SimulatorBackend()
        .run(_bell_program(), shots=4, simulation_config={"seed": 1})
        .result()
    )
    assert result.metadata["runtime"] == "numpy"


def test_numba_runtime_produces_valid_bell_counts_through_the_portal():
    pytest.importorskip("numba")
    counts = (
        SimulatorBackend(runtime="numba")
        .run(_bell_program(), shots=200, simulation_config={"seed": 5})
        .result()
        .get_counts()
    )
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 200
