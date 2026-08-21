"""Runtime selection on Simulator: dispatch, errors, and equivalence."""

import pytest

import fatqat as fq
from fatqat.simulator import Simulator
from fatqat.errors import BackendValidationError


def _bell_program():
    program = fq.Program(2, 2)
    program.add(fq.ops.H, 0)
    program.add(fq.ops.CX, (0, 1))
    program.measure((0, 1), (0, 1))
    return program


@pytest.mark.parametrize(
    ("method", "engine_name"),
    [
        ("statevector", "NumbaSVEngine"),
        ("density_matrix", "NumbaDMEngine"),
        ("unitary", "NumbaUnitaryEngine"),
        ("superop", "NumbaSuperopEngine"),
    ],
)
def test_default_runtime_is_numba_for_every_method(method, engine_name):
    from fatqat.simulator._engine import nb

    assert type(Simulator(method=method)._engine) is getattr(nb, engine_name)


@pytest.mark.parametrize(
    "backend_cls",
    [
        fq.simulator.SCQubitIBMSimulator,
        fq.simulator.SCQubitGoogleSimulator,
    ],
)
def test_fake_simulator_defaults_to_numba(backend_cls):
    from fatqat.simulator._engine.nb import NumbaSVEngine

    assert type(backend_cls()._engine) is NumbaSVEngine


def test_numba_runtime_selects_the_numba_engine():
    pytest.importorskip("numba")
    from fatqat.simulator._engine.nb import NumbaSVEngine

    backend = Simulator(method="SV", runtime="numba")
    assert type(backend._engine) is NumbaSVEngine
    # Case-insensitive, like method.
    backend = Simulator(runtime="NUMBA")
    assert type(backend._engine) is NumbaSVEngine


def test_unknown_runtime_rejected_at_construction():
    with pytest.raises(BackendValidationError, match="runtime"):
        Simulator(runtime="jax")


def test_density_matrix_numba_selects_the_numba_dm_engine():
    pytest.importorskip("numba")
    from fatqat.simulator._engine.nb import NumbaDMEngine

    backend = Simulator(method="DM", runtime="numba")
    assert type(backend._engine) is NumbaDMEngine


def test_metadata_echoes_the_runtime():
    result = (
        Simulator()
        .run(_bell_program(), shots=4, simulation_config={"seed": 1})
        .result()
    )
    assert result.metadata["runtime"] == "numba"


def test_numba_runtime_produces_valid_bell_counts_through_the_portal():
    pytest.importorskip("numba")
    counts = (
        Simulator(runtime="numba")
        .run(_bell_program(), shots=200, simulation_config={"seed": 5})
        .result()
        .get_counts()
    )
    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 200


# --- numba_parallel: in-process thread parallelism, on or off ---


def _dynamic_program():
    # A reset forces the dynamic path, where the fused kernel's `prange` shot
    # loop is what `numba_parallel` switches off.
    program = fq.Program(1, 1)
    program.add(fq.ops.H, 0)
    program.measure(0, 0)
    program.add(fq.ops.Reset, 0)
    return program


def test_numba_parallel_off_produces_identical_counts():
    pytest.importorskip("numba")

    def counts_for(numba_parallel):
        return (
            Simulator(runtime="numba")
            .run(
                _dynamic_program(),
                shots=64,
                simulation_config={"seed": 7, "numba_parallel": numba_parallel},
            )
            .result()
            .get_counts()
        )

    assert counts_for(False) == counts_for(True)


def test_numba_parallel_off_restores_the_thread_count():
    # `set_num_threads` is process-wide, so the run must put it back.
    numba = pytest.importorskip("numba")
    before = numba.get_num_threads()

    Simulator(runtime="numba").run(
        _dynamic_program(),
        shots=64,
        simulation_config={"seed": 7, "numba_parallel": False},
    ).result()

    assert numba.get_num_threads() == before


@pytest.mark.parametrize("method", ["SV", "DM"])
@pytest.mark.parametrize("numba_parallel", [True, False])
def test_numba_parallel_confines_the_pool_for_the_whole_run(
    method, numba_parallel, monkeypatch
):
    # Probed from inside the simulator's own run, where every kernel the run
    # touches sees the pool - the fused shot loop and the gate-level coset
    # chunks alike, on either representation.
    numba = pytest.importorskip("numba")
    from fatqat.simulator._engine.np import _NumpyMatrixEngine

    observed = []
    original = _NumpyMatrixEngine._analyze_plan

    def probe(self, plan):
        observed.append(numba.get_num_threads())
        return original(self, plan)

    monkeypatch.setattr(_NumpyMatrixEngine, "_analyze_plan", probe)
    Simulator(method=method, runtime="numba").run(
        _dynamic_program(),
        shots=8,
        simulation_config={"seed": 7, "numba_parallel": numba_parallel},
    ).result()

    expected = numba.get_num_threads() if numba_parallel else 1
    assert observed == [expected]


def test_numpy_runtime_rejects_numba_parallel():
    with pytest.raises(BackendValidationError, match="numba_parallel"):
        Simulator(runtime="numpy").run(
            _bell_program(),
            shots=4,
            simulation_config={"numba_parallel": False},
        )
    # The default value is not a request, so it stays accepted everywhere.
    Simulator(runtime="numpy").run(
        _bell_program(), shots=4, simulation_config={"numba_parallel": True}
    ).result()


def test_numba_parallel_must_be_a_bool():
    with pytest.raises(BackendValidationError, match="numba_parallel"):
        Simulator(runtime="numpy").run(
            _bell_program(), shots=4, simulation_config={"numba_parallel": 0}
        )


def test_metadata_echoes_numba_parallel():
    result = (
        Simulator()
        .run(_bell_program(), shots=4, simulation_config={"seed": 1})
        .result()
    )
    assert result.metadata["simulation_config"]["numba_parallel"] is True
