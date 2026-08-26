"""Runtime selection on Simulator: dispatch, errors, and equivalence."""

import numpy as np
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


# --- explicit serial and Numba-thread execution ---


def _dynamic_program():
    # A reset selects the compiled multi-shot path and its shot-parallel loop.
    program = fq.Program(1, 1)
    program.add(fq.ops.H, 0)
    program.measure(0, 0)
    program.add(fq.ops.Reset, 0)
    return program


def test_serial_and_threaded_compiled_shots_are_identical():
    pytest.importorskip("numba")

    def counts_for(shot_parallelism, max_workers):
        return (
            Simulator(runtime="numba")
            .run(
                _dynamic_program(),
                shots=64,
                simulation_config={
                    "seed": 7,
                    "shot_parallelism": shot_parallelism,
                    "kernel_parallelism": "serial",
                    "max_workers": max_workers,
                },
            )
            .result()
            .get_counts()
        )

    assert counts_for("serial", 1) == counts_for("threads", 2)


def test_serial_shots_with_threaded_kernels_match_serial_kernels():
    pytest.importorskip("numba")
    from fatqat.simulator import AtomArraySimulator
    from fatqat.simulator._engine import nb

    if nb._MAX_THREADS < 2:
        pytest.skip("Numba exposes no parallel thread capacity")

    program = fq.Program(2, 2)
    program.add(fq.ops.Put, (0, 1))
    program.add(fq.ops.RX(np.pi / 3), 0)
    program.measure((0, 1), (0, 1))
    backend = AtomArraySimulator(num_sites=2, runtime="numba")

    def run(kernel_parallelism, max_workers):
        return backend.run(
            program,
            shots=1,
            simulation_config={
                "seed": 11,
                "shot_parallelism": "serial",
                "kernel_parallelism": kernel_parallelism,
                "max_workers": max_workers,
            },
            result_config={"counts": True, "final_state": True},
        ).result()

    serial = run("serial", 1)
    threaded = run("threads", 2)
    assert threaded.get_counts() == serial.get_counts()
    assert np.array_equal(threaded.get_statevector(), serial.get_statevector())


@pytest.mark.parametrize("fail", [False, True])
def test_numba_kernel_thread_scope_expands_and_restores(fail, monkeypatch):
    numba = pytest.importorskip("numba")
    from fatqat.simulator._engine import nb

    if nb._MAX_THREADS < 2:
        pytest.skip("Numba exposes no parallel thread capacity")

    original_limit = numba.get_num_threads()
    expected = 2
    numba.set_num_threads(1)
    before = numba.get_num_threads()
    original = nb.NumbaSVEngine._run_shot_seed_batch

    def probe(self, *args, **kwargs):
        assert numba.get_num_threads() == expected
        if fail:
            raise RuntimeError("numeric failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(nb.NumbaSVEngine, "_run_shot_seed_batch", probe)
    simulation_config = {
        "seed": 7,
        "shot_parallelism": "serial",
        "kernel_parallelism": "threads",
        "max_workers": expected,
    }

    try:
        job = Simulator(runtime="numba").run(
            _dynamic_program(),
            shots=8,
            simulation_config=simulation_config,
        )

        if fail:
            with pytest.raises(RuntimeError, match="numeric failure"):
                job.result()
        else:
            job.result()

        assert numba.get_num_threads() == before
    finally:
        numba.set_num_threads(original_limit)


def test_explicit_threads_accepts_an_empty_plan():
    pytest.importorskip("numba")
    from fatqat.simulator._engine import nb

    state = (
        Simulator(runtime="numba")
        .run(
            fq.Program(1),
            simulation_config={
                "kernel_parallelism": "threads",
                "max_workers": nb._MAX_THREADS + 1,
            },
        )
        .result()
        .get_statevector()
    )

    assert state.tolist() == [(1 + 0j), 0j]


def test_metadata_echoes_only_the_requested_public_configuration():
    result = (
        Simulator()
        .run(_bell_program(), shots=4, simulation_config={"seed": 1})
        .result()
    )
    assert result.metadata["simulation_config"] == {
        "seed": 1,
        "shot_parallelism": "auto",
        "kernel_parallelism": "auto",
        "max_workers": None,
        "fusion": False,
    }
    assert "execution" not in result.metadata
