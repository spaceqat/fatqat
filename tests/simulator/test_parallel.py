import os

import pytest

import fatqat as fq
from fatqat import operations as ops
from fatqat.simulator import Simulator


def _random_dynamic_program():
    p = fq.Program(2, 2)
    p.add(ops.H, 0)
    p.add(ops.CX, (0, 1))
    p.measure((0, 1), (0, 1))
    p.add(fq.ops.Reset, (0, 1))
    return p


@pytest.mark.parametrize(
    "parallel_mode,seed",
    [
        ("multiprocessing", 2026),
        ("loky", 7),
        ("auto", 99),
    ],
)
def test_parallel_counts_match_serial_for_same_seed(parallel_mode, seed):
    p = _random_dynamic_program()

    serial = (
        Simulator("SV")
        .run(p, shots=40, simulation_config={"seed": seed, "max_workers": 1})
        .result()
        .get_counts()
    )
    parallel = (
        Simulator("SV")
        .run(
            p,
            shots=40,
            simulation_config={
                "seed": seed,
                "max_workers": 2,
                "parallel_mode": parallel_mode,
            },
        )
        .result()
        .get_counts()
    )

    assert parallel == serial


def test_parallel_mode_serial_wins_over_max_workers():
    p = _random_dynamic_program()

    counts = (
        Simulator("SV")
        .run(
            p,
            shots=12,
            simulation_config={"seed": 11, "max_workers": 2, "parallel_mode": "serial"},
        )
        .result()
        .get_counts()
    )

    assert sum(counts.values()) == 12


def test_unknown_parallel_mode_rejected_at_run():
    # Per-run simulation configuration is validated before execution, rather
    # than being swallowed into a failed Job.
    with pytest.raises(
        fq.errors.BackendValidationError, match="unsupported parallel_mode"
    ):
        Simulator("SV").run(
            _random_dynamic_program(),
            simulation_config={"max_workers": 2, "parallel_mode": "not-a-mode"},
        )


def test_workers_are_asked_to_start_single_threaded():
    # A worker runs independent shots on small local matrices, so a BLAS thread
    # pool inside it can only oversubscribe - the parallelism is already spent
    # on processes. On a many-core host the reservations for those unused
    # threads are what breaks the pool, so this pins that workers are told to
    # start with one.
    from fatqat.simulator._engine.parallel import (
        _WORKER_THREAD_VARS,
        _single_threaded_workers,
    )

    before = {name: os.environ.get(name) for name in _WORKER_THREAD_VARS}

    with _single_threaded_workers():
        assert all(os.environ[name] == "1" for name in _WORKER_THREAD_VARS)

    # And the parent is handed back exactly what it had, set or unset.
    assert {name: os.environ.get(name) for name in _WORKER_THREAD_VARS} == before


def test_worker_thread_limit_survives_a_failure():
    from fatqat.simulator._engine.parallel import (
        _WORKER_THREAD_VARS,
        _single_threaded_workers,
    )

    before = {name: os.environ.get(name) for name in _WORKER_THREAD_VARS}

    with pytest.raises(RuntimeError):
        with _single_threaded_workers():
            raise RuntimeError("boom")

    assert {name: os.environ.get(name) for name in _WORKER_THREAD_VARS} == before
