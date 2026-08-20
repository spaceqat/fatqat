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


def test_multiprocessing_pool_is_created_with_the_thread_limit_applied():
    """Pin the wiring, not the helper.

    The helper on its own proves nothing: deleting the `with` around the pool
    leaves a test that only calls the helper directly still green, which is
    exactly the regression it exists to prevent. So this intercepts the
    executor and asserts on the environment it was actually constructed in -
    the moment that decides what a spawned worker inherits.
    """
    from fatqat.simulator._engine import parallel

    seen = {}

    class _RecordingExecutor:
        def __init__(self, max_workers=None, **kwargs):
            seen["env"] = dict(os.environ)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, function, *iterables):
            # Run in-process; the point is what the environment looked like.
            return [function(*arguments) for arguments in zip(*iterables)]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(parallel, "ProcessPoolExecutor", _RecordingExecutor)
        Simulator("SV", runtime="numpy").run(
            _random_dynamic_program(),
            shots=40,
            simulation_config={
                "seed": 5,
                "max_workers": 2,
                "parallel_mode": "multiprocessing",
            },
        ).result().get_counts()

    assert all(seen["env"][name] == "1" for name in parallel._WORKER_THREAD_VARS)


def test_loky_executor_is_given_the_thread_limit_as_env():
    """The loky path takes the same limit through its own `env=`.

    loky applies it to workers itself and restarts a reusable pool when it
    changes, so it needs no environment mutation - but only if it is actually
    passed, which is what this checks.
    """
    loky = pytest.importorskip("loky")
    from fatqat.simulator._engine import parallel

    seen = {}

    class _RecordingExecutor:
        def map(self, function, *iterables):
            return [function(*arguments) for arguments in zip(*iterables)]

    def _fake_get_reusable_executor(max_workers=None, env=None, **kwargs):
        seen["env"] = env
        return _RecordingExecutor()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(loky, "get_reusable_executor", _fake_get_reusable_executor)
        Simulator("SV", runtime="numpy").run(
            _random_dynamic_program(),
            shots=40,
            simulation_config={"seed": 5, "max_workers": 2, "parallel_mode": "loky"},
        ).result().get_counts()

    assert seen["env"] == {name: "1" for name in parallel._WORKER_THREAD_VARS}


def test_the_parent_environment_is_restored_even_on_failure():
    """A parallel run must not leave the caller's process reconfigured."""
    from fatqat.simulator._engine.parallel import (
        _WORKER_THREAD_VARS,
        _single_threaded_workers,
    )

    before = {name: os.environ.get(name) for name in _WORKER_THREAD_VARS}

    with pytest.raises(RuntimeError):
        with _single_threaded_workers():
            assert all(os.environ[name] == "1" for name in _WORKER_THREAD_VARS)
            raise RuntimeError("boom")

    assert {name: os.environ.get(name) for name in _WORKER_THREAD_VARS} == before
