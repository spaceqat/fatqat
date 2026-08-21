"""Parallel dynamic-shot execution for the NumPy matrix engines.

Mirrors the serial per-shot path (`_NumpyMatrixEngine._run_one_shot`) across
worker processes when counts are requested for enough shots and options allow
it. Kept separate from `np.py` because it is pure execution-strategy plumbing
(worker counts, batching, process-pool dispatch) with no state/physics content:
each worker constructs the requested `MatrixEngine` subclass and runs the shared
per-shot loop. The subclass is passed as a plain class object, which pickles
trivially to workers, so this module has no compile-time dependency on any
concrete engine.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from itertools import repeat
from typing import TYPE_CHECKING

import numpy as np

from ..._backends.engine_contract import (
    _DensityMatrixResultRequest as DensityMatrixResultRequest,
    _EngineConfig as EngineConfig,
    _StateVectorResultRequest as StateVectorResultRequest,
)
from ..._backends.steps import ResolvedStep

if TYPE_CHECKING:
    from typing import Protocol

    class _DynamicSimulator(Protocol):
        """The slice of the `MatrixEngine` interface a per-shot worker touches."""

        def initialize(self, system_dims: Sequence[int], n_clbits: int = 0) -> None: ...

        initial_state: np.ndarray | None

        def _run_one_shot(
            self, plan: list[ResolvedStep], rng: np.random.Generator
        ) -> tuple[int, ...]: ...

    class _EngineFactory(Protocol):
        def __call__(self) -> _DynamicSimulator: ...


_ResultRequest = StateVectorResultRequest | DensityMatrixResultRequest


def _shot_seed_sequences(
    seed: int | None, n_iters: int
) -> list[np.random.SeedSequence]:
    """Spawn one independent child `SeedSequence` per sampled shot.

    Child streams are derived from a single root sequence in shot order, so
    serial and parallel execution draw from the same reproducible per-shot
    streams regardless of how shots are distributed across workers.
    """
    root = np.random.SeedSequence(seed)
    return root.spawn(n_iters)


def _loky_available() -> bool:
    return importlib.util.find_spec("loky") is not None


# Minimum shot count before automatic parallelism (max_workers=None) kicks in.
# Below this, process-pool startup and per-batch pickling cost more than the
# per-shot simulation they save, so automatic mode stays serial - keeping the
# default backend and small/interactive runs fast. An explicit max_workers > 1
# always parallelizes and bypasses this floor.
_PARALLEL_MIN_SHOTS = 32


def _effective_max_workers(max_workers: int | None, n_iters: int) -> int:
    if max_workers is None:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)
        return min(n_iters, cpu_count() or 1)
    return int(max_workers)


def _planned_workers(
    config: EngineConfig, request: _ResultRequest, n_iters: int
) -> int | None:
    """Resolve the worker count, or ``None`` to stay on the serial path."""
    if not request.counts:
        return None
    if n_iters <= 1:
        return None
    if config.parallel_mode == "serial":
        return None
    if config.max_workers is None and n_iters < _PARALLEL_MIN_SHOTS:
        # Automatic mode stays serial for small runs; explicit max_workers wins.
        return None
    workers = min(_effective_max_workers(config.max_workers, n_iters), n_iters)
    return workers if workers > 1 else None


def _resolve_parallel_mode_name(name: str) -> str:
    """Resolve ``"auto"`` to a concrete backend; validated names pass through."""
    if name == "auto":
        return "loky" if _loky_available() else "multiprocessing"
    return name


def _split_into_batches(
    seed_sequences: list[np.random.SeedSequence], n_batches: int
) -> list[list[np.random.SeedSequence]]:
    """Split shots into up to n_batches contiguous batches, one task per worker.

    Batching pickles the (matrix-carrying) plan once per worker instead of once
    per shot, and lets each worker reuse a single engine across its batch.
    Shot i keeps ``seed_sequences[i]`` regardless of batching, so aggregated
    counts are byte-for-byte identical to the serial path.
    """
    n_items = len(seed_sequences)
    n_batches = max(1, min(n_batches, n_items))
    base, extra = divmod(n_items, n_batches)
    batches: list[list[np.random.SeedSequence]] = []
    start = 0
    for i in range(n_batches):
        size = base + (1 if i < extra else 0)
        batches.append(seed_sequences[start : start + size])
        start += size
    return batches


def _run_dynamic_shot_batch(
    plan: list[ResolvedStep],
    system_dims: tuple[int, ...],
    n_clbits: int,
    seed_batch: list[np.random.SeedSequence],
    engine_cls: _EngineFactory,
    initial_state: np.ndarray | None = None,
) -> list[tuple[int, ...]]:
    """Run a contiguous batch of dynamic shots in one worker with one engine.

    A worker builds its own engine, so the run's initial state has to travel
    with the work: without it the worker would start every shot from the
    all-zero state while the serial path started somewhere else, and the two
    would disagree silently rather than fail.
    """
    engine = engine_cls()
    engine.initial_state = initial_state
    snapshots: list[tuple[int, ...]] = []
    for seed_sequence in seed_batch:
        engine.initialize(system_dims, n_clbits)
        snapshots.append(
            engine._run_one_shot(plan, np.random.default_rng(seed_sequence))
        )
    return snapshots


# Thread-count variables the common BLAS builds read when they load. A worker
# runs one batch of independent shots on small local matrices, so a BLAS thread
# pool inside it can only oversubscribe: the parallelism is already spent on
# processes.
_WORKER_THREAD_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@contextmanager
def _single_threaded_workers() -> Iterator[None]:
    """Make workers start their BLAS with one thread, then restore the parent.

    Under ``spawn`` a worker is a fresh interpreter that inherits this process's
    environment and imports NumPy afterwards, so setting these here - before the
    pool starts a worker - is what reaches them. Setting them from inside a
    worker would be too late: BLAS reads them when it loads, which has already
    happened by the time any initializer runs.

    Without this, each worker brings up a thread pool sized to the whole
    machine. On a 32-thread host, two workers plus the parent reserve buffers
    for 96 BLAS threads to run shots that use none of them, and OpenBLAS
    eventually fails an allocation and aborts the worker, which surfaces as
    ``BrokenProcessPool``.

    The parent is unaffected either way: its BLAS is already loaded, so it keeps
    the thread count it started with.

    This does mutate ``os.environ``, which is process-global, for as long as the
    pool is in use - so another thread spawning an unrelated subprocess during
    that window would inherit the limit too. Two things bound that. It applies
    only to ``parallel_mode="multiprocessing"``: ``"auto"`` resolves to loky
    whenever loky is importable, and loky is a base dependency, so the default
    path takes its own ``env=`` and never touches this process. And the window
    cannot be narrowed to pool construction, because a pool starts workers
    lazily as tasks are submitted - a worker spawned after the window closed
    would read the wrong value, which is the bug this exists to prevent.

    ``threadpoolctl`` does not replace this. It retunes thread pools already
    loaded in the *current* process, and the pool that matters here belongs to
    a process that has not started yet and will read its configuration during
    its own import. joblib and loky solve it the same way underneath - their
    ``inner_max_num_threads`` also arrives as environment - which is what
    ``env=`` on the loky path already uses.
    """
    previous = {name: os.environ.get(name) for name in _WORKER_THREAD_VARS}
    os.environ.update({name: "1" for name in _WORKER_THREAD_VARS})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_dynamic_shots_multiprocessing(
    plan: list[ResolvedStep],
    system_dims: tuple[int, ...],
    n_clbits: int,
    seed_sequences: list[np.random.SeedSequence],
    max_workers: int,
    engine_cls: _EngineFactory,
    initial_state: np.ndarray | None = None,
) -> list[tuple[int, ...]]:
    batches = _split_into_batches(seed_sequences, max_workers)
    # The scope has to cover the map, not just construction: the pool starts
    # workers lazily on first submit.
    with _single_threaded_workers():
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(
                _run_dynamic_shot_batch,
                repeat(plan),
                repeat(system_dims),
                repeat(n_clbits),
                batches,
                repeat(engine_cls),
                repeat(initial_state),
            )
            return [snapshot for batch in results for snapshot in batch]


def _run_dynamic_shots_loky(
    plan: list[ResolvedStep],
    system_dims: tuple[int, ...],
    n_clbits: int,
    seed_sequences: list[np.random.SeedSequence],
    max_workers: int,
    engine_cls: _EngineFactory,
    initial_state: np.ndarray | None = None,
) -> list[tuple[int, ...]]:
    from loky import get_reusable_executor

    batches = _split_into_batches(seed_sequences, max_workers)
    # loky applies `env` to its workers itself, and restarts the reusable pool
    # when it changes - the proper route for the same reason as above.
    executor = get_reusable_executor(
        max_workers=max_workers,
        env={name: "1" for name in _WORKER_THREAD_VARS},
    )
    results = executor.map(
        _run_dynamic_shot_batch,
        repeat(plan),
        repeat(system_dims),
        repeat(n_clbits),
        batches,
        repeat(engine_cls),
        repeat(initial_state),
    )
    return [snapshot for batch in results for snapshot in batch]


def _run_dynamic_shots_parallel(
    config: EngineConfig,
    plan: list[ResolvedStep],
    system_dims: tuple[int, ...],
    n_clbits: int,
    seed_sequences: list[np.random.SeedSequence],
    max_workers: int,
    engine_cls: _EngineFactory,
    initial_state: np.ndarray | None = None,
) -> list[tuple[int, ...]]:
    """Dispatch shots to worker processes. Caller must supply ``max_workers``.

    ``max_workers`` is the same value `_planned_workers` already returned, passed
    through rather than recomputed so there is one source of truth for the
    parallel/serial decision. ``engine_cls`` selects which `MatrixEngine`
    subclass each worker constructs.
    """
    mode_name = _resolve_parallel_mode_name(config.parallel_mode)
    if mode_name == "loky":
        if _loky_available():
            return _run_dynamic_shots_loky(
                plan,
                system_dims,
                n_clbits,
                seed_sequences,
                max_workers,
                engine_cls,
                initial_state,
            )
        warnings.warn(
            "parallel_mode='loky' requested but loky is unavailable; "
            "falling back to multiprocessing",
            UserWarning,
            stacklevel=3,
        )
    # "multiprocessing", plus the loky-unavailable fallback above.
    return _run_dynamic_shots_multiprocessing(
        plan,
        system_dims,
        n_clbits,
        seed_sequences,
        max_workers,
        engine_cls,
        initial_state,
    )
