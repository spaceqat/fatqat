"""Loky process routes for already-resolved matrix executions."""

from __future__ import annotations

from itertools import repeat
from typing import TYPE_CHECKING, Any

import numpy as np

from .._execution_contract import (
    _ExecutionContext as ExecutionContext,
    _ExecutionPolicy as ExecutionPolicy,
)
from .._execution_policy import _process_child_policy
from .base import _shot_seed_sequences

if TYPE_CHECKING:
    from typing import Protocol

    class _ProcessEngine(Protocol):
        def execute_shot_batch(
            self,
            context: ExecutionContext,
            payload: Any,
            seed_batch: list[np.random.SeedSequence],
            policy: ExecutionPolicy,
        ) -> list[tuple[int, ...]]: ...

    class _EngineFactory(Protocol):
        def __call__(self) -> _ProcessEngine: ...


# Installed before scientific modules import in a loky child. The explicit
# Numba mask is also applied by the child's local execution scope.
_WORKER_THREAD_VARS = (
    "BLIS_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _split_into_batches(
    seed_sequences: list[np.random.SeedSequence], n_batches: int
) -> list[list[np.random.SeedSequence]]:
    """Split ordered shots into no more batches than useful work items."""
    n_items = len(seed_sequences)
    n_batches = max(1, min(n_batches, n_items))
    base, extra = divmod(n_items, n_batches)
    batches: list[list[np.random.SeedSequence]] = []
    start = 0
    for index in range(n_batches):
        size = base + (1 if index < extra else 0)
        batches.append(seed_sequences[start : start + size])
        start += size
    return batches


def _run_shot_batch(
    engine_cls: _EngineFactory,
    context: ExecutionContext,
    payload: Any,
    seed_batch: list[np.random.SeedSequence],
    child_policy: ExecutionPolicy,
) -> list[tuple[int, ...]]:
    engine = engine_cls()
    return engine.execute_shot_batch(context, payload, seed_batch, child_policy)


def _loky_executor(max_workers: int):
    from loky import get_reusable_executor

    return get_reusable_executor(
        max_workers=max_workers,
        env={name: "1" for name in _WORKER_THREAD_VARS},
    )


def _run_shots_in_processes(
    engine_cls: _EngineFactory,
    context: ExecutionContext,
    payload: Any,
    policy: ExecutionPolicy,
) -> list[tuple[int, ...]]:
    """Run shot batches in real child processes under a serial child policy."""
    assert policy.shot_strategy == "processes"
    assert policy.worker_limit is not None, "process policy requires a worker ceiling"
    seeds = _shot_seed_sequences(context.seed, context.shots)
    batches = _split_into_batches(seeds, policy.worker_limit)
    child = _process_child_policy(policy)
    # Keep Loky's process-global reusable pool at the stable resolved ceiling;
    # the number of submitted batches still bounds useful work for this run.
    executor = _loky_executor(policy.worker_limit)
    results = executor.map(
        _run_shot_batch,
        repeat(engine_cls),
        repeat(context),
        repeat(payload),
        batches,
        repeat(child),
    )
    return [snapshot for batch in results for snapshot in batch]
