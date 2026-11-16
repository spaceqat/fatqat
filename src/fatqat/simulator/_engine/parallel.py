"""Loky process routes for already-resolved matrix executions."""

from __future__ import annotations

from collections.abc import Callable
from itertools import repeat
from typing import TYPE_CHECKING, Any

import numpy as np

from .state import EvolutionState
from .._execution_contract import (
    _RunContext,
)
from ..._backends.steps import ResolvedStep
from ._execution_policy import (
    _ExecutionPolicy as ExecutionPolicy,
    _process_child_policy,
)

if TYPE_CHECKING:
    from typing import Protocol

    class _ProcessEngine(Protocol):
        def prepare_payload(
            self,
            plan: tuple[ResolvedStep, ...],
            *,
            system_dims: tuple[int, ...],
            n_clbits: int,
            policy: ExecutionPolicy,
        ) -> Any: ...

        def execute_worker_shots(
            self,
            context: _RunContext,
            payload: Any,
            seed_batch: list[np.random.SeedSequence],
            policy: ExecutionPolicy,
            *,
            initial_state: EvolutionState,
        ) -> list[tuple[int, ...]]: ...


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
    engine_factory: Callable[[], _ProcessEngine],
    context: _RunContext,
    plan: tuple[ResolvedStep, ...],
    seed_batch: list[np.random.SeedSequence],
    child_policy: ExecutionPolicy,
    initial_state: EvolutionState,
) -> list[tuple[int, ...]]:
    engine = engine_factory()
    # Bind worker-local numerical resources once for the whole batch. No parent
    # callbacks, caches, or device allocations cross the process boundary.
    payload = engine.prepare_payload(
        plan,
        system_dims=context.system_dims,
        n_clbits=context.n_clbits,
        policy=child_policy,
    )
    return engine.execute_worker_shots(
        context, payload, seed_batch, child_policy, initial_state=initial_state
    )


def _loky_executor(max_workers: int):
    from loky import get_reusable_executor

    return get_reusable_executor(
        max_workers=max_workers,
        env={name: "1" for name in _WORKER_THREAD_VARS},
    )


def _run_shots_in_processes(
    engine_factory: Callable[[], _ProcessEngine],
    context: _RunContext,
    plan: tuple[ResolvedStep, ...],
    policy: ExecutionPolicy,
    initial_state: EvolutionState,
) -> list[tuple[int, ...]]:
    """Run shot batches in real child processes under a serial child policy."""
    assert policy.shot_strategy == "processes"
    assert policy.worker_limit is not None, "process policy requires a worker ceiling"
    # Assign streams before batching so worker count cannot change shot seeds.
    seeds = np.random.SeedSequence(context.seed).spawn(context.shots)
    batches = _split_into_batches(seeds, policy.worker_limit)
    child = _process_child_policy(policy)
    # Keep Loky's process-global reusable pool at the stable resolved ceiling;
    # the number of submitted batches still bounds useful work for this run.
    executor = _loky_executor(policy.worker_limit)
    results = executor.map(
        _run_shot_batch,
        repeat(engine_factory),
        repeat(context),
        repeat(plan),
        batches,
        repeat(child),
        repeat(initial_state),
    )
    return [snapshot for batch in results for snapshot in batch]
