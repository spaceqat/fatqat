"""Parallel dynamic-shot execution: multiprocessing/loky worker dispatch.

Shared by `StateVectorBackend`'s dynamic (per-shot) execution path when counts
are requested, multiple shots are needed, and backend options allow it. Kept
separate from `statevector.py` because it is pure execution-strategy plumbing
(worker counts, batching, process-pool dispatch) with no state/physics
content of its own.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import TYPE_CHECKING, Any

import numpy as np

from .statevectorengine import StateVectorEngine
from .steps import ResolvedStep

if TYPE_CHECKING:
    from .statevector import _BackendConfig, _ResultRequest


def _shot_seed_sequences(seed: int | None, n_iters: int) -> list[np.random.SeedSequence]:
    """Spawn one independent child `SeedSequence` per logical shot.

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
# per-shot simulation they save, so automatic mode stays serial — this keeps the
# default backend and small runs (including the existing Phase 2 dynamic tests)
# fast instead of spawning workers for a handful of shots. The value is a tunable
# heuristic (it keys off shot count, not per-shot cost); 32 is chosen to stay
# serial for typical small/interactive runs while parallelizing real multishot
# jobs. An explicit max_workers > 1 always parallelizes and bypasses this floor.
_PARALLEL_MIN_SHOTS = 32


def _effective_max_workers(max_workers: object, n_iters: int) -> int:
    if max_workers is None:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)
        return min(n_iters, cpu_count() or 1)
    return int(max_workers)


def _planned_workers(
    config: "_BackendConfig",
    request: "_ResultRequest",
    n_iters: int,
) -> int | None:
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
    """Resolve `"auto"` to a concrete backend; other names are passed through.

    `name` is already one of the validated `_PARALLEL_MODE_NAMES` (checked
    in `_BackendConfig.__post_init__`), so this only expands `"auto"`.
    """
    if name == "auto":
        return "loky" if _loky_available() else "multiprocessing"
    return name


def _split_into_batches(
    seed_sequences: list[np.random.SeedSequence],
    n_batches: int,
) -> list[list[np.random.SeedSequence]]:
    """Split shots into up to n_batches contiguous batches, one task per worker.

    Batching pickles the (matrix-carrying) plan once per worker instead of once
    per shot, and lets each worker reuse a single engine across its batch. Shot i
    keeps ``seed_sequences[i]`` regardless of batching, so aggregated counts are
    byte-for-byte identical to the serial path.
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
) -> list[tuple[int, ...]]:
    """Run a contiguous batch of dynamic shots in one worker with one engine."""
    from .statevector import _execute_dynamic_plan_one_shot

    engine = StateVectorEngine()
    snapshots: list[tuple[int, ...]] = []
    for seed_sequence in seed_batch:
        rng = np.random.default_rng(seed_sequence)
        engine.initialize(system_dims)
        snapshots.append(_execute_dynamic_plan_one_shot(engine, plan, n_clbits, rng))
    return snapshots


def _run_dynamic_shots_multiprocessing(
    plan: list[ResolvedStep],
    system_dims: tuple[int, ...],
    n_clbits: int,
    seed_sequences: list[np.random.SeedSequence],
    max_workers: int,
) -> list[tuple[int, ...]]:
    batches = _split_into_batches(seed_sequences, max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            _run_dynamic_shot_batch,
            repeat(plan),
            repeat(system_dims),
            repeat(n_clbits),
            batches,
        )
        return [snapshot for batch in results for snapshot in batch]


def _run_dynamic_shots_loky(
    plan: list[ResolvedStep],
    system_dims: tuple[int, ...],
    n_clbits: int,
    seed_sequences: list[np.random.SeedSequence],
    max_workers: int,
) -> list[tuple[int, ...]]:
    from loky import get_reusable_executor

    batches = _split_into_batches(seed_sequences, max_workers)
    executor = get_reusable_executor(max_workers=max_workers)
    results = executor.map(
        _run_dynamic_shot_batch,
        repeat(plan),
        repeat(system_dims),
        repeat(n_clbits),
        batches,
    )
    return [snapshot for batch in results for snapshot in batch]


def _run_dynamic_shots_parallel(
    config: "_BackendConfig",
    plan: list[ResolvedStep],
    system_dims: tuple[int, ...],
    n_clbits: int,
    seed_sequences: list[np.random.SeedSequence],
    max_workers: int,
) -> list[tuple[int, ...]]:
    """Dispatch shots to worker processes. Caller must supply `max_workers`.

    `max_workers` is the same value `_planned_workers` already returned to
    decide whether to call this function at all; it is passed through rather
    than recomputed here, so there is exactly one source of truth for the
    parallel/serial decision.
    """
    mode_name = _resolve_parallel_mode_name(config.parallel_mode)
    if mode_name == "loky":
        if _loky_available():
            return _run_dynamic_shots_loky(
                plan, system_dims, n_clbits, seed_sequences, max_workers
            )
        warnings.warn(
            "parallel_mode='loky' requested but loky is unavailable; "
            "falling back to multiprocessing",
            UserWarning,
            stacklevel=3,
        )
    # "multiprocessing", plus the loky-unavailable fallback above.
    return _run_dynamic_shots_multiprocessing(
        plan, system_dims, n_clbits, seed_sequences, max_workers
    )
