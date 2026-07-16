"""Parallel dynamic-shot execution for the NumPy matrix simulators.

Mirrors the serial per-shot path (`_NumpyMatrixSimulator._run_one_shot`) across
worker processes when counts are requested for enough shots and options allow
it. Kept separate from `np.py` because it is pure execution-strategy plumbing
(worker counts, batching, process-pool dispatch) with no state/physics content:
each worker constructs the requested `Simulator` subclass and runs the shared
per-shot loop. The subclass is passed as a plain class object, which pickles
trivially to workers, so this module has no compile-time dependency on any
concrete simulator.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import TYPE_CHECKING

import numpy as np

from ..backends.engine_contract import (
    _DensityMatrixResultRequest as DensityMatrixResultRequest,
    _EngineConfig as EngineConfig,
    _StateVectorResultRequest as StateVectorResultRequest,
)
from ..backends.steps import ResolvedStep

if TYPE_CHECKING:
    from typing import Protocol

    class _DynamicSimulator(Protocol):
        """The slice of the `Simulator` interface a per-shot worker touches."""

        def initialize(self, system_dims: Sequence[int], n_clbits: int = 0) -> None: ...

        def _run_one_shot(
            self, plan: list[ResolvedStep], rng: np.random.Generator
        ) -> tuple[int, ...]: ...

    class _SimulatorFactory(Protocol):
        def __call__(self) -> _DynamicSimulator: ...


_ResultRequest = StateVectorResultRequest | DensityMatrixResultRequest


def _shot_seed_sequences(
    seed: int | None, n_iters: int
) -> list[np.random.SeedSequence]:
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
    per shot, and lets each worker reuse a single simulator across its batch.
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
    simulator_cls: _SimulatorFactory,
) -> list[tuple[int, ...]]:
    """Run a contiguous batch of dynamic shots in one worker with one simulator."""
    simulator = simulator_cls()
    snapshots: list[tuple[int, ...]] = []
    for seed_sequence in seed_batch:
        simulator.initialize(system_dims, n_clbits)
        snapshots.append(
            simulator._run_one_shot(plan, np.random.default_rng(seed_sequence))
        )
    return snapshots


def _run_dynamic_shots_multiprocessing(
    plan: list[ResolvedStep],
    system_dims: tuple[int, ...],
    n_clbits: int,
    seed_sequences: list[np.random.SeedSequence],
    max_workers: int,
    simulator_cls: _SimulatorFactory,
) -> list[tuple[int, ...]]:
    batches = _split_into_batches(seed_sequences, max_workers)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            _run_dynamic_shot_batch,
            repeat(plan),
            repeat(system_dims),
            repeat(n_clbits),
            batches,
            repeat(simulator_cls),
        )
        return [snapshot for batch in results for snapshot in batch]


def _run_dynamic_shots_loky(
    plan: list[ResolvedStep],
    system_dims: tuple[int, ...],
    n_clbits: int,
    seed_sequences: list[np.random.SeedSequence],
    max_workers: int,
    simulator_cls: _SimulatorFactory,
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
        repeat(simulator_cls),
    )
    return [snapshot for batch in results for snapshot in batch]


def _run_dynamic_shots_parallel(
    config: EngineConfig,
    plan: list[ResolvedStep],
    system_dims: tuple[int, ...],
    n_clbits: int,
    seed_sequences: list[np.random.SeedSequence],
    max_workers: int,
    simulator_cls: _SimulatorFactory,
) -> list[tuple[int, ...]]:
    """Dispatch shots to worker processes. Caller must supply ``max_workers``.

    ``max_workers`` is the same value `_planned_workers` already returned, passed
    through rather than recomputed so there is one source of truth for the
    parallel/serial decision. ``simulator_cls`` selects which `Simulator`
    subclass each worker constructs.
    """
    mode_name = _resolve_parallel_mode_name(config.parallel_mode)
    if mode_name == "loky":
        if _loky_available():
            return _run_dynamic_shots_loky(
                plan, system_dims, n_clbits, seed_sequences, max_workers, simulator_cls
            )
        warnings.warn(
            "parallel_mode='loky' requested but loky is unavailable; "
            "falling back to multiprocessing",
            UserWarning,
            stacklevel=3,
        )
    # "multiprocessing", plus the loky-unavailable fallback above.
    return _run_dynamic_shots_multiprocessing(
        plan, system_dims, n_clbits, seed_sequences, max_workers, simulator_cls
    )
