"""Pure resolution of public matrix controls into one private run policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from .._backends.engine_contract import _SimulationConfig as SimulationConfig
from ..errors import BackendValidationError
from ._execution_contract import (
    _KernelCapabilities as KernelCapabilities,
    _PlanFacts as PlanFacts,
)


@dataclass(frozen=True, slots=True)
class _ExecutionPolicy:
    """Final implementation and routing decisions for one execution."""

    shot_strategy: Literal["none", "serial", "threads", "processes"]
    kernel_strategy: Literal["serial", "adaptive", "threads"]
    worker_limit: int | None
    fusion: bool
    use_compiled_multi_shot_kernel: bool = False


_PARALLEL_MIN_SHOTS = 32


def _process_worker_ceiling(requested: int | None) -> int:
    """Resolve the stable size of the reusable process executor."""
    if requested is not None:
        return requested
    if hasattr(os, "process_cpu_count"):
        return max(1, os.process_cpu_count() or 1)
    # Before Python 3.13, os.cpu_count counts every core the system has,
    # ignoring the process's affinity mask: a pinned or container-limited
    # process would start more workers than it may run.
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def _explicit_thread_worker_ceiling(
    requested: int | None, capabilities: KernelCapabilities
) -> int:
    """Resolve a concrete ceiling for a required threaded axis."""
    return max(
        1,
        min(
            requested or capabilities.thread_capacity,
            capabilities.thread_capacity,
        ),
    )


def _adaptive_thread_worker_ceiling(
    requested: int | None, capabilities: KernelCapabilities
) -> int | None:
    """Clamp an explicit ceiling while preserving an omitted caller mask."""
    if requested is None:
        return None
    return max(1, min(requested, capabilities.thread_capacity))


def _validate_execution_controls(
    simulation: SimulationConfig,
    capabilities: KernelCapabilities,
) -> None:
    """Reject plan-independent engine controls before lowering."""
    if (
        simulation.kernel_parallelism == "threads"
        and not capabilities.supports_kernel_threads
    ):
        raise BackendValidationError(
            "kernel_parallelism='threads' requires an engine with threaded "
            "numerical kernels"
        )
    if simulation.fusion and not capabilities.supports_fusion:
        raise BackendValidationError(
            "fusion=True is not supported by the selected matrix engine; fusion "
            "does not control compiled multi-shot execution"
        )


def _materialization_policy(parent: _ExecutionPolicy) -> _ExecutionPolicy:
    """Project process-shot preparation into local parent execution controls."""
    if parent.shot_strategy != "processes":
        return parent
    return _ExecutionPolicy(
        shot_strategy="serial",
        kernel_strategy="serial",
        worker_limit=1,
        fusion=parent.fusion,
        use_compiled_multi_shot_kernel=False,
    )


def _process_child_policy(parent: _ExecutionPolicy) -> _ExecutionPolicy:
    """Revoke dispatch and preparation authority in a process child."""
    assert parent.shot_strategy == "processes"
    return _ExecutionPolicy(
        shot_strategy="serial",
        kernel_strategy="serial",
        worker_limit=1,
        # The parent already materialized any requested fusion into the payload.
        fusion=False,
        use_compiled_multi_shot_kernel=False,
    )


def _should_probe_compiled_multi_shot(
    simulation: SimulationConfig,
    *,
    facts: PlanFacts,
    counts_requested: bool,
    state_requested: bool,
    initial_occupied: frozenset[int] | None,
) -> bool:
    """Whether cheap prerequisites justify exact-plan compatibility analysis."""
    shot_request = simulation.shot_parallelism
    kernel_request = simulation.kernel_parallelism
    can_select_compiled = (
        shot_request == "threads"
        or (shot_request == "serial" and kernel_request == "serial")
        or (shot_request == "auto" and kernel_request != "threads")
    )
    return (
        can_select_compiled
        and facts.execution_shape == "per_shot"
        and counts_requested
        and not state_requested
        and initial_occupied is None
    )


def _adaptive_kernel_policy(
    simulation: SimulationConfig,
    capabilities: KernelCapabilities,
) -> tuple[str, int | None]:
    """Resolve public kernel auto without reading the active caller mask."""
    if (
        not capabilities.supports_kernel_threads
        or capabilities.thread_capacity < 2
        or simulation.max_workers == 1
    ):
        return "serial", 1
    return (
        "adaptive",
        _adaptive_thread_worker_ceiling(simulation.max_workers, capabilities),
    )


def _resolve_execution_policy(
    simulation: SimulationConfig,
    *,
    facts: PlanFacts,
    counts_requested: bool,
    state_requested: bool,
    capabilities: KernelCapabilities,
    compiled_multi_shot_compatible: bool,
    shots: int,
    initial_occupied: frozenset[int] | None,
    plan_is_empty: bool = False,
) -> _ExecutionPolicy:
    """Resolve validated controls and semantic facts into one final policy."""
    execution_has_shots = facts.execution_shape == "per_shot"
    shot_shardable = execution_has_shots and counts_requested and not state_requested
    compiled_eligible = (
        shot_shardable and initial_occupied is None and compiled_multi_shot_compatible
    )
    shot_request = simulation.shot_parallelism
    kernel_request = simulation.kernel_parallelism

    if shot_request in {"threads", "processes"}:
        if not execution_has_shots:
            raise BackendValidationError(
                f"shot_parallelism={shot_request!r} requires independent "
                "per-shot evolution"
            )
        if shots < 2:
            raise BackendValidationError(
                f"shot_parallelism={shot_request!r} requires at least two shots"
            )
        if not shot_shardable:
            raise BackendValidationError(
                f"shot_parallelism={shot_request!r} cannot shard the requested "
                "result fields"
            )

    if kernel_request == "threads":
        assert (
            capabilities.supports_kernel_threads
        ), "explicit kernel threads require validated engine capabilities"
        kernel_workers = _explicit_thread_worker_ceiling(
            simulation.max_workers, capabilities
        )
        if kernel_workers < 2 and not plan_is_empty:
            raise BackendValidationError(
                "kernel_parallelism='threads' requires an effective thread "
                "capacity of at least two"
            )
        kernel_strategy = "threads"
    elif kernel_request == "auto":
        kernel_strategy, kernel_workers = _adaptive_kernel_policy(
            simulation, capabilities
        )
    else:
        kernel_strategy, kernel_workers = "serial", 1

    use_compiled = False

    if shot_request == "threads":
        if not compiled_eligible:
            raise BackendValidationError(
                "shot_parallelism='threads' requires compiled multi-shot support "
                "from the selected engine for this plan"
            )
        shot_workers = _explicit_thread_worker_ceiling(
            simulation.max_workers, capabilities
        )
        if shot_workers < 2:
            raise BackendValidationError(
                "shot_parallelism='threads' requires an effective thread "
                "capacity of at least two"
            )
        shot_strategy = "threads"
        kernel_strategy = "serial"
        worker_limit = shot_workers
        use_compiled = True
    elif shot_request == "processes":
        process_workers = _process_worker_ceiling(simulation.max_workers)
        if process_workers < 2:
            raise BackendValidationError(
                "shot_parallelism='processes' requires an effective process "
                "capacity of at least two"
            )
        shot_strategy = "processes"
        kernel_strategy = "serial"
        worker_limit = process_workers
    elif shot_request == "serial":
        shot_strategy = "serial" if execution_has_shots else "none"
        if kernel_request == "serial" and compiled_eligible:
            kernel_strategy = "serial"
            worker_limit = 1
            use_compiled = True
        else:
            worker_limit = kernel_workers
    elif kernel_request == "threads":
        # An explicit kernel axis owns the only parallel work in auto/threads.
        shot_strategy = "serial" if execution_has_shots else "none"
        kernel_strategy = "threads"
        worker_limit = kernel_workers
    elif compiled_eligible:
        shot_workers = _adaptive_thread_worker_ceiling(
            simulation.max_workers, capabilities
        )
        if capabilities.thread_capacity >= 2 and (
            shot_workers is None or shot_workers > 1
        ):
            shot_strategy = "threads"
            worker_limit = shot_workers
        else:
            shot_strategy = "serial"
            worker_limit = 1
        kernel_strategy = "serial"
        use_compiled = True
    else:
        process_workers = _process_worker_ceiling(simulation.max_workers)
        if shot_shardable and shots >= _PARALLEL_MIN_SHOTS and process_workers > 1:
            shot_strategy = "processes"
            kernel_strategy = "serial"
            worker_limit = process_workers
        else:
            shot_strategy = "serial" if execution_has_shots else "none"
            worker_limit = kernel_workers

    assert not use_compiled or kernel_strategy == "serial"
    return _ExecutionPolicy(
        shot_strategy=shot_strategy,
        kernel_strategy=kernel_strategy,
        worker_limit=worker_limit,
        fusion=simulation.fusion,
        use_compiled_multi_shot_kernel=use_compiled,
    )
