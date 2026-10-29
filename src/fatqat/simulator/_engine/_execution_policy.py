"""Request classification, kernel scheduling, and Python workflow scheduling.

Engines select workflows. Kernel scheduling resolves only numerical parallelism;
Python workflow scheduling owns its shot restrictions and process execution.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from .state import EvolutionState

from .._execution_contract import (
    _SimulationConfig as SimulationConfig,
    _RunContext,
    _EngineCapabilities as EngineCapabilities,
    _QuantumCapabilities as QuantumCapabilities,
    _KernelCapabilities as KernelCapabilities,
)
from ...errors import BackendValidationError
from ..._backends.steps import (
    ApplyChannelStep,
    CustomOperationStep,
    LossStep,
    MeasurementStep,
    PutStep,
    ResetStep,
    ResolvedStep,
)


@dataclass(frozen=True, slots=True)
class _ExecutionPolicy:
    """Final implementation and routing decisions for one execution."""

    execution_path: Literal["operator", "single_pass", "per_shot"]
    shot_strategy: Literal["none", "serial", "threads", "processes"]
    kernel_strategy: Literal["serial", "adaptive", "threads"]
    worker_limit: int | None
    fusion: bool
    # Engine-owned identifier: scheduling preserves it without interpreting it.
    workflow: str = "python"


_PARALLEL_MIN_SHOTS = 32


# Request classification and common validation


def _select_execution_path(
    plan: Sequence[ResolvedStep],
    *,
    quantum_capabilities: QuantumCapabilities,
    initial_state: EvolutionState,
    classical_requested: bool = False,
) -> Literal["operator", "single_pass", "per_shot"]:
    """Classify the required evolution shape from plan and engine semantics.

    Single-pass sampling defers measurements and never writes their digits, so
    a requested final classical state with measurements needs a trajectory.
    """
    if quantum_capabilities.is_operator:
        return "operator"
    # Only the trajectory executor handles occupancy and erasure reports.
    if initial_state.classical.occupied is not None:
        return "per_shot"
    measured: set[int] = set()
    for step in plan:
        if getattr(step, "condition", None) is not None or isinstance(
            step, (PutStep, LossStep, CustomOperationStep)
        ):
            return "per_shot"
        if isinstance(step, MeasurementStep):
            if classical_requested or measured.intersection(step.measured_indices):
                return "per_shot"
            measured.update(step.measured_indices)
            continue
        if isinstance(step, ResetStep):
            targets = step.reset_indices
        else:
            targets = step.target_indices
        if (
            isinstance(step, (ResetStep, ApplyChannelStep))
            and quantum_capabilities.nonunitary_is_stochastic
        ):
            return "per_shot"
        if measured.intersection(targets):
            return "per_shot"
    return "single_pass"


def _validate_shot_parallelism(
    simulation_config: SimulationConfig,
    *,
    execution_path: Literal["operator", "single_pass", "per_shot"],
    counts_requested: bool,
    state_requested: bool,
    shots: int,
) -> None:
    """Validate common constraints for explicit shot parallelism."""
    shot_request = simulation_config.shot_parallelism
    if shot_request in {"threads", "processes"}:
        if execution_path != "per_shot":
            raise BackendValidationError(
                f"shot_parallelism={shot_request!r} requires independent "
                "per-shot evolution"
            )
        if shots < 2:
            raise BackendValidationError(
                f"shot_parallelism={shot_request!r} requires at least two shots"
            )
        if not counts_requested or state_requested:
            raise BackendValidationError(
                f"shot_parallelism={shot_request!r} cannot shard the requested "
                "result fields"
            )


# Worker limits and kernel scheduling


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


def _resolve_kernel_scheduling(
    kernel_parallelism: str,
    *,
    max_workers: int | None,
    capabilities: KernelCapabilities,
) -> tuple[Literal["serial", "adaptive", "threads"], int | None]:
    """Resolve kernel strategy and worker limit without selecting a workflow."""
    if kernel_parallelism == "threads":
        workers = _explicit_thread_worker_ceiling(max_workers, capabilities)
        if workers < 2:
            raise BackendValidationError(
                "kernel_parallelism='threads' requires an effective thread "
                "capacity of at least two"
            )
        return "threads", workers
    if (
        kernel_parallelism == "serial"
        or not capabilities.supports_kernel_threads
        or capabilities.thread_capacity < 2
        or max_workers == 1
    ):
        return "serial", 1
    return "adaptive", _adaptive_thread_worker_ceiling(max_workers, capabilities)


def _plan_python_execution(
    plan: Sequence[ResolvedStep],
    *,
    context: _RunContext,
    simulation_config: SimulationConfig,
    initial_state: EvolutionState,
    capabilities: EngineCapabilities,
) -> _ExecutionPolicy:
    """Classify and schedule execution after an engine selects Python sequencing.

    Local shots run serially; eligible batches may run in processes. Kernel
    threading follows the selected engine's capabilities and controls.
    """
    execution_path = _select_execution_path(
        plan,
        quantum_capabilities=capabilities.quantum,
        initial_state=initial_state,
        classical_requested=context.request.final_classical_state,
    )
    _validate_shot_parallelism(
        simulation_config,
        execution_path=execution_path,
        counts_requested=context.request.counts,
        state_requested=(
            context.request.final_state or context.request.final_classical_state
        ),
        shots=context.shots,
    )
    return _resolve_python_workflow_scheduling(
        simulation_config,
        execution_path=execution_path,
        counts_requested=context.request.counts,
        state_requested=(
            context.request.final_state or context.request.final_classical_state
        ),
        capabilities=capabilities.kernels,
        supports_process_shots=capabilities.supports_process_shots,
        shots=context.shots,
    )


def _resolve_python_workflow_scheduling(
    simulation_config: SimulationConfig,
    *,
    execution_path: Literal["operator", "single_pass", "per_shot"],
    counts_requested: bool,
    state_requested: bool,
    capabilities: KernelCapabilities,
    supports_process_shots: bool,
    shots: int,
) -> _ExecutionPolicy:
    """Schedule the Python workflow over serial or process shots.

    The caller validates common shot constraints. Python sequencing cannot
    thread independent shots; numerical kernels may use threads within one shot.
    """
    execution_has_shots = execution_path == "per_shot"
    can_parallelize_shots = (
        execution_has_shots and counts_requested and not state_requested
    )
    shot_request = simulation_config.shot_parallelism
    kernel_request = simulation_config.kernel_parallelism
    if shot_request == "threads":
        raise BackendValidationError(
            "shot_parallelism='threads' is not supported "
            "by the Python workflow for this plan"
        )
    kernel_strategy, kernel_workers = _resolve_kernel_scheduling(
        kernel_request,
        max_workers=simulation_config.max_workers,
        capabilities=capabilities,
    )

    if shot_request == "processes":
        process_workers = _process_worker_ceiling(simulation_config.max_workers)
        if process_workers < 2:
            raise BackendValidationError(
                "shot_parallelism='processes' requires an effective process "
                "capacity of at least two"
            )
        shot_strategy = "processes"
        kernel_strategy = "serial"
        worker_limit = process_workers
    elif shot_request == "serial" or kernel_request == "threads":
        # Serial shots or explicit kernel threading leave no process-shot axis.
        shot_strategy = "serial" if execution_has_shots else "none"
        worker_limit = kernel_workers
    else:
        process_workers = _process_worker_ceiling(simulation_config.max_workers)
        if (
            supports_process_shots
            and can_parallelize_shots
            and shots >= _PARALLEL_MIN_SHOTS
            and process_workers > 1
        ):
            shot_strategy = "processes"
            kernel_strategy = "serial"
            worker_limit = process_workers
        else:
            shot_strategy = "serial" if execution_has_shots else "none"
            worker_limit = kernel_workers

    return _ExecutionPolicy(
        execution_path=execution_path,
        shot_strategy=shot_strategy,
        kernel_strategy=kernel_strategy,
        worker_limit=worker_limit,
        fusion=simulation_config.fusion,
        workflow="python",
    )


# Process-worker policy


def _process_child_policy(parent: _ExecutionPolicy) -> _ExecutionPolicy:
    """Restrict a process worker to serial shots and kernels.

    ``parallel._run_shots_in_processes`` derives this policy once and passes it
    to every worker batch. Each worker uses it to build numerical bindings and
    execute its assigned shots through the Python workflow, with either NumPy
    or Numba kernels. Compiled Numba trajectories and operator workflows do not
    use this process route.

    Fusion is unsupported on these state-evolution routes. Compiled operator
    fusion is independent of this policy.
    """
    assert parent.shot_strategy == "processes"
    return replace(
        parent,
        shot_strategy="serial",
        kernel_strategy="serial",
        worker_limit=1,
    )
