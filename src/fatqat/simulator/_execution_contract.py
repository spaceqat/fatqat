"""Private semantic and execution records for matrix simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .._backends.engine_contract import _ResultRequest

ExecutionShape = Literal["operator", "single_pass", "per_shot"]


@dataclass(frozen=True, slots=True)
class _PlanFacts:
    """Runtime-independent semantic facts derived from one lowered plan."""

    execution_shape: ExecutionShape
    deferred_measurements: tuple[tuple[int, int], ...]
    written_clbits: frozenset[int]
    stochastic_final_state: bool
    has_measurement: bool
    has_reset: bool
    has_channel: bool
    has_condition: bool


@dataclass(frozen=True, slots=True)
class _EngineCapabilities:
    """Static numerical support exposed without inspecting a plan."""

    supports_kernel_threads: bool
    thread_capacity: int
    supports_fusion: bool


@dataclass(frozen=True, slots=True)
class _ExecutionPolicy:
    """Final implementation and routing decisions for one execution."""

    shot_strategy: Literal["none", "serial", "threads", "processes"]
    kernel_strategy: Literal["serial", "adaptive", "threads"]
    worker_limit: int | None
    fusion: bool
    use_compiled_multi_shot_kernel: bool = False


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    """Semantic and numerical values executed under a resolved policy."""

    execution_shape: ExecutionShape
    request: _ResultRequest
    system_dims: tuple[int, ...]
    n_clbits: int
    shots: int
    seed: int | None
    initial_state: np.ndarray | None
    initial_occupied: frozenset[int] | None
