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
class _QuantumCapabilities:
    """Representation semantics, independent of numerical runtime."""

    # Quantum buffer interpretation and method-native result field.
    representation: Literal["statevector", "density_matrix", "unitary", "superop"]
    # Whether this representation can execute reset and finite channel maps.
    supports_nonunitary: bool
    # Reset/channel execution samples a branch per shot instead of evolving
    # the full ensemble. Support is declared separately by supports_nonunitary.
    nonunitary_is_stochastic: bool

    @property
    def is_operator(self) -> bool:
        """Whether evolution computes a map rather than a state under it."""
        return self.representation in {"unitary", "superop"}


@dataclass(frozen=True, slots=True)
class _TrajectoryCapabilities:
    """Classical components supported by the complete trajectory executor."""

    # Per-shot reported measurement digits, also read by feedforward conditions.
    classical_register: bool = False
    # Per-shot loaded-carrier state, updated by loss/Put and used to guard gates
    # and measurement. Support may come from fallback, not the compiled loop.
    occupancy: bool = False


@dataclass(frozen=True, slots=True)
class _KernelCapabilities:
    """Numerical controls, separate from supported trajectory state."""

    # Numerical operations can use threads within one evolution; this alone
    # does not imply support for running complete shots in parallel.
    supports_kernel_threads: bool
    # Runtime thread-pool ceiling, independent of the caller's active mask.
    # Current Numba kernel and compiled-shot execution share this pool.
    thread_capacity: int
    # Plan materialization can combine compatible adjacent operations.
    # This does not enable or disable compiled multi-shot execution.
    supports_fusion: bool


@dataclass(frozen=True, slots=True)
class _EngineCapabilities:
    """Engine-owned static support; no evolving storage or plan allocation."""

    quantum: _QuantumCapabilities
    # Flags declare support independently of the current classical container.
    trajectory: _TrajectoryCapabilities
    kernels: _KernelCapabilities

    @property
    def supports_classical_register(self) -> bool:
        """Whether measurement reports and conditions have a register."""
        return self.trajectory.classical_register

    @property
    def supports_occupancy(self) -> bool:
        """Whether an execution may own explicit carrier occupancy."""
        return self.trajectory.occupancy


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
