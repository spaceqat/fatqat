"""Private controls, capabilities, and execution records for matrix simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import numpy as np

from ..errors import BackendValidationError
from ..result import _ResultConfig

if TYPE_CHECKING:
    from ._engine.state import ClassicalState

StateDataT = TypeVar("StateDataT")

_SHOT_PARALLELISM_NAMES = frozenset({"auto", "serial", "threads", "processes"})
_KERNEL_PARALLELISM_NAMES = frozenset({"auto", "serial", "threads"})


@dataclass(frozen=True, slots=True)
class _SimulationConfig:
    """Normalized public matrix-simulator controls for one backend run."""

    seed: int | None = None
    shot_parallelism: Any = "auto"
    kernel_parallelism: Any = "auto"
    max_workers: Any = None
    fusion: Any = False
    error_handling: str = "raise"

    def __post_init__(self) -> None:
        if not isinstance(self.error_handling, str) or self.error_handling not in {
            "raise",
            "capture",
        }:
            raise BackendValidationError(
                f"unsupported error_handling={self.error_handling!r}; expected "
                "'raise' or 'capture'"
            )
        if self.seed is not None and (type(self.seed) is not int):
            raise BackendValidationError(
                f"seed must be an int or None, got {self.seed!r}"
            )
        if not isinstance(self.shot_parallelism, str) or (
            self.shot_parallelism not in _SHOT_PARALLELISM_NAMES
        ):
            raise BackendValidationError(
                f"unsupported shot_parallelism={self.shot_parallelism!r}; expected "
                "'auto', 'serial', 'threads', or 'processes'"
            )
        if not isinstance(self.kernel_parallelism, str) or (
            self.kernel_parallelism not in _KERNEL_PARALLELISM_NAMES
        ):
            raise BackendValidationError(
                "unsupported kernel_parallelism="
                f"{self.kernel_parallelism!r}; expected 'auto', 'serial', or "
                "'threads'"
            )
        if self.max_workers is not None and (
            type(self.max_workers) is not int or self.max_workers < 1
        ):
            raise BackendValidationError(
                "max_workers must be a positive int or None, got "
                f"{self.max_workers!r}"
            )
        if self.shot_parallelism in {"threads", "processes"} and (
            self.kernel_parallelism == "threads"
        ):
            raise BackendValidationError(
                "shot and kernel parallelism cannot both be explicitly parallel"
            )
        if self.max_workers == 1 and (
            self.shot_parallelism in {"threads", "processes"}
            or self.kernel_parallelism == "threads"
        ):
            raise BackendValidationError(
                "max_workers=1 contradicts explicit threaded or process parallelism"
            )
        if type(self.fusion) is not bool:
            raise BackendValidationError(f"fusion must be a bool, got {self.fusion!r}")


@dataclass(frozen=True)
class _SimulatorResultConfig(_ResultConfig):
    """Result requests accepted by gate-level simulators."""

    final_classical_state: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.final_classical_state) is not bool:
            raise BackendValidationError(
                "final_classical_state must be a bool, got "
                f"{self.final_classical_state!r}"
            )


@dataclass(frozen=True)
class _ResultRequest:
    """Resolved outputs requested from an engine in its own representation."""

    counts: bool
    final_state: bool
    final_classical_state: bool = False


@dataclass(frozen=True)
class RawResult(Generic[StateDataT]):
    """Engine output before public result assembly.

    State uses runtime-native storage and may borrow engine data. Consumers copy
    or transfer it when needed. Count arrays remain NumPy arrays.
    """

    outcome_keys: np.ndarray | None = None
    outcome_counts: np.ndarray | None = None
    state: StateDataT | None = None
    # An independent copy with allocated register digits.
    classical_state: ClassicalState | None = None


@dataclass(frozen=True, slots=True)
class _PlanFacts:
    """Runtime-independent semantic facts derived from one lowered plan."""

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
    # Complete shot batches can run in CPU processes with transferable payloads.
    supports_process_shots: bool = False

    @property
    def supports_classical_register(self) -> bool:
        """Whether measurement reports and conditions have a register."""
        return self.trajectory.classical_register

    @property
    def supports_occupancy(self) -> bool:
        """Whether an execution may own explicit carrier occupancy."""
        return self.trajectory.occupancy


@dataclass(frozen=True, slots=True)
class _RunContext:
    """System dimensions, requested outputs, and sampling controls for a run."""

    request: _ResultRequest
    system_dims: tuple[int, ...]
    n_clbits: int
    shots: int
    seed: int | None
