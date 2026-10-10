"""Private controls, capabilities, and execution records for matrix simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

import numpy as np

from ..errors import BackendValidationError

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

    def __post_init__(self) -> None:
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
class _StateVectorResultRequest:
    """Resolved result fields requested for one statevector execution."""

    counts: bool
    statevector: bool


@dataclass(frozen=True)
class _DensityMatrixResultRequest:
    """Resolved result fields requested for one density-matrix execution."""

    counts: bool
    density_matrix: bool


@dataclass(frozen=True)
class _UnitaryResultRequest:
    """Resolved result fields requested for one unitary execution."""

    counts: bool
    unitary: bool


@dataclass(frozen=True)
class _SuperopResultRequest:
    """Resolved result fields requested for one super-operator execution."""

    counts: bool
    superop: bool


_ResultRequest = (
    _StateVectorResultRequest
    | _DensityMatrixResultRequest
    | _UnitaryResultRequest
    | _SuperopResultRequest
)


@dataclass(frozen=True)
class RawResult(Generic[StateDataT]):
    """Engine output before public result assembly.

    State uses runtime-native storage and may borrow engine data. Consumers copy
    or transfer it when needed. Count arrays remain NumPy arrays.
    """

    outcome_keys: np.ndarray | None = None
    outcome_counts: np.ndarray | None = None
    state: StateDataT | None = None


@dataclass(frozen=True, slots=True)
class _PlanFacts:
    """Runtime-independent semantic facts derived from one lowered plan."""

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
class _InitialClassicalState:
    """Optional classical values used to initialize each trajectory.

    Fields are always present. None requests default behavior without requiring
    support or storage for that component. Any supplied value, including an
    empty tuple or set, requires engine support; unsupported values are rejected.
    Simulator validates support and register width before execution. Direct
    engine callers must provide equally validated input. Engines only copy
    supplied values into their own mutable execution state.
    """

    # None keeps zero initialization lazy. Digits include qudit and erasure reports.
    clbits: tuple[int, ...] | None = None
    # None leaves occupancy implicit; an empty set means no carriers.
    occupied: frozenset[int] | None = None

    def __post_init__(self) -> None:
        if self.clbits is not None:
            if any(
                not isinstance(digit, (int, np.integer))
                or isinstance(digit, bool)
                or not 0 <= digit <= np.iinfo(np.int64).max
                for digit in self.clbits
            ):
                raise BackendValidationError(
                    "initial classical digits must be non-negative int64 integers"
                )
            object.__setattr__(
                self, "clbits", tuple(int(digit) for digit in self.clbits)
            )

    def validate(self, *, capabilities: _EngineCapabilities, n_clbits: int) -> None:
        """Check initialization against the complete engine and register width.

        Compiled-path eligibility is checked separately from fallback support.
        """
        if self.clbits is not None and not capabilities.supports_classical_register:
            raise BackendValidationError(
                "engine does not support initial classical digits"
            )
        if self.occupied is not None and not capabilities.supports_occupancy:
            raise BackendValidationError(
                f"method={capabilities.quantum.representation!r} cannot track carrier "
                "occupancy; the selected engine has no occupancy state"
            )

        if self.clbits is not None and len(self.clbits) != n_clbits:
            raise BackendValidationError(
                f"initial classical register has {len(self.clbits)} digits; expected {n_clbits}"
            )


@dataclass(frozen=True, slots=True)
class _InitialEvolutionState(Generic[StateDataT]):
    """Borrowed initial values; engines create their own mutable storage."""

    quantum: np.ndarray | StateDataT | None = None
    classical: _InitialClassicalState = field(default_factory=_InitialClassicalState)


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    """System dimensions, requested outputs, and sampling controls."""

    request: _ResultRequest
    system_dims: tuple[int, ...]
    n_clbits: int
    shots: int
    seed: int | None
