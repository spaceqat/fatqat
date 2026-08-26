"""Normalized controls, result requests, and raw matrix execution results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import BackendValidationError

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
class RawResult:
    """Engine-produced execution data before public Result packaging."""

    outcome_keys: np.ndarray | None = None
    outcome_counts: np.ndarray | None = None
    state: np.ndarray | None = None
