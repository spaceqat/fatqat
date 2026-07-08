"""Value objects crossing the statevector backend/engine boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import BackendValidationError


_PARALLEL_MODE_NAMES = frozenset({"auto", "serial", "multiprocessing", "loky"})


@dataclass(frozen=True)
class _EngineConfig:
    """Normalized statevector engine execution-strategy options."""

    max_workers: Any = None
    parallel_mode: Any = "auto"

    def __post_init__(self) -> None:
        mw = self.max_workers
        if mw is not None and (type(mw) is not int or mw < 1):
            raise BackendValidationError(
                f"max_workers must be a positive int or None, got {mw!r}"
            )
        if self.parallel_mode not in _PARALLEL_MODE_NAMES:
            raise BackendValidationError(
                f"unsupported parallel_mode={self.parallel_mode!r}"
            )


@dataclass(frozen=True)
class _ResultRequest:
    """Resolved result fields requested for one execution."""

    counts: bool
    statevector: bool


@dataclass(frozen=True)
class RawResult:
    """Engine-produced execution data before public Result packaging."""

    outcome_keys: np.ndarray | None = None
    outcome_counts: np.ndarray | None = None
    state: np.ndarray | None = None
