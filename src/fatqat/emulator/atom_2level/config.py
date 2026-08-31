"""Per-run simulation configuration for the two-level atom emulator."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from ...errors import BackendValidationError
from .._core.config import _EmulatorConfig

_CUTOFF_ERROR = "interaction_cutoff must be None or a finite nonnegative real number"


def _normalize_interaction_cutoff(value: object) -> float | None:
    """Normalize one public interaction-distance cutoff."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise BackendValidationError(_CUTOFF_ERROR)
    try:
        cutoff = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BackendValidationError(_CUTOFF_ERROR) from exc
    if not isfinite(cutoff) or cutoff < 0.0:
        raise BackendValidationError(_CUTOFF_ERROR)
    return cutoff


@dataclass(frozen=True)
class _Atom2LevelSimulationConfig(_EmulatorConfig):
    """Normalized controls for one two-level atom simulation run."""

    interaction_cutoff: float | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "interaction_cutoff",
            _normalize_interaction_cutoff(self.interaction_cutoff),
        )


__all__: list[str] = []
