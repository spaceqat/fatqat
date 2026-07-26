"""Pulse-family run configuration and planning-only result contracts."""

from __future__ import annotations

from dataclasses import dataclass

from ...backends.engine_contract import _SimulationConfig
from ...errors import BackendValidationError
from ...result import _ResultConfig


@dataclass(frozen=True)
class PulseSimulationConfig(_SimulationConfig):
    """v0.1 pulse execution controls, normalized to the serial engine policy."""

    placement_mode: str = "ASAP"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.parallel_mode == "auto":
            object.__setattr__(self, "parallel_mode", "serial")
        elif self.parallel_mode != "serial":
            raise BackendValidationError(
                "PulseBackend v0.1 supports only parallel_mode='auto' or 'serial'"
            )
        if self.max_workers not in (None, 1):
            raise BackendValidationError(
                "PulseBackend v0.1 supports only max_workers=None or 1"
            )
        if self.placement_mode not in ("ASAP", "ALAP"):
            raise BackendValidationError(
                "PulseBackend placement_mode must be 'ASAP' or 'ALAP'"
            )


@dataclass(frozen=True)
class PulseResultConfig(_ResultConfig):
    """Pulse backend result request; ``final_state`` maps to density_matrix."""


@dataclass(frozen=True)
class PulseResultRequest:
    """Resolved pulse result fields passed to the future density-matrix engine."""

    counts: bool
    density_matrix: bool
