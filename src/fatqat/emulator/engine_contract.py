"""Pulse-family run configuration and planning-only result contracts."""

from __future__ import annotations

from dataclasses import dataclass

from ..backends.engine_contract import _SimulationConfig
from ..errors import BackendValidationError
from ..result import _ResultConfig
from .scheduling import SchedulingMode, _validate_schedule_mode


@dataclass(frozen=True)
class PulseSimulationConfig(_SimulationConfig):
    """Internal normalized schema for pulse simulation settings.

    The public input remains ``PulseBackend.run(simulation_config={...})``.
    ``parallel_mode="auto"`` normalizes to ``"serial"``; v0.1 rejects worker
    counts above one. ``schedule_mode`` selects lightweight ASAP or ALAP pulse
    placement.
    """

    schedule_mode: SchedulingMode = "ASAP"

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
        object.__setattr__(
            self, "schedule_mode", _validate_schedule_mode(self.schedule_mode)
        )


@dataclass(frozen=True)
class PulseResultConfig(_ResultConfig):
    """Internal normalized pulse-result request schema.

    The public ``final_state`` flag requests a physical density matrix; the
    backend deliberately does not expose a separate density-matrix key.
    """
