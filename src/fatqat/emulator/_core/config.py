"""Pulse-family run configuration.

The emulator's result request is the shared `fatqat.result._ResultConfig`
rather than a pulse-specific subclass: ``counts`` and ``final_state`` describe
what the caller wants back, which is family-neutral. The run controls below
are also shared by all three physics systems.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...errors import BackendValidationError
from .scheduling import SchedulingMode, _validate_schedule_mode


@dataclass(frozen=True)
class _EmulatorConfig:
    """Internal normalized schema for one public pulse-emulator run.

    Each of the three public emulators accepts this schema through
    ``run(simulation_config={...})``.

    Deliberately standalone rather than derived from the matrix family's
    ``_SimulationConfig``: that base carries ``parallel_mode``,
    ``max_workers``, ``numba_parallel``, and ``fusion``, which steer the matrix
    engine's process-, thread-, and plan-level execution. Pulse execution
    integrates a
    physics model through one serial solver call and has no such engine to
    steer, so inheriting those fields would advertise tuning that silently
    does nothing. `_normalize_config` derives accepted keys from this
    schema, so they are now rejected by name instead.

    ``schedule_mode`` selects lightweight ASAP or ALAP pulse placement.
    """

    seed: int | None = None
    schedule_mode: SchedulingMode = "ASAP"

    def __post_init__(self) -> None:
        if self.seed is not None and type(self.seed) is not int:
            raise BackendValidationError(
                f"seed must be an int or None, got {self.seed!r}"
            )
        object.__setattr__(
            self, "schedule_mode", _validate_schedule_mode(self.schedule_mode)
        )
