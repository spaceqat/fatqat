"""Private scheduling for boundary-free pulse runs.

This is deliberately a lightweight scheduler for unscheduled pulse blocks.  A
future compiler-level scheduler can provide exact block start times instead;
those explicit times take precedence over the automatic ASAP/ALAP placement
here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Literal, cast

from ..errors import BackendValidationError
from .pulse import PulseBlock, ResourceClaim

SchedulingMode = Literal["ASAP", "ALAP"]
_EPSILON = 1e-12


def _validate_schedule_mode(mode: str) -> SchedulingMode:
    """Validate and narrow one public or engine scheduling-mode value."""
    if mode not in ("ASAP", "ALAP"):
        raise BackendValidationError("pulse schedule_mode must be 'ASAP' or 'ALAP'")
    return cast(SchedulingMode, mode)


@dataclass(frozen=True)
class _ScheduledPulseRun:
    """One engine-private continuous region after pulse placement.

    ``blocks`` retain source order while ``starts`` stores the aligned absolute
    start of each block. ``start_time`` is the preceding execution boundary;
    ``end_time`` is the region makespan.
    """

    blocks: tuple[PulseBlock, ...]
    starts: tuple[float, ...]
    start_time: float
    end_time: float

    def __post_init__(self) -> None:
        if not self.blocks or len(self.blocks) != len(self.starts):
            raise BackendValidationError(
                "a scheduled pulse run requires one start per block"
            )


def _resource_claims_conflict(
    first: Iterable[ResourceClaim], second: Iterable[ResourceClaim]
) -> bool:
    """Return whether two blocks reserve at least one identical model handle."""
    return not set(first).isdisjoint(second)


def _schedule_starts(
    blocks: tuple[PulseBlock, ...], boundary_time: float, mode: SchedulingMode
) -> tuple[tuple[float, ...], float]:
    """Return conservative DAG-list-scheduler starts and the shared makespan."""
    relative_asap: list[float] = []
    for index, block in enumerate(blocks):
        predecessors = (
            relative_asap[earlier] + blocks[earlier].duration
            for earlier in range(index)
            if _resource_claims_conflict(
                blocks[earlier].resource_claims, block.resource_claims
            )
        )
        relative_asap.append(max((0.0, *predecessors)))
    horizon = max(start + block.duration for start, block in zip(relative_asap, blocks))
    if mode == "ASAP":
        relative = relative_asap
    else:
        relative = [0.0] * len(blocks)
        for index in range(len(blocks) - 1, -1, -1):
            successors = (
                relative[later]
                for later in range(index + 1, len(blocks))
                if _resource_claims_conflict(
                    blocks[index].resource_claims, blocks[later].resource_claims
                )
            )
            relative[index] = min((horizon, *successors)) - blocks[index].duration
    return tuple(boundary_time + start for start in relative), boundary_time + horizon


def schedule_pulse_run(
    blocks: Iterable[PulseBlock],
    *,
    boundary_time: float,
    mode: SchedulingMode = "ASAP",
) -> _ScheduledPulseRun:
    """Schedule an unplaced continuous region or validate explicit starts.

    With no explicit ``PulseBlock.start_time`` values, a conservative list
    scheduler places conflicting resource claims in source order and permits
    disjoint blocks to overlap. ASAP and ALAP share the ASAP makespan. If any
    block carries an explicit start, every block must do so; those starts take
    precedence over ``mode`` and are checked against the current boundary and
    resource-order constraints.

    Args:
        blocks: Non-empty boundary-free pulse-block sequence.
        boundary_time: Absolute time after the preceding measurement/reset.
        mode: Automatic ``"ASAP"`` or ``"ALAP"`` placement policy.

    Returns:
        Internal scheduled region with starts aligned to source blocks.

    Raises:
        BackendValidationError: If inputs are empty, non-finite, partially
            explicit, before the boundary, or reverse a claimed-resource
            dependency.
    """
    blocks = tuple(blocks)
    if not blocks:
        raise BackendValidationError("cannot schedule an empty pulse run")
    mode = _validate_schedule_mode(mode)
    if not isfinite(boundary_time) or boundary_time < 0:
        raise BackendValidationError(
            "pulse scheduling boundary must be finite and non-negative"
        )

    explicit = tuple(block.start_time is not None for block in blocks)
    if any(explicit) and not all(explicit):
        raise BackendValidationError(
            "a continuous pulse run must use either all explicit starts or no starts"
        )
    if not any(explicit):
        starts, end = _schedule_starts(blocks, boundary_time, mode)
        return _ScheduledPulseRun(blocks, starts, boundary_time, end)

    starts = tuple(
        float(block.start_time) for block in blocks if block.start_time is not None
    )
    if any(not isfinite(start) or start < boundary_time - _EPSILON for start in starts):
        raise BackendValidationError(
            "an explicit pulse start cannot precede the current execution boundary"
        )
    for later, later_block in enumerate(blocks):
        for earlier in range(later):
            earlier_block = blocks[earlier]
            if _resource_claims_conflict(
                earlier_block.resource_claims, later_block.resource_claims
            ) and starts[later] < (starts[earlier] + earlier_block.duration - _EPSILON):
                raise BackendValidationError(
                    "explicit scheduling reverses source order on a claimed resource"
                )
    end = max(start + block.duration for start, block in zip(starts, blocks))
    return _ScheduledPulseRun(blocks, starts, boundary_time, end)
