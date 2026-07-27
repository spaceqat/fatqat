"""Private placement for boundary-free superconducting pulse runs."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Literal

from ..errors import BackendValidationError
from .resolved import PulseBlock, ResourceClaim

PlacementMode = Literal["ASAP", "ALAP"]
_EPSILON = 1e-12


@dataclass(frozen=True)
class _PlacedPulseRun:
    """One engine-private run with starts aligned to the original blocks."""

    blocks: tuple[PulseBlock, ...]
    starts_ns: tuple[float, ...]
    start_ns: float
    end_ns: float

    def __post_init__(self) -> None:
        if not self.blocks or len(self.blocks) != len(self.starts_ns):
            raise BackendValidationError(
                "a placed pulse run requires one start per block"
            )


def _conflicts(first: Iterable[ResourceClaim], second: Iterable[ResourceClaim]) -> bool:
    return not set(first).isdisjoint(second)


def _scheduled_starts(
    blocks: tuple[PulseBlock, ...], boundary_ns: float, mode: PlacementMode
) -> tuple[tuple[float, ...], float]:
    """Return conservative DAG-list-scheduler starts and the shared makespan."""
    relative_asap: list[float] = []
    for index, block in enumerate(blocks):
        predecessors = (
            relative_asap[earlier] + blocks[earlier].duration_ns
            for earlier in range(index)
            if _conflicts(blocks[earlier].resource_claims, block.resource_claims)
        )
        relative_asap.append(max((0.0, *predecessors)))
    horizon = max(
        start + block.duration_ns for start, block in zip(relative_asap, blocks)
    )
    if mode == "ASAP":
        relative = relative_asap
    else:
        relative = [0.0] * len(blocks)
        for index in range(len(blocks) - 1, -1, -1):
            successors = (
                relative[later]
                for later in range(index + 1, len(blocks))
                if _conflicts(
                    blocks[index].resource_claims, blocks[later].resource_claims
                )
            )
            relative[index] = min((horizon, *successors)) - blocks[index].duration_ns
    return tuple(boundary_ns + start for start in relative), boundary_ns + horizon


def place_pulse_run(
    blocks: Iterable[PulseBlock],
    *,
    boundary_ns: float,
    mode: PlacementMode = "ASAP",
) -> _PlacedPulseRun:
    """Schedule an unplaced run or validate a fully explicit one."""
    blocks = tuple(blocks)
    if not blocks:
        raise BackendValidationError("cannot place an empty pulse run")
    if mode not in ("ASAP", "ALAP"):
        raise BackendValidationError("pulse placement mode must be 'ASAP' or 'ALAP'")
    if not isfinite(boundary_ns) or boundary_ns < 0:
        raise BackendValidationError(
            "pulse placement boundary must be finite and non-negative"
        )

    explicit = tuple(block.start_ns is not None for block in blocks)
    if any(explicit) and not all(explicit):
        raise BackendValidationError(
            "a continuous pulse run must use either all explicit starts or no starts"
        )
    if not any(explicit):
        starts, end = _scheduled_starts(blocks, boundary_ns, mode)
        return _PlacedPulseRun(blocks, starts, boundary_ns, end)

    starts = tuple(
        float(block.start_ns) for block in blocks if block.start_ns is not None
    )
    if any(not isfinite(start) or start < boundary_ns - _EPSILON for start in starts):
        raise BackendValidationError(
            "an explicit pulse start cannot precede the current execution boundary"
        )
    for later, later_block in enumerate(blocks):
        for earlier in range(later):
            earlier_block = blocks[earlier]
            if _conflicts(
                earlier_block.resource_claims, later_block.resource_claims
            ) and starts[later] < (
                starts[earlier] + earlier_block.duration_ns - _EPSILON
            ):
                raise BackendValidationError(
                    "explicit placement reverses source order on a claimed resource"
                )
    end = max(start + block.duration_ns for start, block in zip(starts, blocks))
    return _PlacedPulseRun(blocks, starts, boundary_ns, end)
