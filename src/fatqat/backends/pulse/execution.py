"""Private pulse-run placement and execution-boundary orchestration.

The values in this module are intentionally engine-private.  Lowering exposes
only an ordered flat plan; this layer assigns pulse starts immediately before
continuous execution and never stores a schedule on a public backend object.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Callable, Iterable, TypeVar

from ...backends.steps import MeasurementStep, ResetStep
from ...errors import BackendValidationError
from .resolved import PulseBlock, ResourceClaim

_EPSILON = 1e-12


@dataclass(frozen=True)
class _PlacedPulseRun:
    """One private, boundary-free region of placed physical pulse blocks."""

    blocks: tuple[PulseBlock, ...]
    start_ns: float
    end_ns: float


@dataclass(frozen=True)
class _SchedulerInstruction:
    """Ephemeral scheduler input retaining the source block's identity."""

    block: PulseBlock
    source_index: int


def _claims_conflict(
    first: Iterable[ResourceClaim], second: Iterable[ResourceClaim]
) -> bool:
    return not set(first).isdisjoint(second)


def _place_explicit_run(
    blocks: tuple[PulseBlock, ...], boundary_ns: float
) -> _PlacedPulseRun:
    """Validate explicitly placed blocks without weakening their timing."""
    starts = [block.start_ns for block in blocks]
    if any(start is None for start in starts) and any(
        start is not None for start in starts
    ):
        raise BackendValidationError(
            "a continuous pulse run must use either all explicit starts or no starts"
        )
    placed = tuple(blocks)
    for block in placed:
        assert block.start_ns is not None
        if not isfinite(block.start_ns) or block.start_ns < boundary_ns - _EPSILON:
            raise BackendValidationError(
                "an explicit pulse start cannot precede the prior execution boundary"
            )
    for later_index, later in enumerate(placed):
        assert later.start_ns is not None
        for earlier in placed[:later_index]:
            assert earlier.start_ns is not None
            overlaps = (
                later.start_ns < earlier.start_ns + earlier.duration_ns - _EPSILON
                and earlier.start_ns < later.start_ns + later.duration_ns - _EPSILON
            )
            if overlaps and _claims_conflict(
                later.resource_claims, earlier.resource_claims
            ):
                raise BackendValidationError(
                    "explicit pulse blocks with conflicting resource claims overlap"
                )
    start = min(block.start_ns for block in placed if block.start_ns is not None)
    end = max(
        block.start_ns + block.duration_ns
        for block in placed
        if block.start_ns is not None
    )
    return _PlacedPulseRun(placed, start, end)


def _schedule_run(
    blocks: tuple[PulseBlock, ...], boundary_ns: float, mode: str
) -> _PlacedPulseRun:
    """Place a run using non-permuting ASAP/ALAP resource scheduling.

    ``_SchedulerInstruction`` is the pulse engine's scheduler-adapter seam.
    qutip-qip is deliberately not allowed to supply a public schedule: the
    adapter keeps source order and identity while applying its resource starts
    back to fresh ``PulseBlock`` values.  The current atomic pulse vocabulary
    has no dependency edges beyond conservative resource claims, so ASAP and
    ALAP share this deterministic earliest-start placement.
    """
    if mode not in {"ASAP", "ALAP"}:
        raise BackendValidationError("pulse placement mode must be 'ASAP' or 'ALAP'")
    instructions = tuple(
        _SchedulerInstruction(block, source_index)
        for source_index, block in enumerate(blocks)
    )
    placed: list[PulseBlock] = []
    for instruction in instructions:
        start = boundary_ns
        for earlier in placed:
            assert earlier.start_ns is not None
            if _claims_conflict(
                instruction.block.resource_claims, earlier.resource_claims
            ):
                start = max(start, earlier.start_ns + earlier.duration_ns)
        placed.append(replace(instruction.block, start_ns=start))
    end = max(
        block.start_ns + block.duration_ns
        for block in placed
        if block.start_ns is not None
    )
    return _PlacedPulseRun(tuple(placed), boundary_ns, end)


def _place_run(
    blocks: tuple[PulseBlock, ...], boundary_ns: float, mode: str
) -> _PlacedPulseRun:
    explicit = [block.start_ns is not None for block in blocks]
    if any(explicit):
        return _place_explicit_run(blocks, boundary_ns)
    return _schedule_run(blocks, boundary_ns, mode)


T = TypeVar("T")


def execute_with_boundaries(
    plan: Iterable[PulseBlock | MeasurementStep | ResetStep],
    execute_run: Callable[[_PlacedPulseRun], T],
    execute_boundary: Callable[[MeasurementStep | ResetStep, float], None],
    *,
    placement_mode: str = "ASAP",
) -> tuple[list[T], float]:
    """Place continuous regions and dispatch them around shared boundaries.

    A guarded pulse is always a region of its own.  This reserves its interval
    even when a later executor finds its condition false, while keeping the
    preceding and following frame/control regions independent.
    """
    results: list[T] = []
    pending: list[PulseBlock] = []
    boundary_ns = 0.0

    def flush() -> None:
        nonlocal boundary_ns
        if not pending:
            return
        run = _place_run(tuple(pending), boundary_ns, placement_mode)
        results.append(execute_run(run))
        boundary_ns = run.end_ns
        pending.clear()

    for step in plan:
        if isinstance(step, PulseBlock):
            if step.condition is not None:
                flush()
                pending.append(step)
                flush()
            else:
                pending.append(step)
        elif isinstance(step, (MeasurementStep, ResetStep)):
            flush()
            execute_boundary(step, boundary_ns)
        else:
            raise BackendValidationError("pulse plan contains an unknown boundary step")
    flush()
    return results, boundary_ns
