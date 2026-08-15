"""Solver-independent helpers shared by pulse-emulator adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .pulse import PhaseShift, PhaseSwap
from .value_validation import TIME_EPSILON


@dataclass(frozen=True)
class _BoundFrames:
    """Binding result for a zero-duration, virtual-frame-only region."""

    output_frames: dict[Any, float]


@dataclass(frozen=True)
class _BoundDynamics:
    """Binding result for solver dynamics and its resulting frame ledger."""

    hamiltonian: Any
    output_frames: dict[Any, float]
    collapse_operators: tuple[Any, ...] = ()


def apply_ready_actions(
    events: list[tuple[float, int, tuple[PhaseShift | PhaseSwap, ...]]],
    start_time: float,
    frames: dict[Any, float],
) -> None:
    """Apply frame events completed before a later control starts."""
    ready = [event for event in events if event[0] <= start_time + TIME_EPSILON]
    for _end, _source, actions in sorted(ready, key=lambda event: event[:2]):
        apply_actions(actions, frames)
    events[:] = [event for event in events if event not in ready]


def apply_actions(
    actions: tuple[PhaseShift | PhaseSwap, ...], frames: dict[Any, float]
) -> None:
    """Apply phase shifts/swaps to a mutable virtual-frame ledger."""
    for action in actions:
        if isinstance(action, PhaseShift):
            frames[action.frame] = frames.get(action.frame, 0.0) + action.angle_rad
        else:
            first = frames.get(action.first, 0.0)
            second = frames.get(action.second, 0.0)
            frames[action.first], frames[action.second] = second, first
