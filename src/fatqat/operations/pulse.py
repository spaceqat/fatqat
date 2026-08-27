"""Backend-independent direct pulse-operation authoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import ClassVar

from .._pulse_values import PulseControl, TIME_EPSILON
from .base import Operation


@dataclass(frozen=True)
class PulseOperation(Operation):
    """Run concurrent channel-addressed direct controls for one time block.

    ``PulseOperation`` does not take a separate ``targets`` argument. Each
    ``fatqat.emulator.PulseControl`` channel identifies the physical resource
    or resources it drives. Add the block with
    ``program.add(operation)`` and do not pass a ``targets`` value.

    During program preparation, the selected pulse emulator resolves every
    channel address against its physical model. Those bindings determine
    scheduling resource claims and physical engine indices. For a
    ``TransmonModel``, drive and detuning channels bind one declared subsystem;
    an exchange channel binds two subsystems and their declared coupling. The
    emulator also validates model-family compatibility, resource names,
    duration, waveform, amplitude, and concurrency constraints at this stage.

    Ordinary operations receive logical ``RegisterRef`` operands, which
    ``ResourceLayout`` maps to device resources. Direct channel addresses bind
    against the emulator's physical model and are not remapped by
    ``ResourceLayout``. A control may therefore address an otherwise
    unreferenced modeled resource. Channel addresses and controls are reusable
    immutable values, not handles owned by one model instance. The matrix
    ``Simulator`` does not support direct pulse operations.

    ``program.add(operation, condition=...)`` may guard the block. A false
    condition disables the block's controls and condition-scoped generators,
    but the full block duration still elapses under model drift and background
    Lindblad sources. Operation-scoped noise cannot be attached to a direct
    pulse operation.

    Controls begin at their own ``start_offset`` and may finish before the
    block ends. Each must finish no later than ``duration``; construction
    allows the shared endpoint to exceed it only within the package's absolute
    time tolerance of ``1e-12``. Controls in one block are concurrent, and the
    same channel may occur only once. Sum same-channel samples before creating
    the operation.

    Every direct-control block has positive duration and at least one control.
    Omit the operation when no time should elapse.

    Args:
        duration: Positive finite real block duration in the selected model's
            native time unit. Booleans are rejected. Stored as ``float``.
        controls: Non-empty iterable of ``fatqat.emulator.PulseControl``
            values. The iterable is consumed once and copied into a tuple;
            the immutable control values themselves are retained.

    Raises:
        TypeError: If ``duration`` is not a real number, ``controls`` is not
            iterable, or an element is not a ``PulseControl``.
        ValueError: If ``duration`` is non-finite or not positive, ``controls``
            is empty, a channel occurs more than once, or a control ends after
            the block tolerance.
    """

    duration: float
    controls: tuple[PulseControl, ...]

    name: ClassVar[str] = "PulseOperation"
    num_subsystems: ClassVar[int] = 0
    _min_subsystems: ClassVar[int] = 0
    _is_direct_control: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if isinstance(self.duration, bool) or not isinstance(self.duration, Real):
            raise TypeError("duration must be a finite real number")
        duration = float(self.duration)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration must be finite and greater than zero")
        controls = tuple(self.controls)
        if not controls:
            raise ValueError("controls must contain at least one PulseControl")
        seen_channels = set()
        for control in controls:
            if not isinstance(control, PulseControl):
                raise TypeError("controls must contain only PulseControl values")
            if control.channel in seen_channels:
                raise ValueError("PulseOperation cannot sum controls on one channel")
            seen_channels.add(control.channel)
            if (
                control.start_offset + control.waveform.duration
                > duration + TIME_EPSILON
            ):
                raise ValueError("control extends beyond PulseOperation duration")
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "controls", controls)


__all__ = ["PulseOperation"]
