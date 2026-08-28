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
    """Define a timed block of concurrent channel-addressed controls.

    Unlike an ordinary operation, which takes logical targets when it is added
    to a program, a ``PulseOperation`` already identifies its physical targets
    through its ``PulseControl.channel`` values. Add it with
    ``program.add(operation)`` and do not pass targets. ``ResourceLayout`` does
    not remap its channels.

    Every block has positive duration and at least one control. Controls start
    at their own ``start_offset``, run concurrently, and must finish within the
    block. A channel may occur only once; combine same-channel samples before
    construction.

    A pulse emulator checks channel compatibility, waveform limits, and
    concurrent resource use when the program is run. Matrix simulators reject
    direct pulse operations. Operation-scoped noise cannot be attached to a
    ``PulseOperation``; background pulse noise still applies.

    On pulse emulators that support conditions, a false ``condition=``
    disables the controls but the full duration still elapses under model
    drift and background Lindblad noise.

    Args:
        duration: Positive finite block duration in the model's time unit.
            Booleans are rejected.
        controls: Non-empty iterable of ``fatqat.emulator.PulseControl``
            values.

    Raises:
        TypeError: If ``duration`` is not real, ``controls`` is not iterable,
            or an element is not a ``PulseControl``.
        ValueError: If ``duration`` is non-finite or not positive, ``controls``
            is empty, a channel is repeated, or a control extends beyond the
            block.

    Examples:
        >>> import fatqat as fq
        >>> import fatqat.operations as ops
        >>> model = fq.emulator.TransmonModel.from_document(
        ...     fq.emulator.load_model_document("transmon.reference")
        ... )
        >>> waveform = fq.emulator.SampledWaveform(
        ...     (0.0, 10.0, 20.0), (0.0, 0.02, 0.0)
        ... )
        >>> control = fq.emulator.PulseControl(model.control.drive("q0"), waveform)
        >>> operation = ops.PulseOperation(duration=20.0, controls=(control,))
        >>> program = fq.Program(1)
        >>> program.add(operation)
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
