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
    """One concurrent, model-neutral direct-control time block.

    Each model-defined structural control address identifies its physical
    target, so this operation always has zero ordinary program targets. The
    selected backend resolves the addresses, derives resource claims, and
    validates the waveform representation before constructing its private
    execution block.

    Args:
        duration: Positive finite block duration in the selected model's
            native time unit.
        controls: Non-empty iterable of concurrent
            :class:`~fatqat.emulator.PulseControl` bindings. A channel may
            appear only once; sum same-channel samples before construction.

    Attributes:
        duration: Normalized floating-point block duration.
        controls: Immutable tuple of concurrent control bindings.

    Raises:
        TypeError: If ``duration`` is not real or a control is not a
            :class:`~fatqat.emulator.PulseControl`.
        ValueError: If duration is non-positive/non-finite, controls are
            empty or duplicate a channel, or a control extends past the block.

    Examples:
        >>> from fatqat import ops
        >>> from fatqat.emulator import PulseControl, TransmonModel
        >>> from fatqat.waveforms import SampledWaveform
        >>> model = TransmonModel({
        ...     "format": {"id": "sc.transmon_exchange", "version": 1},
        ...     "model": {"id": "doc-example", "revision": "1"},
        ...     "system": {
        ...         "subsystem_type": "transmon",
        ...         "subsystems": ["q0"],
        ...         "control_edges": [],
        ...     },
        ...     "units": {"frequency": "GHz", "anharmonicity": "GHz"},
        ...     "parameters": {"subsystems": {
        ...         "q0": {"frequency": 5.0, "anharmonicity": -0.2},
        ...     }},
        ... })
        >>> control = PulseControl(
        ...     model.drive_control("q0"),
        ...     SampledWaveform((0.0, 1.0), (0.0, 0.2j)),
        ... )
        >>> operation = ops.PulseOperation(1.0, (control,))
        >>> operation.num_targets
        0
    """

    duration: float
    controls: tuple[PulseControl, ...]

    name: ClassVar[str] = "PulseOperation"
    _num_subsystems: ClassVar[int] = 0
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
