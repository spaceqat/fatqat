"""Solver-free values shared by pulse authoring and emulator lowering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from .waveforms import Waveform

TIME_EPSILON = 1e-12
"""Absolute tolerance for comparisons on a model's native time axis."""


class ControlChannel:
    """Nominal base for immutable, model-family-specific control addresses."""

    __slots__ = ()


@dataclass(frozen=True)
class PulseControl:
    """Bind one symbolic control channel to an immutable waveform.

    The channel is a target-independent structural control address; the
    waveform contains no model or target information. The bound target resolves
    the address during program preparation. ``start_offset`` is measured from
    the enclosing pulse operation's local origin.

    Args:
        channel: Structural control address returned by a model factory.
        waveform: Backend-independent immutable waveform to bind.
        start_offset: Non-negative local offset from the enclosing operation's
            origin, in the model's native time unit.

    Attributes:
        channel: Original model-family control address.
        waveform: Original immutable waveform; samples are not copied here.
        start_offset: Normalized floating-point local offset.

    Raises:
        TypeError: If ``channel`` or ``waveform`` has the wrong nominal type,
            or ``start_offset`` is not a real number.
        ValueError: If ``start_offset`` is negative or non-finite.

    Examples:
        >>> from fatqat.emulator import PulseControl, TransmonModel
        >>> from fatqat.waveforms import SampledWaveform
        >>> model = TransmonModel.from_document({
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
        >>> binding = PulseControl(
        ...     model.control.drive("q0"),
        ...     SampledWaveform((0.0, 0.5), (0.0, 1.0j)),
        ...     start_offset=0.25,
        ... )
        >>> binding.start_offset
        0.25
    """

    channel: ControlChannel
    waveform: Waveform
    start_offset: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.channel, ControlChannel):
            raise TypeError("channel must be a ControlChannel")
        if not isinstance(self.waveform, Waveform):
            raise TypeError("waveform must be a Waveform")
        if isinstance(self.start_offset, bool) or not isinstance(
            self.start_offset, Real
        ):
            raise TypeError("start_offset must be a finite real number")
        offset = float(self.start_offset)
        if not math.isfinite(offset) or offset < 0.0:
            raise ValueError("start_offset must be finite and non-negative")
        object.__setattr__(self, "start_offset", offset)


__all__ = ["ControlChannel", "PulseControl", "TIME_EPSILON"]
