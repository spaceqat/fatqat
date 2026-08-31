"""Public pulse-control values shared by emulator families."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from ._waveforms import Waveform

TIME_EPSILON = 1e-12
"""Absolute tolerance for comparisons on a model's native time axis."""


class ControlChannel:
    """Base type for a control channel returned by an emulator model.

    Use the selectors under ``model.control`` to obtain a channel; do not
    construct this base class directly. A channel names the physical resource
    driven by a ``PulseControl`` and is not changed by ``ResourceLayout``.
    """

    __slots__ = ()


@dataclass(frozen=True)
class PulseControl:
    """Assign a sampled waveform to one physical control channel.

    Obtain ``channel`` from a compatible model's ``control`` selectors.
    Channels may be reused with another model of the same family when that
    model contains the named resources. The emulator checks the channel,
    waveform values, and model-specific limits when the control is used.

    Args:
        channel: Physical channel returned by ``model.control``.
        waveform: `Waveform` applied to the channel. Built-in pulse
            emulators currently accept only ``SampledWaveform``.
        start_offset: Finite non-negative delay from the start of the enclosing
            pulse block, in the model's time unit. The default is ``0.0``.

    Raises:
        TypeError: If ``channel`` or ``waveform`` has an unsupported type, or
            ``start_offset`` is not a real number. Booleans are rejected.
        ValueError: If ``start_offset`` is negative or non-finite.

    Examples:
        >>> import fatqat as fq
        >>> model = fq.emulator.TransmonModel.from_document(
        ...     fq.emulator.load_model_document("transmon.reference")
        ... )
        >>> control = fq.emulator.PulseControl(
        ...     model.control.drive("q0"),
        ...     fq.emulator.SampledWaveform(
        ...         (0.0, 0.5, 1.0), (0.0, 1.0j, 0.0)
        ...     ),
        ...     start_offset=0.25,
        ... )
        >>> control.start_offset
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
