"""Solver-free values shared by pulse authoring and emulator lowering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from ._waveforms import Waveform

TIME_EPSILON = 1e-12
"""Absolute tolerance for comparisons on a model's native time axis."""


class ControlChannel:
    """Nominal base for immutable model-family control-channel addresses.

    Obtain concrete addresses from a model's ``control`` selectors. Each
    address identifies the physical resource or resources driven by a
    ``PulseControl``. The selected emulator resolves that address against its
    physical model during program preparation; ``ResourceLayout`` does not
    remap it.
    """

    __slots__ = ()


@dataclass(frozen=True)
class PulseControl:
    """Bind one physical control-channel address to an immutable waveform.

    Obtain ``channel`` from a model's ``control`` selectors. The selected
    emulator resolves it against the physical model during preparation;
    ``ResourceLayout`` does not remap it. Controls are immutable and reusable
    with compatible models. ``start_offset`` is measured from the enclosing
    pulse operation's local origin.

    Args:
        channel: Structural address returned by a model's ``control`` selector;
            identifies the physical resource or resources to drive.
        waveform: Backend-independent immutable waveform to bind.
        start_offset: Non-negative local offset from the enclosing operation's
            origin, in the model's native time unit.

    Raises:
        TypeError: If ``channel`` or ``waveform`` has the wrong nominal type,
            or ``start_offset`` is not a real number.
        ValueError: If ``start_offset`` is negative or non-finite.
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
