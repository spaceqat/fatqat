"""Default channel implementation map: wires built-in channels to their rules."""

from __future__ import annotations

from .base import ChannelImplementationMap
from .catalog import (
    AmplitudeDamping,
    Depolarizing,
    PhaseDamping,
    amplitude_damping_rule,
    depolarizing_rule,
    phase_damping_rule,
)


def default_channel_implementation_map() -> ChannelImplementationMap:
    """Build the default channel implementation map for the matrix family."""
    channel_map = ChannelImplementationMap()
    channel_map.register(Depolarizing, depolarizing_rule)
    channel_map.register(AmplitudeDamping, amplitude_damping_rule)
    channel_map.register(PhaseDamping, phase_damping_rule)
    return channel_map
