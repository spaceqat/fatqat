"""Default channel implementation map: wires built-in channels to their rules."""

from __future__ import annotations

from .base import ChannelImplementationMap
from .catalog import (
    AmplitudeDamping,
    Depolarizing,
    PauliChannel,
    PhaseDamping,
    amplitude_damping_rule,
    depolarizing_rule,
    pauli_channel_rule,
    phase_damping_rule,
)
from .transition_relaxation import TransitionRelaxation, transition_relaxation_rule


def default_channel_implementation_map() -> ChannelImplementationMap:
    """Return a fresh map for FATQAT's built-in simulator channels.

    The map registers `Depolarizing`, `PauliChannel`, `AmplitudeDamping`,
    `PhaseDamping`, and `TransitionRelaxation`. Registration is by exact channel
    type. Whether one instance
    can run still depends on its parameter form, scope, targets, and simulator
    method; for example, matrix simulators do not support rate-form damping.

    Each call returns an independent registration container.

    Returns:
        A new map containing the five built-in simulator channel rules.
    """
    channel_map = ChannelImplementationMap()
    channel_map.add(Depolarizing, depolarizing_rule)
    channel_map.add(AmplitudeDamping, amplitude_damping_rule)
    channel_map.add(PhaseDamping, phase_damping_rule)
    channel_map.add(PauliChannel, pauli_channel_rule)
    channel_map.add(TransitionRelaxation, transition_relaxation_rule)
    return channel_map
