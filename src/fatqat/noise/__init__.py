"""Channel-representable noise: descriptors, resolution rules, and the model.

Public surface for noise simulation on the matrix backend family. Users
build a `NoiseModel` from catalog descriptors (``fq.noise.Depolarizing`` and
friends) and pass it to a backend via ``noise=``; the backend resolves each
attached descriptor into Kraus operators through its
`ChannelImplementationMap` at lowering time.
"""

from .base import (
    Channel,
    ChannelImplementation,
    ChannelImplementationMap,
    NoiseSupportReport,
)
from .catalog import AmplitudeDamping, Depolarizing, PhaseDamping
from .model import NoiseModel
from .registry import default_channel_implementation_map

__all__ = [
    "Channel",
    "ChannelImplementation",
    "ChannelImplementationMap",
    "NoiseSupportReport",
    "Depolarizing",
    "AmplitudeDamping",
    "PhaseDamping",
    "NoiseModel",
    "default_channel_implementation_map",
]
