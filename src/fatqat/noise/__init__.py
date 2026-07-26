"""Noise simulation surface: channels, readout error, and calibration converters.

Users build a `NoiseModel` from catalog descriptors (``fq.noise.Depolarizing``
and friends, or `relaxation_channels` for T1/T2-derived rates, all in the catalog) plus classical
readout confusion matrices, and pass it to a backend via ``noise=``. The
backend resolves each attached channel into Kraus operators through its
`ChannelImplementationMap` at lowering time; readout error stays classical
and only resamples reported measurement values.
"""

from .base import (
    Channel,
    ChannelImplementation,
    ChannelImplementationMap,
    NoiseSupportReport,
)
from .catalog import (
    AmplitudeDamping,
    Depolarizing,
    PhaseDamping,
    relaxation_channels,
)
from .continuous import ContinuousNoise, ThermalRelaxation
from .model import NoiseModel
from .registry import default_channel_implementation_map

__all__ = [
    "Channel",
    "ChannelImplementation",
    "ChannelImplementationMap",
    "ContinuousNoise",
    "ThermalRelaxation",
    "NoiseSupportReport",
    "Depolarizing",
    "AmplitudeDamping",
    "PhaseDamping",
    "NoiseModel",
    "default_channel_implementation_map",
    "relaxation_channels",
]
