"""Noise simulation surface: channels, readout error, and calibration converters.

Users build a `NoiseModel` from channel descriptors
(``fq.noise.Depolarizing`` and friends, plus `ThermalRelaxation`) and
classical readout confusion matrices, then pass it to a backend via
``noise=``. Matrix backends resolve supported operation-scoped channels into
Kraus operators; pulse backends resolve supported descriptors into
collapse-operator bindings. Readout error stays classical.
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
)
from .relaxation import ThermalRelaxation
from .model import NoiseModel
from .registry import default_channel_implementation_map

__all__ = [
    "Channel",
    "ChannelImplementation",
    "ChannelImplementationMap",
    "ThermalRelaxation",
    "NoiseSupportReport",
    "Depolarizing",
    "AmplitudeDamping",
    "PhaseDamping",
    "NoiseModel",
    "default_channel_implementation_map",
]
