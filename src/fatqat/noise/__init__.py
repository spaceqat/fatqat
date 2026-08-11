"""Noise simulation surface: channels, readout error, and calibration converters.

Users build a `NoiseModel` from channel descriptors
(``fq.noise.Depolarizing`` and friends, plus `ThermalRelaxation`) and
classical readout confusion matrices, then pass it to a backend via
``noise=``. Matrix backends resolve supported operation-scoped channels into
Kraus operators; continuous simulators resolve supported descriptors into
local Lindblad operators. Readout error stays classical.
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
from .lindblad import (
    LindbladImplementationMap,
    default_lindblad_implementation_map,
)
from .model import NoiseModel
from .registry import default_channel_implementation_map
from .loss import AtomLoss

__all__ = [
    "Channel",
    "ChannelImplementation",
    "ChannelImplementationMap",
    "LindbladImplementationMap",
    "ThermalRelaxation",
    "NoiseSupportReport",
    "Depolarizing",
    "AmplitudeDamping",
    "PhaseDamping",
    "NoiseModel",
    "default_channel_implementation_map",
    "default_lindblad_implementation_map",
    "AtomLoss",
]
