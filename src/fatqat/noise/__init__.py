"""Noise models and built-in noise sources for FATQAT backends.

Build a `NoiseModel` from quantum channels, carrier `Loss`, and classical
`ReadoutConfusion`, then pass it to a backend through ``noise=``. Simulator
behavior uses supported probability forms on matching operations. Emulator
behavior uses supported rate or time forms as local Lindblad operators over
elapsed time. FATQAT never converts a probability into a rate implicitly.

`ChannelImplementationMap` and `LindbladImplementationMap` are optional
extension APIs for custom noise types or backend-specific behavior.
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
    PauliChannel,
    PhaseDamping,
)
from .relaxation import ThermalRelaxation
from .lindblad import (
    LindbladImplementationMap,
    default_lindblad_implementation_map,
)
from .model import NoiseModel
from .registry import default_channel_implementation_map
from .loss import Loss
from .readout import ReadoutConfusion

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
    "PauliChannel",
    "NoiseModel",
    "default_channel_implementation_map",
    "default_lindblad_implementation_map",
    "Loss",
    "ReadoutConfusion",
]
