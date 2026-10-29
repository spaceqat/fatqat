"""Noise models and built-in noise sources for FATQAT backends.

Build a `NoiseModel` from quantum channels, carrier `Loss`, and classical
`ReadoutConfusion`, then pass it to a backend through ``noise=``. Simulator
behavior uses supported probability forms on matching operations. Emulator
behavior uses supported rate or time forms as local Lindblad operators over
elapsed time. FATQAT never converts a probability into a rate implicitly.

`fatqat.implementation.MatrixImplementationMap` registers custom simulator channel
rules. Pulse-emulator Lindblad realizations are owned by each emulator family.
"""

from .base import (
    Channel,
    ChannelImplementation,
)
from .catalog import (
    AmplitudeDamping,
    Depolarizing,
    PauliChannel,
    PhaseDamping,
)
from .relaxation import ThermalRelaxation
from .transition_relaxation import TransitionRelaxation
from .model import NoiseModel
from .loss import Loss
from .readout import ReadoutConfusion

__all__ = [
    "Channel",
    "ChannelImplementation",
    "ThermalRelaxation",
    "Depolarizing",
    "AmplitudeDamping",
    "TransitionRelaxation",
    "PhaseDamping",
    "PauliChannel",
    "NoiseModel",
    "Loss",
    "ReadoutConfusion",
]
