"""Superconducting transmon pulse emulator and portable calibration API.

Models create structural controls and frames; the concrete backend binds them
to fixed device topology and accepts replacement gate and Lindblad maps.
"""

from .backend import TransmonEmulator
from .calibration import TransmonCalibration, default_transmon_calibration
from .model import (
    Coupling,
    TransmonModel,
    Transmon,
    angular_rate_from_ghz,
)
from .realization import default_transmon_gate_implementation_map

__all__ = [
    "TransmonEmulator",
    "Coupling",
    "TransmonCalibration",
    "TransmonModel",
    "Transmon",
    "angular_rate_from_ghz",
    "default_transmon_calibration",
    "default_transmon_gate_implementation_map",
]
