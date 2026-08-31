"""Superconducting transmon pulse emulator and portable calibration API.

Models create structural controls and frames; the concrete backend binds them
to fixed device topology and accepts a replacement gate implementation map.
"""

from .backend import TransmonEmulator
from .calibration import TransmonCalibration, default_transmon_calibration
from .grid_reference import generate_transmon_grid_documents
from .model import TransmonModel, angular_rate_from_ghz
from .realization import default_transmon_gate_implementation_map

__all__ = [
    "TransmonEmulator",
    "TransmonCalibration",
    "TransmonModel",
    "angular_rate_from_ghz",
    "default_transmon_calibration",
    "default_transmon_gate_implementation_map",
    "generate_transmon_grid_documents",
]
