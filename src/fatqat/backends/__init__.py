"""Matrix-family backend package: validate, execute, assemble Result, return Job."""

from __future__ import annotations

from .density_matrix_backend import DensityMatrixBackend
from .fake_superconducting import FakeSuperconducting4x4Backend
from .statevector_backend import StateVectorBackend

# Re-exported for white-box tests to import directly; not part of the public
# API (a Program is lowered to these internally, users never construct them).
from .steps import ApplyMatrixStep, MeasurementStep, ResetStep

__all__ = ["DensityMatrixBackend", "FakeSuperconducting4x4Backend", "StateVectorBackend"]
