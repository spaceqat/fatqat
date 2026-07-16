"""Matrix-family backend package: validate, execute, assemble Result, return Job."""

from __future__ import annotations

from .fake_superconducting import FakeSuperconducting4x4Backend
from .simulator_backend import SimulatorBackend

# Re-exported for white-box tests to import directly; not part of the public
# API (a Program is lowered to these internally, users never construct them).
from .steps import ApplyChannelStep, ApplyMatrixStep, MeasurementStep, ResetStep

__all__ = ["FakeSuperconducting4x4Backend", "SimulatorBackend"]
