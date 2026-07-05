"""Statevector backend package: validate, execute, assemble Result, return Job."""

from __future__ import annotations

import numpy as np

from .parallel import _planned_workers
from .statevector import (
    MeasurementStep,
    ResetStep,
    StateVectorBackend,
    _BackendConfig,
    _ResultRequest,
)

__all__ = ["StateVectorBackend"]
