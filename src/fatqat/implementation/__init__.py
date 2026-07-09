"""Class-keyed matrix implementations."""

from __future__ import annotations

import inspect

from .base import FixedMatrix, MatrixImplementation, MatrixImplementationMap, TargetKey
from .registry import default_matrix_implementation_map

__all__ = [
    "MatrixImplementation",
    "FixedMatrix",
    "MatrixImplementationMap",
    "TargetKey",
    "default_matrix_implementation_map",
]
