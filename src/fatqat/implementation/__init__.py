"""Custom matrix rules for gate-level simulators.

Use `default_matrix_implementation_map` to copy FATQAT's built-in gate rules,
or create an empty `MatrixImplementationMap` for a backend that supports only
the rules you add. Rules may be fixed matrices, callables, or configured
`MatrixImplementation` objects.
"""

from __future__ import annotations

import inspect

from .base import (
    DeviceOperands,
    FixedMatrix,
    MatrixImplementationMap,
    MatrixImplementation,
)
from .registry import default_matrix_implementation_map

__all__ = [
    "MatrixImplementation",
    "FixedMatrix",
    "MatrixImplementationMap",
    "DeviceOperands",
    "default_matrix_implementation_map",
]
