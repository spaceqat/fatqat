"""Matrix, channel, and custom-operation rules for gate-level simulators.

Use `default_matrix_implementation_map` to copy FATQAT's built-in gate and channel rules,
or create an empty `MatrixImplementationMap` for a backend that supports only
the rules you add. Rules may be fixed matrices, callables, or configured
`MatrixImplementation` objects. CustomOperation descriptions use
`CustomOperationImplementation` in the same map.
"""

from __future__ import annotations

import inspect

from .matrix_map import (
    DeviceOperands,
    FixedMatrix,
    MatrixImplementationMap,
    MatrixImplementation,
)
from .custom import CustomOperationImplementation
from .registry import default_matrix_implementation_map

__all__ = [
    "CustomOperationImplementation",
    "MatrixImplementation",
    "FixedMatrix",
    "MatrixImplementationMap",
    "DeviceOperands",
    "default_matrix_implementation_map",
]
