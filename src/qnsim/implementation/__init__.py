"""Class-keyed matrix implementations and the flat payload the engine consumes."""

from __future__ import annotations

import inspect

from .base import ApplyMatrixStep, FixedMatrix, MatrixImplementation, MatrixImplementationMap
from .matrices import (
    cclock_matrix,
    clock_matrix,
    fourier_matrix,
    fourierdg_matrix,
    shift_matrix,
    subspace_rx_matrix,
    subspace_ry_matrix,
    subspace_rz_matrix,
    sum_matrix,
    swap_levels_matrix,
)
from .registry import default_matrix_implementation_map

__all__ = [
    "MatrixImplementation",
    "FixedMatrix",
    "ApplyMatrixStep",
    "MatrixImplementationMap",
    "default_matrix_implementation_map",
    "shift_matrix",
    "clock_matrix",
    "sum_matrix",
    "swap_levels_matrix",
    "fourier_matrix",
    "fourierdg_matrix",
    "subspace_rx_matrix",
    "subspace_ry_matrix",
    "subspace_rz_matrix",
    "cclock_matrix",
]
