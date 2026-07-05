"""Class-keyed matrix implementations and the flat payload the engine consumes."""

from __future__ import annotations

import inspect

from .base import (
    ApplyMatrixStep,
    FixedMatrix,
    MatrixImplementation,
    MatrixImplementationMap,
    _CallableMatrixImplementation,
    _DimMatrix,
    _callable_wants_targets,
    _require_fixed_arity,
    _resolve_operation_class,
    _validate_square_matrix,
    _wrap_rule,
)
from .matrices import (
    _cphase,
    _phase,
    _rx,
    _ry,
    _rz,
    _clock_rule,
    _shift_rule,
    clock_matrix,
    shift_matrix,
    sum_matrix,
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
]
