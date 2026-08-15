"""Numeric helpers and the one time tolerance shared across the emulator.

Deliberately dependency-free (errors and NumPy only) so every emulator module
- the model, the pulse representation, the scheduler, the realization rules,
and the solver adapter - can import it without creating a cycle. Persistence
parsing keeps its own ``superconducting._number``: that one reports a document
path and enforces strict positivity, which is a different contract from
``_finite``'s value-level check.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np

from ..._pulse_values import TIME_EPSILON as _TIME_EPSILON
from ...errors import BackendValidationError

TIME_EPSILON = _TIME_EPSILON


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    """Return ``value`` as a finite float, rejecting bools and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendValidationError(f"{name} must be a finite number")
    value = float(value)
    if not isfinite(value) or (nonnegative and value < 0):
        raise BackendValidationError(f"{name} must be finite and non-negative")
    return value


def _freeze(values: Any, *, dtype: type = complex) -> np.ndarray:
    """Return an owned, read-only copy of ``values`` as a NumPy array."""
    array = np.array(values, dtype=dtype, copy=True)
    array.flags.writeable = False
    return array
