"""Strict JSON-document validation shared by emulator constructors."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from ...errors import BackendValidationError


def _fail(path: str, message: str) -> None:
    """Raise one path-qualified model/calibration validation failure."""
    raise BackendValidationError(f"{path}: {message}")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    """Require a mapping with string keys and return it unchanged."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(path, "must be an object with string keys")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    """Require an exact object schema, reporting missing and unknown keys."""
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        detail = []
        if missing:
            detail.append(f"missing {sorted(missing)!r}")
        if extra:
            detail.append(f"unknown {sorted(extra)!r}")
        _fail(path, "; ".join(detail))


def _string(value: Any, path: str) -> str:
    """Require a non-empty string persistence value."""
    if not isinstance(value, str) or not value:
        _fail(path, "must be a non-empty string")
    return value


def _version(value: Any, path: str) -> int:
    """Require a positive integer document-format version."""
    if type(value) is not int or value < 1:
        _fail(path, "must be a positive integer")
    return value


def _number(
    value: Any, path: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    """Normalize one finite numeric value with optional sign constraints."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(path, "must be a finite number")
    result = float(value)
    if not isfinite(result):
        _fail(path, "must be finite")
    if positive and result <= 0:
        _fail(path, "must be positive")
    if nonnegative and result < 0:
        _fail(path, "must be non-negative")
    return result
