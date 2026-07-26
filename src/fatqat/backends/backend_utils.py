"""Shared backend utilities that are representation-agnostic."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from .._engine_index_allocation import _EngineIndexAllocation
from ..errors import BackendValidationError
from ..resource_layout import ResourceLayout


def _validate_grid_size(grid_size: object) -> tuple[int, int]:
    """Return a validated ``(rows, columns)`` grid shape."""
    if not isinstance(grid_size, tuple):
        raise TypeError(
            "grid_size must be a tuple of two positive integers, got "
            f"{type(grid_size)!r}"
        )
    if len(grid_size) != 2:
        raise ValueError("grid_size must contain exactly two values: (rows, columns)")
    rows, cols = grid_size
    for name, value in (("grid_size[0]", rows), ("grid_size[1]", cols)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int, got {type(value)!r}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    return rows, cols


@dataclass(frozen=True)
class _LoweringContext:
    """Private per-run pairing of the resolved `ResourceLayout` and `_EngineIndexAllocation`.

    A lowering-time implementation convenience only: it is not public and it
    does not combine one program ref's device label and engine index into a
    single per-resource value (contrast the removed `BoundResource`, which
    did exactly that). Matrix lowering reads `resource_layout` to build
    `ImplementationMap` lookup keys (`device_operands`) and reads
    `engine_index_allocation` for every execution-plan index/dimension
    (`ApplyMatrixStep` targets, measurement, reset, and condition lowering).
    Both values are resolved once per run and threaded through unchanged;
    neither is re-resolved during lowering.
    """

    resource_layout: ResourceLayout
    engine_index_allocation: _EngineIndexAllocation


@dataclass(frozen=True)
class _PlanFacts:
    """Program facts needed for backend result defaults and validation.

    ``has_channel`` defaults to ``False`` so channel-free construction sites
    (and the fake-backend lowering) stay unchanged.
    """

    has_measurement: bool
    has_reset: bool
    has_channel: bool = False


def _normalize_config(
    config: dict[str, Any] | None,
    config_cls: type,
    param_name: str,
    backend_name: str = "SimulatorBackend",
) -> Any:
    """Strictly normalize a plain dictionary into a frozen config dataclass.

    Supported keys are derived from ``config_cls``'s dataclass fields, so a
    newly added configuration entry needs no registry update. Field-level
    compatibility is owned by the config dataclass's ``__post_init__``;
    subclasses are responsible for validating fields they add.
    """
    if config is None:
        return config_cls()
    if not isinstance(config, dict):
        raise TypeError(f"{param_name} must be a dict or None, got {type(config)!r}")
    supported = {field.name for field in fields(config_cls)}
    unknown = set(config) - supported
    if unknown:
        # Dict keys need not be mutually orderable (for example, ``"foo"``
        # and ``3``). Sort their representations so malformed input still
        # receives our validation error rather than a Python ``TypeError``.
        names = ", ".join(sorted(map(repr, unknown)))
        expected = ", ".join(repr(name) for name in sorted(supported)) or "no keys"
        raise BackendValidationError(
            f"{backend_name} does not support {param_name} key(s) {names}; "
            f"expected {expected}"
        )
    return config_cls(**config)


def _resolve_condition(
    condition: tuple[tuple[object, int], ...] | None,
    engine_index_allocation: _EngineIndexAllocation,
) -> tuple[tuple[int, int], ...] | None:
    """Lower a frontend condition to ``(clbit_index, value)`` AND-terms."""
    if condition is None:
        return None
    return tuple(
        (engine_index_allocation.clbit_index(ref), val) for ref, val in condition
    )
