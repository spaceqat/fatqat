"""Shared backend utilities that are representation-agnostic."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from .._engine_index_allocation import _EngineIndexAllocation
from ..resource_layout import ResourceLayout


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


def _normalize_dict_options(
    options: dict[str, Any] | None,
    known_keys: set[str],
    config_cls: type,
    param_name: str,
    warning_noun: str,
    backend_name: str = "SimulatorBackend",
) -> Any:
    """Normalize a plain dict of options into a frozen config dataclass."""
    if options is None:
        return config_cls()
    if not isinstance(options, dict):
        raise TypeError(f"{param_name} must be a dict or None, got {type(options)!r}")
    known = {key: value for key, value in options.items() if key in known_keys}
    ignored = {key: value for key, value in options.items() if key not in known_keys}
    if ignored:
        warnings.warn(
            f"{backend_name} ignored unsupported {warning_noun} options: {ignored!r}",
            UserWarning,
            stacklevel=3,
        )
    return config_cls(**known)


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
