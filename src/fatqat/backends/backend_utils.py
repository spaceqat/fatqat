"""Shared backend utilities that are representation-agnostic."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from ..layout import ResourceLayout
from ..result import _ResultConfig
from .engine_contract import _ResultRequest


@dataclass(frozen=True)
class _PlanFacts:
    """Program facts needed for backend result defaults and validation."""

    has_measurement: bool
    has_reset: bool


def _normalize_dict_options(
    options: dict[str, Any] | None,
    known_keys: set[str],
    config_cls: type,
    param_name: str,
    warning_noun: str,
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
            f"StateVectorBackend ignored unsupported {warning_noun} options: {ignored!r}",
            UserWarning,
            stacklevel=3,
        )
    return config_cls(**known)


def _resolve_condition(
    condition: tuple[tuple[object, int], ...] | None,
    layout: ResourceLayout,
) -> tuple[tuple[int, int], ...] | None:
    """Lower a frontend condition to ``(clbit_index, value)`` AND-terms."""
    if condition is None:
        return None
    return tuple((layout.clbit_index(ref), val) for ref, val in condition)


def _resolve_result_request(config: _ResultConfig, facts: _PlanFacts) -> _ResultRequest:
    """Resolve default result fields from config and lowered program facts."""
    stochastic = facts.has_measurement or facts.has_reset
    counts = config.counts if config.counts is not None else facts.has_measurement
    statevector = config.statevector
    if statevector is None:
        statevector = not stochastic
    return _ResultRequest(counts=counts, statevector=statevector)
