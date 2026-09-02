"""Explicit transition-relaxation channels and their finite realization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from numbers import Number
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from ..errors import BackendValidationError
from ..registers import RegisterRef
from .base import Channel
from .catalog import _require_channel_arity, _require_probability, _require_rate


@dataclass(frozen=True, kw_only=True, init=False)
class TransitionRelaxation(Channel):
    """Relaxation through one explicitly authored transition operator.

    ``coefficients`` maps ``(source, destination)`` level pairs to the
    corresponding coefficient in
    ``A = sum coefficient * |destination><source|``. Exactly one of ``p`` or
    ``rate`` selects an elementary finite channel or a Lindblad generator.
    Multiple terms in one descriptor form one coherent jump operator.

    Level bounds and probability-form completeness are checked when a backend
    resolves the descriptor for a concrete physical dimension.

    Args:
        coefficients: Nonempty mapping from ``(source, destination)`` level
            pairs to finite, nonzero real or complex coefficients. Each pair
            must contain distinct nonnegative integer levels.
        p: Dimensionless finite-channel jump strength in ``[0, 1]``.
        rate: Nonnegative Lindblad rate in the inverse of the backend's time
            unit.

    Raises:
        TypeError: If coefficients is not a mapping.
        ValueError: If exactly one of p and rate is not supplied, a strength is
            outside its accepted range, a level pair is invalid, or a
            coefficient is zero or non-finite.

    Attributes:
        p: The finite-channel jump strength, or ``None`` in rate mode.
        rate: The Lindblad rate, or ``None`` in probability mode.
        coefficients: The validated coefficients as a read-only mapping.

    Examples:
        >>> import fatqat as fq
        >>> relaxation = fq.noise.TransitionRelaxation(
        ...     p=0.1, coefficients={(2, 0): 1.0}
        ... )
        >>> relaxation.p
        0.1
    """

    num_subsystems: ClassVar[int | None] = 1
    p: float | None
    rate: float | None
    _coefficient_items: tuple[tuple[tuple[int, int], complex], ...] = field(repr=False)

    def __init__(
        self,
        *,
        coefficients: Mapping[tuple[int, int], complex],
        p: float | None = None,
        rate: float | None = None,
    ) -> None:
        if (p is None) == (rate is None):
            raise ValueError("TransitionRelaxation requires exactly one of p or rate")
        if p is not None:
            _require_probability(p, "TransitionRelaxation.p")
        else:
            _require_rate(rate, "TransitionRelaxation.rate")
        if not isinstance(coefficients, Mapping):
            raise TypeError("TransitionRelaxation.coefficients must be a mapping")
        if not coefficients:
            raise ValueError(
                "TransitionRelaxation.coefficients requires at least one transition"
            )

        normalized: list[tuple[tuple[int, int], complex]] = []
        for key, value in coefficients.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError(
                    "TransitionRelaxation coefficient keys must be "
                    "(source, destination) pairs"
                )
            source, destination = key
            if any(
                not isinstance(level, int) or isinstance(level, bool) or level < 0
                for level in key
            ):
                raise ValueError(
                    "TransitionRelaxation levels must be non-negative integers"
                )
            if source == destination:
                raise ValueError(
                    "TransitionRelaxation source and destination must differ"
                )
            if isinstance(value, bool) or not isinstance(value, Number):
                raise ValueError(
                    "TransitionRelaxation coefficients must be finite nonzero "
                    "numbers"
                )
            coefficient = complex(value)
            if (
                not isfinite(coefficient.real)
                or not isfinite(coefficient.imag)
                or coefficient == 0.0
            ):
                raise ValueError(
                    "TransitionRelaxation coefficients must be finite and nonzero"
                )
            normalized.append(((source, destination), coefficient))

        object.__setattr__(self, "p", p)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(
            self,
            "_coefficient_items",
            tuple(sorted(normalized, key=lambda item: item[0])),
        )

    @property
    def coefficients(self) -> Mapping[tuple[int, int], complex]:
        """Return the normalized transition coefficients as a read-only mapping."""
        return MappingProxyType(dict(self._coefficient_items))

    def __repr__(self) -> str:
        strength = f"p={self.p!r}" if self.p is not None else f"rate={self.rate!r}"
        return (
            f"{type(self).__name__}({strength}, "
            f"coefficients={dict(self._coefficient_items)!r})"
        )


def transition_operator(
    channel: TransitionRelaxation, physical_dimension: int
) -> np.ndarray:
    """Build one local transition operator for a known physical dimension."""
    operator = np.zeros((physical_dimension, physical_dimension), dtype=complex)
    for (source, destination), coefficient in channel._coefficient_items:
        if source >= physical_dimension or destination >= physical_dimension:
            raise BackendValidationError(
                "TransitionRelaxation transition "
                f"({source}, {destination}) is outside physical dimension "
                f"{physical_dimension}"
            )
        operator[destination, source] = coefficient
    return operator


_TRANSITION_COMPLETENESS_ATOL = 1e-12


def transition_relaxation_rule(
    channel: TransitionRelaxation,
    *,
    targets: tuple[RegisterRef, ...],
) -> tuple[np.ndarray, ...]:
    """Resolve one probability-form transition into its finite Kraus channel."""
    _require_channel_arity(channel, targets, "TransitionRelaxation")
    if channel.p is None:
        raise BackendValidationError(
            "TransitionRelaxation in rate mode has no matrix-backend Kraus "
            "implementation; use p mode or a pulse backend"
        )
    dimension = targets[0].register.dim
    jump = np.sqrt(channel.p) * transition_operator(channel, dimension)
    remainder = np.eye(dimension, dtype=complex) - jump.conj().T @ jump
    remainder = (remainder + remainder.conj().T) / 2
    if not np.all(np.isfinite(remainder)):
        raise BackendValidationError(
            "TransitionRelaxation resolved to a non-finite completeness matrix"
        )
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(remainder)
    except np.linalg.LinAlgError as exc:
        raise BackendValidationError(
            "TransitionRelaxation completeness matrix could not be diagonalized"
        ) from exc
    if eigenvalues[0] < -_TRANSITION_COMPLETENESS_ATOL:
        raise BackendValidationError(
            "TransitionRelaxation probability and coefficients do not define "
            "a trace-preserving channel"
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    no_jump = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.conj().T
    return no_jump, jump
