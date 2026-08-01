"""Pulse-model binding for resolved local Lindblad operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import BackendValidationError


@dataclass(frozen=True)
class ResolvedLindbladTerm:
    """One local Lindblad operator bound to physical-model subsystems.

    ``local_operator`` acts on one model subsystem. ``model_ordinals`` selects
    every subsystem on which that same local operator is active. The matrix is
    read-only so a resolved execution plan cannot be mutated by a simulator.
    Operation scope belongs to the containing ``PulseBlock``.
    """

    local_operator: np.ndarray
    model_ordinals: tuple[int, ...]

    def __post_init__(self) -> None:
        operator = np.asarray(self.local_operator, dtype=complex)
        if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
            raise BackendValidationError("Lindblad local operator must be square")
        if operator.flags.writeable:
            operator = np.array(operator, copy=True)
        operator.flags.writeable = False
        ordinals = tuple(self.model_ordinals)
        if (
            not ordinals
            or len(set(ordinals)) != len(ordinals)
            or any(type(ordinal) is not int or ordinal < 0 for ordinal in ordinals)
        ):
            raise BackendValidationError(
                "Lindblad model ordinals must be distinct non-negative ints"
            )
        object.__setattr__(self, "local_operator", operator)
        object.__setattr__(self, "model_ordinals", ordinals)


def bind_lindblad_operators(
    local_operators: tuple[np.ndarray, ...], *, model_ordinals: tuple[int, ...]
) -> tuple[ResolvedLindbladTerm, ...]:
    """Bind reusable local collapse operators to model subsystem ordinals.

    One resolved local operator becomes one :class:`ResolvedLindbladTerm` and
    may apply to every ordinal in ``model_ordinals``. Tensor expansion remains
    the concrete adapter's responsibility.
    """
    return tuple(
        ResolvedLindbladTerm(operator, model_ordinals) for operator in local_operators
    )
