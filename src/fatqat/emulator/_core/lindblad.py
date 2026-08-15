"""Classify pulse noise and bind resolved Lindblad operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...errors import BackendValidationError
from ...noise import (
    AmplitudeDamping,
    LindbladImplementationMap,
    NoiseModel,
    NoiseSupportReport,
)

_NO_RATE = object()


@dataclass(frozen=True)
class ResolvedLindbladTerm:
    """One local Lindblad operator bound to target subsystems.

    ``local_operator`` acts on one target subsystem. ``engine_indices`` selects
    every subsystem on which that same local operator is active. The matrix is
    read-only so a resolved execution plan cannot be mutated by a simulator.
    Operation scope belongs to the containing ``PulseBlock``.
    """

    local_operator: np.ndarray
    engine_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        operator = np.asarray(self.local_operator, dtype=complex)
        if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
            raise BackendValidationError("Lindblad local operator must be square")
        operator = np.array(operator, dtype=complex, copy=True)
        operator.flags.writeable = False
        indices = tuple(self.engine_indices)
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(type(index) is not int or index < 0 for index in indices)
        ):
            raise BackendValidationError(
                "Lindblad engine indices must be distinct non-negative ints"
            )
        object.__setattr__(self, "local_operator", operator)
        object.__setattr__(self, "engine_indices", indices)


def bind_lindblad_operators(
    local_operators: tuple[np.ndarray, ...], *, engine_indices: tuple[int, ...]
) -> tuple[ResolvedLindbladTerm, ...]:
    """Bind reusable local collapse operators to target subsystem ordinals.

    One resolved local operator becomes one :class:`ResolvedLindbladTerm` and
    may apply to every index in ``engine_indices``. Tensor expansion remains
    the concrete adapter's responsibility.
    """
    return tuple(
        ResolvedLindbladTerm(operator, engine_indices) for operator in local_operators
    )


def _classify_lindblad_noise(
    noise_model: NoiseModel,
    implementation_map: LindbladImplementationMap,
    *,
    local_dimension: int,
    backend_name: str,
    allow_operation_scoped: bool = True,
    supports_readout_error: bool,
    readout_error_shape: tuple[int, int] | None = None,
) -> NoiseSupportReport:
    """Classify shared pulse-noise rules plus narrow family policy knobs."""
    accepted: list[str] = []
    rejected: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for channel, operation in noise_model.channel_registrations():
        always_on = operation is None
        rate = getattr(channel, "rate", _NO_RATE)
        mode: str | None = None
        damping_values = None
        if rate is not _NO_RATE:
            mode = "rate" if rate is not None else "p"
            damping_values = rate if rate is not None else getattr(channel, "p")

        invalid_amplitude_arity = (
            isinstance(channel, AmplitudeDamping)
            and len(damping_values) != local_dimension - 1
        )
        if invalid_amplitude_arity:
            assert mode is not None
            mode += f"-arity-{len(damping_values)}"

        qualifiers: list[str] = []
        if mode is not None:
            qualifiers.append(mode)
        if always_on:
            qualifiers.append("always-on")
        label = type(channel).__name__
        if qualifiers:
            label += f"({', '.join(qualifiers)})"
        if label in seen:
            continue
        seen.add(label)

        reason = None
        if invalid_amplitude_arity:
            reason = (
                f"local dimension {local_dimension} requires "
                f"{local_dimension - 1} damping values"
            )
        elif always_on and rate is None:
            reason = "always-on damping requires rate mode"
        elif not always_on and not allow_operation_scoped:
            reason = "the built-in defaults accept only always-on rate damping"
        elif implementation_map.get(type(channel)) is None:
            reason = "no registered Lindblad implementation"

        if reason is None:
            accepted.append(label)
        else:
            rejected.append(label)
            warnings.append(f"{label} is not supported by {backend_name}: {reason}")

    if noise_model.has_readout_error():
        reason = None
        if not supports_readout_error:
            reason = "this family has no readout-confusion boundary"
        elif readout_error_shape is not None and any(
            matrix.shape != readout_error_shape
            for _selector, matrix in noise_model._readout_errors
        ):
            shape = " x ".join(str(dimension) for dimension in readout_error_shape)
            reason = f"readout confusion must be a {shape} matrix"
        if reason is None:
            accepted.append("readout_error")
        else:
            rejected.append("readout_error")
            warnings.append(
                f"readout_error is not supported by {backend_name}: {reason}"
            )

    return NoiseSupportReport(
        supported=not rejected,
        accepted_sources=tuple(accepted),
        rejected_sources=tuple(rejected),
        warnings=tuple(warnings),
    )
