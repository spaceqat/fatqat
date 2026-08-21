"""Classify pulse noise and bind resolved Lindblad operators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...errors import BackendValidationError
from ...noise import (
    AmplitudeDamping,
    Depolarizing,
    Loss,
    LindbladImplementationMap,
    NoiseModel,
    NoiseSupportReport,
    PauliChannel,
    PhaseDamping,
)


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
    supports_readout_confusion: bool,
    readout_confusion_shape: tuple[int, int] | None = None,
) -> NoiseSupportReport:
    """Classify shared pulse-noise rules plus narrow family policy knobs."""
    accepted: list[str] = []
    rejected: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()

    for channel, operation in noise_model._noise_sources():
        background = operation is None
        channel_type = type(channel)
        built_in_damping = channel_type in (
            AmplitudeDamping,
            PhaseDamping,
            Depolarizing,
        )
        finite_mode = built_in_damping and channel.p is not None
        mode = None
        if built_in_damping:
            mode = "p" if finite_mode else "rate"

        invalid_amplitude_arity = channel_type is AmplitudeDamping and (
            channel.rate is not None and len(channel.rate) != local_dimension - 1
        )
        if invalid_amplitude_arity:
            mode += f"-arity-{len(channel.rate)}"

        qualifiers: list[str] = []
        if mode is not None:
            qualifiers.append(mode)
        if background:
            qualifiers.append("background")
        label = channel_type.__name__
        if qualifiers:
            label += f"({', '.join(qualifiers)})"
        if label in seen:
            continue
        seen.add(label)

        reason = None
        authored_arity = None if isinstance(channel, Loss) else channel.num_subsystems
        if isinstance(channel, Loss):
            reason = "carrier occupancy loss is not a Lindblad generator"
        elif authored_arity is not None and authored_arity != 1:
            reason = "pulse Lindblad declarations must be single-subsystem"
        elif invalid_amplitude_arity:
            reason = (
                f"local dimension {local_dimension} requires "
                f"{local_dimension - 1} damping values"
            )
        elif finite_mode:
            reason = "finite probability mode is not a pulse generator"
        elif channel_type is PauliChannel:
            reason = (
                "pulse-family policy treats PauliChannel as finite-only; a "
                "registered Lindblad implementation does not override that policy"
            )
        elif not background and not allow_operation_scoped:
            reason = "the built-in defaults accept only background generators"
        elif implementation_map.get(channel_type) is None:
            reason = "no registered Lindblad implementation"

        if reason is None:
            accepted.append(label)
        else:
            rejected.append(label)
            warnings.append(f"{label} is not supported by {backend_name}: {reason}")

    readout_confusions = noise_model._readout_confusions()
    if readout_confusions:
        reason = None
        if not supports_readout_confusion:
            reason = "this family has no readout-confusion boundary"
        elif readout_confusion_shape is not None and any(
            declaration.matrix.shape != readout_confusion_shape
            for declaration in readout_confusions
        ):
            shape = " x ".join(str(dimension) for dimension in readout_confusion_shape)
            reason = f"readout confusion must be a {shape} matrix"
        if reason is None:
            accepted.append("ReadoutConfusion")
        else:
            rejected.append("ReadoutConfusion")
            warnings.append(
                f"ReadoutConfusion is not supported by {backend_name}: {reason}"
            )

    return NoiseSupportReport(
        supported=not rejected,
        accepted_sources=tuple(accepted),
        rejected_sources=tuple(rejected),
        warnings=tuple(warnings),
    )
