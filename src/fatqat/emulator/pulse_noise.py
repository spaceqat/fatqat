"""Private engine-facing pulse noise bindings and pulse channel coverage.

`ResolvedPulseNoise` is the pulse family's analogue of a resolved Kraus
channel: it carries only model-plane target indices, resolved rates, and an
optional condition - never user selectors or calibration objects. The same
binding representation is used for always-on and operation-scoped noise;
placement determines whether its coefficient is constant or block-windowed.

`supported_pulse_noise_types()` is the source-descriptor coverage declaration
shared by lowering and `PulseBackend.validate_noise()`. Source descriptors
resolve here into primitive amplitude/phase bindings; qutip construction
therefore needs rules only for those primitives and never inspects the
original descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import BackendValidationError, UnsupportedOperationError
from ..noise import AmplitudeDamping, Channel, PhaseDamping, ThermalRelaxation

_SUPPORTED_PULSE_CHANNEL_TYPES: frozenset[type[Channel]] = frozenset(
    {AmplitudeDamping, PhaseDamping, ThermalRelaxation}
)


def supported_pulse_noise_types() -> frozenset[type[Channel]]:
    """Return every channel descriptor type the pulse backend can lower."""
    return _SUPPORTED_PULSE_CHANNEL_TYPES


@dataclass(frozen=True)
class ResolvedPulseNoise:
    """Engine-facing collapse-channel binding for one damping mechanism.

    Attributes:
        channel_type: The originating `Channel` subclass, used only to
            select a collapse construction (see `supported_pulse_noise_types`),
            never to inspect the descriptor's own parameters.
        target_indices: Physical-model subsystem ordinals the collapse
            operator(s) act on.
        rate: The channel's already-duration-resolved rate: a per-transition
            tuple for `AmplitudeDamping`, a scalar for `PhaseDamping`.
        condition: The realized block's condition, carried alongside so a
            disabled conditional block's noise is skipped the same way its
            controls are.
    """

    channel_type: type[Channel]
    target_indices: tuple[int, ...]
    rate: tuple[float, ...] | float
    condition: tuple[tuple[int, int], ...] | None = None


def resolve_pulse_noise(
    channel: Channel,
    *,
    target_indices: tuple[int, ...],
    physical_dimension: int,
    duration: float | None,
    condition: tuple[tuple[int, int], ...] | None = None,
) -> tuple[ResolvedPulseNoise, ...]:
    """Resolve one source descriptor into primitive pulse collapse bindings.

    ``duration=None`` denotes always-on scope and therefore requires a rate
    representation. A finite duration denotes one operation block and also
    permits probability-mode damping, which is converted at this boundary.
    Compound `ThermalRelaxation` resolves into amplitude and optional pure
    dephasing bindings, so the adapter only implements primitive mechanisms.
    """
    channel_type = type(channel)
    if channel_type not in _SUPPORTED_PULSE_CHANNEL_TYPES:
        raise UnsupportedOperationError(
            f"{channel_type.__name__} has no pulse channel implementation "
            "on this backend"
        )

    if isinstance(channel, ThermalRelaxation):
        amplitude = ResolvedPulseNoise(
            channel_type=AmplitudeDamping,
            target_indices=target_indices,
            rate=tuple(
                level * channel.amplitude_rate for level in range(1, physical_dimension)
            ),
            condition=condition,
        )
        if channel.pure_dephasing_rate == 0.0:
            return (amplitude,)
        return (
            amplitude,
            ResolvedPulseNoise(
                channel_type=PhaseDamping,
                target_indices=target_indices,
                rate=channel.pure_dephasing_rate,
                condition=condition,
            ),
        )

    if duration is None:
        rate = getattr(channel, "rate", None)
        if rate is None:
            raise BackendValidationError(
                f"always-on {channel_type.__name__} requires rate mode"
            )
    else:
        try:
            rate = channel.as_rate(duration)
        except ValueError as exc:
            raise BackendValidationError(
                f"{channel_type.__name__} cannot be lowered for this pulse "
                f"occurrence: {exc}"
            ) from exc
    return (
        ResolvedPulseNoise(
            channel_type=channel_type,
            target_indices=target_indices,
            rate=rate,
            condition=condition,
        ),
    )
