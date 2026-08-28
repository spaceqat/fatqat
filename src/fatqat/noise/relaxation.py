"""Validated T1/T2 relaxation for emulator noise and simulator conversion."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .base import Channel


@dataclass(frozen=True, kw_only=True)
class ThermalRelaxation(Channel):
    """Zero-temperature relaxation specified by T1 and T2.

    T1 is the population-relaxation time and T2 is the total
    transverse-coherence time. They have no intrinsic unit, but both must use
    the same one. A duration passed to ``as_channels()`` uses that unit too;
    when registering this noise with an emulator, use the emulator model's
    time unit. Both values must be finite positive ``int`` or ``float`` values
    other than ``bool``, and physical consistency requires
    ``t2 <= 2 * t1``.

    The derived amplitude rate is ``1 / t1``. The residual pure-dephasing rate
    is ``1 / t2 - 1 / (2 * t1)``, which avoids counting the coherence loss from
    population relaxation twice. A time-aware emulator may resolve those rates
    to local Lindblad operators.

    The as_channels method returns a simulator's qubit amplitude- and
    phase-damping pair for an explicit duration. That helper is qubit-specific;
    multilevel emulators use their own level-dependent Lindblad operators.
    This noise acts on one subsystem.

    Args:
        t1: Finite positive population-relaxation time.
        t2: Finite positive total transverse-coherence time, no greater than
            ``2 * t1``.

    Raises:
        ValueError: If either value is not a finite positive real number or if
            t2 is greater than ``2 * t1``.

    Attributes:
        t1: Population-relaxation time, normalized to float.
        t2: Total transverse-coherence time, normalized to float.

    Examples:
        >>> import fatqat as fq
        >>> relaxation = fq.noise.ThermalRelaxation(t1=60.0, t2=80.0)
        >>> relaxation.t1, relaxation.t2
        (60.0, 80.0)
    """

    num_subsystems = 1
    t1: float
    t2: float

    def __post_init__(self) -> None:
        for name, value in (("t1", self.t1), ("t2", self.t2)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")
            object.__setattr__(self, name, float(value))
        if self.t2 > 2 * self.t1:
            raise ValueError("t2 must be less than or equal to 2*t1")

    @property
    def amplitude_rate(self) -> float:
        """Return the base population-relaxation rate ``1 / t1``.

        Returns:
            A rate in the inverse of the time unit used by t1.
        """
        return 1.0 / self.t1

    @property
    def pure_dephasing_rate(self) -> float:
        """Return the residual dephasing rate after T1 decoherence.

        Returns:
            ``1 / t2 - 1 / (2 * t1)`` in the inverse of the configured time
            unit. The value is nonnegative because construction enforces
            ``t2 <= 2 * t1``.
        """
        return 1.0 / self.t2 - 1.0 / (2.0 * self.t1)

    def as_channels(self, duration: float):
        """Return the simulator's qubit damping pair for an explicit duration.

        The first result is AmplitudeDamping with probability
        ``1 - exp(-duration / t1)``. The second is PhaseDamping with
        probability ``1 - exp(-pure_dephasing_rate * duration)``. Applying
        them in that order gives qubit population decay set by T1 and total
        coherence decay set by T2.

        This helper is specifically a qubit conversion. Its scalar amplitude
        result does not supply the multiple transition values required by a
        higher-dimensional simulator channel.

        Args:
            duration: Finite nonnegative ``int`` or ``float`` other than
                ``bool``, in the same unit as t1 and t2.

        Returns:
            A tuple ``(amplitude_damping, phase_damping)`` of qubit simulator
            noise objects.

        Raises:
            ValueError: If duration is not a finite nonnegative real number.

        Examples:
            >>> import fatqat as fq
            >>> relaxation = fq.noise.ThermalRelaxation(t1=60.0, t2=80.0)
            >>> damping, dephasing = relaxation.as_channels(duration=2.0)
            >>> damping.p[0] > 0.0
            True
            >>> dephasing.p > 0.0
            True
        """
        # Import locally to avoid a catalog/relaxation import cycle.
        from .catalog import AmplitudeDamping, PhaseDamping

        amplitude = AmplitudeDamping(rate=self.amplitude_rate)
        dephasing = PhaseDamping(rate=self.pure_dephasing_rate)
        return (
            AmplitudeDamping(p=amplitude.as_probability(duration)),
            PhaseDamping(p=dephasing.as_probability(duration)),
        )
