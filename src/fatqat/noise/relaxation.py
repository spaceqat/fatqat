"""Validated T1/T2 relaxation descriptor shared by simulator backends."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .base import Channel


@dataclass(frozen=True, kw_only=True)
class ThermalRelaxation(Channel):
    """T1/T2 relaxation parameters and their finite-channel conversion.

    The values use the inverse of the owning backend/model's declared time
    unit; this descriptor itself does not impose a unit. It may be registered
    as background noise for a time-aware backend, while
    :meth:`as_channels` gives the equivalent finite channel pair over one
    explicit duration.
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
        """Return the base ladder-relaxation rate ``1 / t1``."""
        return 1.0 / self.t1

    @property
    def pure_dephasing_rate(self) -> float:
        """Return the residual dephasing rate after T1 decoherence."""
        return 1.0 / self.t2 - 1.0 / (2.0 * self.t1)

    def as_channels(self, duration: float):
        """Return finite qubit damping channels over ``duration``.

        The returned amplitude- and phase-damping descriptors compose to the
        T1/T2 relaxation channel for one qubit.  ``duration`` must use the
        same time unit as ``t1`` and ``t2``.
        """
        # Import locally to avoid a catalog/relaxation import cycle.
        from .catalog import AmplitudeDamping, PhaseDamping

        amplitude = AmplitudeDamping(rate=self.amplitude_rate)
        dephasing = PhaseDamping(rate=self.pure_dephasing_rate)
        return (
            AmplitudeDamping(p=amplitude.as_probability(duration)),
            PhaseDamping(p=dephasing.as_probability(duration)),
        )
