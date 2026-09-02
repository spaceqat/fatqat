"""Validated T1/T2 relaxation for emulator noise."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .base import Channel


@dataclass(frozen=True, kw_only=True)
class ThermalRelaxation(Channel):
    """Zero-temperature qubit relaxation specified by T1 and T2.

    T1 is the population-relaxation time and T2 is the total
    transverse-coherence time. They have no intrinsic unit, but both must use
    the same one. When this noise is registered with an emulator, that unit
    must match the emulator model's time unit. Both values must be finite
    positive ``int`` or ``float`` values other than ``bool``, and physical
    consistency requires ``t2 <= 2 * t1``.

    The derived amplitude rate is ``1 / t1``. The residual pure-dephasing rate
    is ``1 / t2 - 1 / (2 * t1)``, which avoids counting the coherence loss from
    population relaxation twice. The built-in realization is qubit-only; use
    :class:`~fatqat.noise.TransitionRelaxation` to author explicit
    multilevel jumps.

    This is a continuous-time descriptor for emulators. Matrix simulators use
    finite probability-form
    :class:`~fatqat.noise.AmplitudeDamping` and
    :class:`~fatqat.noise.PhaseDamping` declarations instead. This noise acts
    on one subsystem.

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
