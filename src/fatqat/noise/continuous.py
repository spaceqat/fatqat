"""Validated continuously active noise descriptors for pulse backends."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


class ContinuousNoise:
    """Marker base for typed continuously active noise descriptors."""


@dataclass(frozen=True)
class ThermalRelaxation(ContinuousNoise):
    """Per-subsystem qutrit T1/T2 parameters in nanoseconds."""

    # Public spelling follows the established T1/T2 physics vocabulary.
    # pylint: disable=invalid-name
    T1_ns: float
    T2_ns: float
    # pylint: enable=invalid-name

    def __post_init__(self) -> None:
        for name, value in (("T1_ns", self.T1_ns), ("T2_ns", self.T2_ns)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")
            object.__setattr__(self, name, float(value))
        if self.T2_ns > 2 * self.T1_ns:
            raise ValueError("T2_ns must be less than or equal to 2*T1_ns")
