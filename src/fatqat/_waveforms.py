"""Backend-independent waveform authoring values."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Complex, Real


class Waveform:
    """Base class for waveforms accepted by pulse controls."""

    @property
    def duration(self) -> float:
        """Duration of the waveform in its enclosing model's time unit."""
        raise NotImplementedError


def _finite_tuple(values: Iterable[Real], *, field: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{field} must be a one-dimensional real sequence")
    converted: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{field} must contain only finite real values")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{field} must contain only finite real values")
        converted.append(number)
    return tuple(converted)


def _finite_values(
    values: Iterable[Real | complex], *, field: str
) -> tuple[float | complex, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{field} must be a one-dimensional numeric sequence")
    converted: list[float | complex] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Complex):
            raise TypeError(f"{field} must contain only finite real or complex values")
        if isinstance(value, Real):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(
                    f"{field} must contain only finite real or complex values"
                )
            converted.append(number)
            continue
        number = complex(value)
        if not math.isfinite(number.real) or not math.isfinite(number.imag):
            raise ValueError(f"{field} must contain only finite real or complex values")
        converted.append(number)
    return tuple(converted)


@dataclass(frozen=True)
class SampledWaveform(Waveform):
    """Define a waveform on a strictly increasing local time grid.

    The grid may be nonuniform. It starts at ``0.0``, and its final time is
    the waveform's ``duration``. Built-in pulse emulators use not-a-knot
    spline interpolation.

    Samples may be real or complex. The selected channel determines which form
    is allowed. A pulse emulator also checks interpolation and any amplitude
    or duration limits when the waveform is used.

    Args:
        times: At least two finite real coordinates. The first must be exactly
            ``0.0`` and each later value must be strictly greater than the
            preceding value.
        values: Finite real or complex samples with the same length as
            ``times``. Booleans are rejected.

    Raises:
        TypeError: If ``times`` is not a real one-dimensional iterable, if
            ``values`` is not a numeric one-dimensional iterable, or if
            either contains booleans or values of the wrong numeric kind.
        ValueError: If lengths differ, fewer than two samples are supplied, a
            value is non-finite, the grid does not begin at zero, or
            times are not strictly increasing.

    Examples:
        >>> import fatqat as fq
        >>> waveform = fq.emulator.SampledWaveform(
        ...     (0.0, 0.2, 0.7, 1.0),
        ...     (0.0, 0.8, 0.4, 0.0),
        ... )
        >>> waveform.duration
        1.0
    """

    times: tuple[float, ...]
    values: tuple[float | complex, ...]

    def __init__(self, times: Iterable[Real], values: Iterable[Real | complex]) -> None:
        copied_times = _finite_tuple(times, field="times")
        copied_values = _finite_values(values, field="values")
        if len(copied_times) != len(copied_values):
            raise ValueError("times and values must have equal lengths")
        if len(copied_times) < 2:
            raise ValueError("sampled waveforms require at least two samples")
        if copied_times[0] != 0.0:
            raise ValueError("times must start at exactly 0.0")
        if any(right <= left for left, right in zip(copied_times, copied_times[1:])):
            raise ValueError("times must be strictly increasing")
        object.__setattr__(self, "times", copied_times)
        object.__setattr__(self, "values", copied_values)

    @property
    def duration(self) -> float:
        """Return the final local sample time."""
        return self.times[-1]


__all__ = ["Waveform", "SampledWaveform"]
