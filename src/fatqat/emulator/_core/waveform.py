"""Private SciPy spline helpers shared by pulse emulators.

QuTiP's sampled-array coefficients use ``scipy.interpolate.make_interp_spline``
with the same degree reduction and boundary condition implemented here.  The
helpers that inspect extrema are needed for model limit validation; QuTiP does
not expose those extrema.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.interpolate import BSpline, PPoly, make_interp_spline

from ...errors import BackendValidationError

_REQUESTED_SPLINE_DEGREE = 3


def _effective_spline_degree(sample_count: int) -> int:
    if type(sample_count) is not int or sample_count < 2:
        raise BackendValidationError("sampled waveforms require at least two samples")
    return min(_REQUESTED_SPLINE_DEGREE, sample_count - 1)


def _build_spline(times: Sequence[float], values: Sequence[float]) -> BSpline:
    """Build the SciPy spline used by QuTiP array coefficients."""
    time_array = np.asarray(times, dtype=float)
    value_array = np.asarray(values)
    if (
        time_array.ndim != 1
        or value_array.ndim != 1
        or len(time_array) != len(value_array)
    ):
        raise BackendValidationError(
            "waveform times and values must be matching one-dimensional arrays"
        )
    degree = _effective_spline_degree(len(time_array))
    return make_interp_spline(time_array, value_array, k=degree, bc_type=None)


def _real_spline_minimum_and_maximum(
    times: Sequence[float], values: Sequence[float]
) -> tuple[float, float]:
    """Return exact candidate extrema over the spline's closed sample domain."""
    spline = _build_spline(times, values)
    polynomial = PPoly.from_spline(spline)
    domain_start = float(times[0])
    domain_end = float(times[-1])
    candidates = {domain_start, domain_end}
    derivative = polynomial.derivative()

    for index, (raw_left, raw_right) in enumerate(zip(polynomial.x, polynomial.x[1:])):
        left = max(domain_start, float(raw_left))
        right = min(domain_end, float(raw_right))
        if right <= left:
            continue
        candidates.add(left)
        candidates.add(right)
        coefficients = np.trim_zeros(derivative.c[:, index], trim="f")
        if len(coefficients) <= 1:
            continue
        # PPoly coefficients use powers of (x - raw_left), in descending order.
        for root in np.roots(coefficients):
            tolerance = 1e-12 * max(1.0, abs(float(root.real)))
            if abs(float(root.imag)) > tolerance:
                continue
            point = float(raw_left) + float(root.real)
            if left < point < right:
                candidates.add(point)

    evaluated = np.asarray(spline(sorted(candidates)), dtype=float)
    return float(np.min(evaluated)), float(np.max(evaluated))


def _complex_spline_magnitude_maximum(
    times: Sequence[float], values: Sequence[complex]
) -> float:
    """Return the exact maximum magnitude of a possibly complex spline."""
    spline = _build_spline(times, values)
    value_array = np.asarray(values, dtype=complex)
    real_polynomial = PPoly.from_spline(_build_spline(times, np.real(value_array)))
    imaginary_polynomial = PPoly.from_spline(_build_spline(times, np.imag(value_array)))
    polynomial = PPoly(
        real_polynomial.c + 1j * imaginary_polynomial.c,
        real_polynomial.x,
    )
    domain_start = float(times[0])
    domain_end = float(times[-1])
    candidates = {domain_start, domain_end}

    for index, (raw_left, raw_right) in enumerate(zip(polynomial.x, polynomial.x[1:])):
        left = max(domain_start, float(raw_left))
        right = min(domain_end, float(raw_right))
        if right <= left:
            continue
        candidates.add(left)
        candidates.add(right)
        coefficients = np.trim_zeros(polynomial.c[:, index], trim="f")
        if len(coefficients) <= 1:
            continue
        magnitude_squared = np.polymul(coefficients, np.conjugate(coefficients))
        derivative = np.polyder(magnitude_squared.real)
        for root in np.roots(derivative):
            tolerance = 1e-12 * max(1.0, abs(float(root.real)))
            if abs(float(root.imag)) > tolerance:
                continue
            point = float(raw_left) + float(root.real)
            if left < point < right:
                candidates.add(point)

    evaluated = np.abs(spline(sorted(candidates)))
    return float(np.max(evaluated))


__all__: list[str] = []
