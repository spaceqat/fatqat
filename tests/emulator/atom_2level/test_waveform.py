"""Shared sampled-spline construction and exact-extrema contracts."""

import numpy as np
import pytest
import qutip

from fatqat.emulator._core.waveform import (
    _build_spline,
    _complex_spline_magnitude_maximum,
    _effective_spline_degree,
    _real_spline_minimum_and_maximum,
)


@pytest.mark.parametrize("sample_count", [2, 3, 4, 7])
@pytest.mark.parametrize("complex_values", [False, True])
def test_spline_effective_degree_and_qutip_value_equivalence(
    sample_count, complex_values
):
    rng = np.random.default_rng(20260807 + sample_count)
    intervals = rng.uniform(0.2, 0.8, size=sample_count - 1)
    times = np.concatenate(([0.0], np.cumsum(intervals)))
    values = rng.normal(size=sample_count)
    if complex_values:
        values = values + 1j * rng.normal(size=sample_count)
    spline = _build_spline(times, values)
    coefficient = qutip.coefficient(values, tlist=times, order=3)
    probes = np.concatenate((times, rng.uniform(times[0], times[-1], size=25)))

    assert _effective_spline_degree(sample_count) == min(3, sample_count - 1)
    assert spline.k == min(3, sample_count - 1)
    assert np.asarray([coefficient(point) for point in probes]) == pytest.approx(
        spline(probes), abs=1e-11
    )


def test_real_spline_extrema_include_internal_overshoot_not_only_knots():
    times = (0.0, 1.0, 2.0, 3.0)
    values = (0.0, 1.0, 1.0, 0.0)

    minimum, maximum = _real_spline_minimum_and_maximum(times, values)

    assert minimum == pytest.approx(0.0)
    assert maximum == pytest.approx(1.125)
    assert maximum > max(values)


def test_real_spline_extrema_work_on_nonuniform_multi_segment_grid():
    times = (0.0, 0.2, 1.1, 1.7, 3.0)
    values = (-0.5, 0.7, -0.2, 0.4, 0.1)
    spline = _build_spline(times, values)
    minimum, maximum = _real_spline_minimum_and_maximum(times, values)
    dense = spline(np.linspace(times[0], times[-1], 20001))

    assert minimum <= float(np.min(dense)) + 1e-8
    assert maximum >= float(np.max(dense)) - 1e-8


def test_complex_spline_magnitude_includes_interior_stationary_points():
    times = (0.0, 1.0, 2.0, 3.0)
    values = (0.0j, 1.0j, 1.0j, 0.0j)
    spline = _build_spline(times, values)

    maximum = _complex_spline_magnitude_maximum(times, values)
    dense = np.abs(spline(np.linspace(times[0], times[-1], 20001)))

    assert maximum == pytest.approx(1.125)
    assert maximum >= float(np.max(dense)) - 1e-8
    assert maximum > max(abs(value) for value in values)


def test_complex_spline_magnitude_tracks_rotating_nonuniform_spline():
    times = (0.0, 0.3, 1.1, 2.0, 3.0)
    values = (0.2, 0.7j, -0.4 + 0.2j, 0.5 - 0.1j, -0.2j)
    spline = _build_spline(times, values)

    maximum = _complex_spline_magnitude_maximum(times, values)
    dense = np.abs(spline(np.linspace(times[0], times[-1], 50001)))

    assert maximum >= float(np.max(dense)) - 1e-8
    assert maximum <= float(np.max(dense)) + 1e-6
