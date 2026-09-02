"""ThermalRelaxation T1/T2 validation and derived rates."""

from math import exp, sqrt

import pytest

from fatqat.noise import AmplitudeDamping, PhaseDamping, ThermalRelaxation


def test_thermal_relaxation_is_keyword_only_and_exposes_shared_rates():
    with pytest.raises(TypeError):
        # pylint: disable-next=missing-kwoa
        ThermalRelaxation(60e-6, 80e-6)  # noqa: pyright-ignore

    source = ThermalRelaxation(t1=60e-6, t2=80e-6)
    assert source.amplitude_rate == pytest.approx(1 / source.t1)
    assert source.pure_dephasing_rate == pytest.approx(
        1 / source.t2 - 1 / (2 * source.t1)
    )


def test_explicit_damping_probabilities_reproduce_t2_coherence_decay():
    t1, t2, duration = 60e-6, 90e-6, 5e-6
    relaxation = ThermalRelaxation(t1=t1, t2=t2)
    amplitude_p = AmplitudeDamping(rate=relaxation.amplitude_rate).as_probability(
        duration
    )
    phase_p = PhaseDamping(rate=relaxation.pure_dephasing_rate).as_probability(duration)

    coherence_factor = sqrt(1 - amplitude_p) * (1 - phase_p)
    assert coherence_factor == pytest.approx(exp(-duration / t2))


def test_t2_at_physical_bound_is_pure_t1():
    t1 = 50e-6
    relaxation = ThermalRelaxation(t1=t1, t2=2 * t1)

    assert relaxation.pure_dephasing_rate == 0.0


@pytest.mark.parametrize(
    "t1, t2",
    [
        (0.0, 1e-6),  # t1 must be positive
        (-1e-6, 1e-6),
        (50e-6, 0.0),  # t2 must be positive
        (50e-6, 101e-6),  # t2 > 2*t1 is unphysical
    ],
)
def test_invalid_relaxation_values_raise(t1, t2):
    with pytest.raises(ValueError):
        ThermalRelaxation(t1=t1, t2=t2)
