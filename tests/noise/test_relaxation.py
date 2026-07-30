"""ThermalRelaxation: T1/T2 + duration into catalog channels."""

from math import exp

import numpy as np
import pytest

import fatqat as fq
from fatqat.noise import ThermalRelaxation
from fatqat.noise.catalog import amplitude_damping_rule, phase_damping_rule


def _apply(kraus_ops, rho):
    return sum(k @ rho @ k.conj().T for k in kraus_ops)


def test_population_decay_matches_t1():
    t1, duration = 60e-6, 40e-9
    damping, _ = ThermalRelaxation(t1=t1, t2=0.8 * t1).as_channels(duration)

    assert np.isclose(damping.p[0], 1 - exp(-duration / t1))


def test_thermal_relaxation_is_keyword_only_and_exposes_shared_rates():
    with pytest.raises(TypeError):
        ThermalRelaxation(60e-6, 80e-6)  # noqa: pyright-ignore

    source = ThermalRelaxation(t1=60e-6, t2=80e-6)
    assert source.amplitude_rate == pytest.approx(1 / source.t1)
    assert source.pure_dephasing_rate == pytest.approx(
        1 / source.t2 - 1 / (2 * source.t1)
    )


def test_composed_coherence_decay_matches_t2():
    t1, t2, duration = 60e-6, 90e-6, 5e-6
    damping, dephasing = ThermalRelaxation(t1=t1, t2=t2).as_channels(duration)
    targets = (fq.QuantumRegister(1)[0],)
    plus = np.full((2, 2), 0.5, dtype=complex)
    rho = _apply(amplitude_damping_rule(damping, targets=targets), plus)
    rho = _apply(phase_damping_rule(dephasing, targets=targets), rho)

    # The defining property: off-diagonal coherence decays by exp(-t/T2)
    # after both channels, however the T1 part splits off.
    assert np.isclose(rho[0, 1], 0.5 * exp(-duration / t2))


def test_zero_duration_yields_identity_channels():
    damping, dephasing = ThermalRelaxation(t1=50e-6, t2=70e-6).as_channels(0.0)

    assert damping.p == (0.0,)
    assert dephasing.p == 0.0


def test_t2_at_physical_bound_is_pure_t1():
    t1, duration = 50e-6, 1e-6
    _, dephasing = ThermalRelaxation(t1=t1, t2=2 * t1).as_channels(duration)

    assert dephasing.p == 0.0  # all decoherence already carried by damping


@pytest.mark.parametrize(
    "t1, t2, duration",
    [
        (0.0, 1e-6, 1e-9),  # t1 must be positive
        (-1e-6, 1e-6, 1e-9),
        (50e-6, 0.0, 1e-9),  # t2 must be positive
        (50e-6, 101e-6, 1e-9),  # t2 > 2*t1 is unphysical
        (50e-6, 40e-6, -1e-9),  # negative duration
    ],
)
def test_invalid_relaxation_values_raise(t1, t2, duration):
    with pytest.raises(ValueError):
        ThermalRelaxation(t1=t1, t2=t2).as_channels(duration)
