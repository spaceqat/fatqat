"""Relaxation converter: T1/T2 + duration into catalog channel descriptors."""

from math import exp

import numpy as np
import pytest

import fatqat as fq
from fatqat.noise import relaxation_channels
from fatqat.noise.catalog import amplitude_damping_rule, phase_damping_rule


def _apply(kraus_ops, rho):
    return sum(k @ rho @ k.conj().T for k in kraus_ops)


def test_population_decay_matches_t1():
    t1, duration = 60e-6, 40e-9
    damping, _ = relaxation_channels(t1, 0.8 * t1, duration)

    assert np.isclose(damping.gammas[0], 1 - exp(-duration / t1))


def test_composed_coherence_decay_matches_t2():
    t1, t2, duration = 60e-6, 90e-6, 5e-6
    damping, dephasing = relaxation_channels(t1, t2, duration)
    targets = (fq.QuantumRegister(1)[0],)
    plus = np.full((2, 2), 0.5, dtype=complex)
    rho = _apply(amplitude_damping_rule(damping, targets=targets), plus)
    rho = _apply(phase_damping_rule(dephasing, targets=targets), rho)

    # The defining property: off-diagonal coherence decays by exp(-t/T2)
    # after both channels, however the T1 part splits off.
    assert np.isclose(rho[0, 1], 0.5 * exp(-duration / t2))


def test_zero_duration_yields_identity_channels():
    damping, dephasing = relaxation_channels(50e-6, 70e-6, 0.0)

    assert damping.gammas == (0.0,)
    assert dephasing.p == 0.0


def test_t2_at_physical_bound_is_pure_t1():
    t1, duration = 50e-6, 1e-6
    _, dephasing = relaxation_channels(t1, 2 * t1, duration)

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
def test_invalid_calibration_values_raise(t1, t2, duration):
    with pytest.raises(ValueError):
        relaxation_channels(t1, t2, duration)
