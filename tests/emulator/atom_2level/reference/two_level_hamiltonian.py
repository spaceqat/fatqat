"""Dense two-level Hamiltonian oracle independent of production adapter code."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline
from scipy.linalg import expm

SIGMA_PLUS = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
NUMBER = np.diag([0.0, 1.0]).astype(complex)
IDENTITY = np.eye(2, dtype=complex)


def expand_local(operator, ordinal, site_count):
    factors = [IDENTITY] * site_count
    factors[ordinal] = operator
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def dense_hamiltonian(
    site_count,
    *,
    amplitude,
    detuning,
    phase,
    interactions=(),
):
    raising = sum(
        (expand_local(SIGMA_PLUS, site, site_count) for site in range(site_count)),
        np.zeros((2**site_count, 2**site_count), dtype=complex),
    )
    number = sum(
        (expand_local(NUMBER, site, site_count) for site in range(site_count)),
        np.zeros((2**site_count, 2**site_count), dtype=complex),
    )
    drive = amplitude * np.exp(1j * phase)
    hamiltonian = (drive * raising + np.conjugate(drive) * raising.conj().T) / 2
    hamiltonian -= detuning * number
    for first, second, strength in interactions:
        hamiltonian += strength * (
            expand_local(NUMBER, first, site_count)
            @ expand_local(NUMBER, second, site_count)
        )
    return hamiltonian


def solve_constant(site_count, duration, **hamiltonian_values):
    initial = np.zeros(2**site_count, dtype=complex)
    initial[0] = 1.0
    hamiltonian = dense_hamiltonian(site_count, **hamiltonian_values)
    return expm(-1j * hamiltonian * duration) @ initial


def solve_sampled(
    site_count,
    duration,
    *,
    amplitude_times,
    amplitude_values,
    detuning_times,
    detuning_values,
    phase,
    interactions=(),
):
    amplitude = make_interp_spline(
        amplitude_times,
        amplitude_values,
        k=min(3, len(amplitude_times) - 1),
        bc_type=None,
    )
    detuning = make_interp_spline(
        detuning_times,
        detuning_values,
        k=min(3, len(detuning_times) - 1),
        bc_type=None,
    )
    initial = np.zeros(2**site_count, dtype=complex)
    initial[0] = 1.0

    def derivative(time, state):
        hamiltonian = dense_hamiltonian(
            site_count,
            amplitude=complex(amplitude(time)),
            detuning=float(detuning(time)),
            phase=phase,
            interactions=interactions,
        )
        return -1j * hamiltonian @ state

    result = solve_ivp(
        derivative,
        (0.0, duration),
        initial,
        rtol=2e-11,
        atol=2e-12,
        method="DOP853",
    )
    assert result.success
    return result.y[:, -1]
