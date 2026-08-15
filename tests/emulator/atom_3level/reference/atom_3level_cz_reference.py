"""Independent numerical reference calculations for the three-level atom CZ.

This module intentionally imports neither the production atom realization,
adapter, nor backend. It reconstructs the two-qutrit Hamiltonian directly in
the dimensionless convention used by the committed numerical table.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
from scipy.integrate import ode, solve_ivp
from scipy.interpolate import CubicSpline

ANALYTIC_SOLVER_OPTIONS = {"method": "DOP853", "atol": 1e-11, "rtol": 1e-11}
PRODUCTION_SOLVER_OPTIONS = {
    "method": "adams",
    "atol": 1e-11,
    "rtol": 1e-9,
    "nsteps": 100000,
}
COMPUTATIONAL_INDICES = np.array((0, 1, 3, 4), dtype=int)


@dataclass(frozen=True)
class CZMetrics:
    """Pure computational-subspace metrics for one physical propagator."""

    process_fidelity: float
    average_fidelity: float
    survival: float
    leakage: float
    entangling_phase: float
    local_phases: tuple[float, float]


def principal(angle: float) -> float:
    """Return an angle in the specification interval ``(-pi, pi]``."""
    value = float(np.angle(np.exp(1j * angle)))
    return pi if value <= -pi + 1e-15 else value


def correction_unitary(local_z_correction_rad: float) -> np.ndarray:
    """Return the physical qutrit local-frame correction applied once."""
    local = np.diag((1.0, np.exp(1j * local_z_correction_rad), 1.0))
    return np.kron(local, local)


def analyze_cz_propagator(
    propagator: np.ndarray,
    calibration,
    *,
    apply_correction: bool,
) -> CZMetrics:
    """Analyze a raw or already-final physical two-qutrit propagator.

    ``apply_correction`` must be true only for the raw propagator.  The pure
    routine does no evolution and owns no production implementation object.
    """
    unitary = np.asarray(propagator, dtype=complex)
    if unitary.shape != (9, 9):
        raise ValueError("CZ analysis requires a 9 x 9 physical propagator")
    corrected = (
        correction_unitary(calibration.local_z_correction_rad) @ unitary
        if apply_correction
        else unitary
    )
    block = corrected[np.ix_(COMPUTATIONAL_INDICES, COMPUTATIONAL_INDICES)]
    cz = np.diag((1.0, 1.0, 1.0, -1.0))
    process = float(abs(np.trace(cz.conj().T @ block)) ** 2 / 16.0)
    survival = float(np.trace(block.conj().T @ block).real / 4.0)
    average = float((4.0 * process + survival) / 5.0)
    phases = (principal(np.angle(block[1, 1])), principal(np.angle(block[2, 2])))
    entangling = principal(
        np.angle(block[3, 3])
        - np.angle(block[2, 2])
        - np.angle(block[1, 1])
        + np.angle(block[0, 0])
    )
    return CZMetrics(
        process_fidelity=process,
        average_fidelity=average,
        survival=survival,
        leakage=1.0 - survival,
        entangling_phase=entangling,
        local_phases=phases,
    )


def _operators() -> tuple[np.ndarray, np.ndarray]:
    transition = np.array(((0, 0, 0), (0, 0, 0), (0, 1, 0)), dtype=complex)
    number = np.diag((0.0, 0.0, 1.0))
    identity = np.eye(3, dtype=complex)
    first = np.kron(transition, identity)
    second = np.kron(identity, transition)
    drive_x = (first.conj().T + first) + (second.conj().T + second)
    drive_y = -1j * ((first.conj().T - first) + (second.conj().T - second))
    drift = np.kron(number, number)
    return drive_x, drive_y, drift


def _dimensionless_duration(calibration) -> float:
    physical_duration = calibration.cz_duration_us
    converted = calibration.omega_1r_angular_per_us * physical_duration
    expected = 2.0 * pi * calibration.duration_area_cycles
    if not np.isclose(converted, expected, rtol=0.0, atol=1e-13):
        raise AssertionError(
            "physical-to-dimensionless CZ pulse-area conversion drifted"
        )
    return expected


def _solve_dimensionless(
    calibration,
    ratio: float,
    envelope,
    *,
    options: dict[str, float | str],
) -> np.ndarray:
    if ratio <= 0:
        raise ValueError("V/Omega must be positive")
    duration = _dimensionless_duration(calibration)
    drive_x, drive_y, drift = _operators()
    identity = np.eye(9, dtype=complex)

    def derivative(time: float, flattened: np.ndarray) -> np.ndarray:
        coefficient = envelope(time)
        hamiltonian = ratio * drift + 0.5 * (
            coefficient.real * drive_x + coefficient.imag * drive_y
        )
        return (-1j * hamiltonian @ flattened.reshape(9, 9)).ravel()

    solution = solve_ivp(
        derivative,
        (0.0, duration),
        identity.ravel(),
        method=str(options["method"]),
        atol=float(options["atol"]),
        rtol=float(options["rtol"]),
    )
    if not solution.success:
        raise RuntimeError(f"reference CZ integration failed: {solution.message}")
    return solution.y[:, -1].reshape(9, 9)


def _solve_physical(calibration, ratio: float, envelope) -> np.ndarray:
    """Independently integrate a physical-time cubic-sampled Hamiltonian."""
    if ratio <= 0:
        raise ValueError("V/Omega must be positive")
    drive_x, drive_y, drift = _operators()
    omega = calibration.omega_1r_angular_per_us
    identity = np.eye(9, dtype=complex)

    def derivative(time: float, flattened: np.ndarray) -> np.ndarray:
        coefficient = envelope(time)
        hamiltonian = ratio * omega * drift + 0.5 * (
            coefficient.real * drive_x + coefficient.imag * drive_y
        )
        return (-1j * hamiltonian @ flattened.reshape(9, 9)).ravel()

    integrator = ode(derivative).set_integrator(
        "zvode",
        method="adams",
        atol=PRODUCTION_SOLVER_OPTIONS["atol"],
        rtol=PRODUCTION_SOLVER_OPTIONS["rtol"],
        nsteps=PRODUCTION_SOLVER_OPTIONS["nsteps"],
    )
    integrator.set_initial_value(identity.ravel(), 0.0)
    result = integrator.integrate(calibration.cz_duration_us)
    if not integrator.successful():
        raise RuntimeError("sampled physical-time CZ integration failed")
    return result.reshape(9, 9)


def analytic_cz_oracle(
    calibration,
    ratio: float,
    *,
    physical_interaction_angular_per_us: float | None = None,
) -> tuple[np.ndarray, CZMetrics]:
    """Solve the analytic dimensionless two-atom Hamiltonian at ``V/Omega``.

    Production comparisons pass their physical interaction independently.  This
    keeps the reference boundary explicit: the physical angular interaction is
    divided by the physical one-radian Rabi frequency before integration.
    """
    omega = calibration.omega_1r_angular_per_us
    physical_interaction = (
        ratio * omega
        if physical_interaction_angular_per_us is None
        else physical_interaction_angular_per_us
    )
    converted_ratio = physical_interaction / omega
    if not np.isclose(converted_ratio, ratio, rtol=1e-13, atol=1e-13):
        raise AssertionError(
            "physical interaction does not match the requested V/Omega ratio"
        )

    def envelope(time: float) -> complex:
        phase = (
            calibration.phase_amplitude_rad
            * np.cos(calibration.phase_rate_ratio * time - calibration.phase_offset_rad)
            + calibration.linear_phase_rate_ratio * time
        )
        return np.exp(-1j * phase)

    raw = _solve_dimensionless(
        calibration, converted_ratio, envelope, options=ANALYTIC_SOLVER_OPTIONS
    )
    return raw, analyze_cz_propagator(raw, calibration, apply_correction=True)


def sampled_cz_reference(
    calibration, ratio: float, grid_points: int
) -> tuple[np.ndarray, CZMetrics]:
    """Independently solve cubic samples in physical time and physical units."""
    if type(grid_points) is not int or grid_points < 2:
        raise ValueError("grid_points must be an integer >= 2")
    duration = calibration.cz_duration_us
    physical_grid = np.linspace(0.0, duration, grid_points)
    physical_phase = (
        calibration.phase_amplitude_rad
        * np.cos(
            calibration.cz_phase_rate_angular_per_us * physical_grid
            - calibration.phase_offset_rad
        )
        + calibration.cz_linear_phase_rate_angular_per_us * physical_grid
    )
    physical_coefficients = calibration.omega_1r_angular_per_us * np.exp(
        -1j * physical_phase
    )
    real = CubicSpline(physical_grid, physical_coefficients.real)
    imag = CubicSpline(physical_grid, physical_coefficients.imag)
    raw = _solve_physical(
        calibration, ratio, lambda time: complex(real(time), imag(time))
    )
    return raw, analyze_cz_propagator(raw, calibration, apply_correction=True)


def aligned_frobenius_error(full: np.ndarray, independent: np.ndarray) -> float:
    """Phase-align two propagators then return normalized Frobenius distance."""
    full = np.asarray(full, dtype=complex)
    independent = np.asarray(independent, dtype=complex)
    if full.shape != independent.shape:
        raise ValueError("parallel propagators must share a shape")
    alpha = np.angle(np.trace(independent.conj().T @ full))
    return float(
        np.linalg.norm(full - np.exp(1j * alpha) * independent, ord="fro")
        / np.linalg.norm(independent, ord="fro")
    )
