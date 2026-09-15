"""GRAPE optimization of the global two-atom CZ pulse.

Reference: S. Jandura and G. Pupillo, "Time-Optimal Two- and Three-Qubit Gates
for Rydberg Atoms", Quantum 6, 712 (2022), doi:10.22331/q-2022-06-20-712,
Secs. 2.2.1, 3.1 and Fig. 1.

The optimizer implements the GRAPE ansatz of the paper.  The laser phase is
piecewise constant on ``N`` intervals of a fixed duration ``T``,

    u(t) = phi_j  for  t in [j dt, (j + 1) dt],   dt = T / N,

the amplitude stays at ``Omega_max``, and the objective is the averaged gate
error of paper Eq. (7).  The free single-qubit phase ``theta`` of the phase gate
is optimized together with the phases, as in the paper.  Gradients with respect
to all ``N + 1`` parameters are computed analytically with the forward/backward
scheme that gives GRAPE its name; the optimizer is a limited-memory
Broyden-Fletcher-Goldfarb-Shanno method, the algorithm used in the paper.

The time-optimal pulse duration ``T*`` is found the way the paper does it
(Fig. 1(c)): pulses are optimized on a descending grid of durations with warm
starts, and the resulting gate errors are fitted with

    1 - F = A (T* - T)**2   for  T < T*,   1 - F = 0   for  T >= T*.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.optimize import least_squares, minimize

from ....errors import BackendValidationError
from .problem import BLOCK_COUPLINGS, CZPulse

#: Gate error below which a GRAPE run is considered converged.  The paper
#: quotes the same order of magnitude as its convergence criterion (Fig. 1(c)).
CONVERGED_GATE_ERROR: Final[float] = 1e-10


@dataclass(frozen=True)
class TimeOptimalScan:
    """Result of a descending GRAPE sweep over pulse durations.

    Args:
        durations: Durations ``T * Omega_max`` in decreasing order.
        gate_errors: Best averaged gate error found at each duration.
        pulses: The optimized piecewise-constant pulse of each duration.
    """

    durations: tuple[float, ...]
    gate_errors: tuple[float, ...]
    pulses: tuple[CZPulse, ...]


@dataclass(frozen=True)
class TimeOptimalFit:
    """Least-squares fit of ``1 - F = A (T* - T)**2`` to a duration sweep.

    Args:
        optimal_duration: Fitted time-optimal duration ``T* Omega_max``.
        coefficient: Fitted curvature ``A``.
        residual: Root-mean-square residual of the fit.
        samples: The ``(T Omega_max, 1 - F)`` pairs that entered the fit.
    """

    optimal_duration: float
    coefficient: float
    residual: float
    samples: tuple[tuple[float, float], ...]


def optimize_cz_pulse(
    duration: float,
    *,
    pieces: int = 99,
    initial: CZPulse | None = None,
    seed: int | None = None,
    max_iterations: int = 2000,
    gradient_tolerance: float = 1e-12,
) -> CZPulse:
    """Optimize a piecewise-constant phase-modulated CZ pulse with GRAPE.

    Args:
        duration: Pulse duration ``T * Omega_max``; positive and finite.
        pieces: Number of piecewise-constant phase intervals ``N``.  The paper
            uses 99 pieces for the CZ gate and reports that doubling the number
            of pieces changes the gate error by at most ``3e-6``.
        initial: Optional warm start.  Its phases are reused when the number of
            pieces matches, which is how the paper follows the optimum across
            durations; otherwise a random start is used.
        seed: Seed of the random number generator used for the random start.
            Pass an integer to make a run reproducible.
        max_iterations: Iteration limit of the optimizer.
        gradient_tolerance: Gradient-norm stopping threshold.

    Returns:
        The optimized :class:``~.problem.CZPulse`` with
        ``interpolation="piecewise_constant"``, together with the optimized
        single-qubit phase ``theta``.

    Raises:
        BackendValidationError: If the duration, the number of pieces, the
            iteration limit, or the warm start is invalid.
    """
    _validate_setup(duration, pieces, max_iterations)
    phases, theta = _initial_parameters(duration, pieces, initial, seed)
    step = duration / pieces

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        return _objective_and_gradient(parameters[:-1], parameters[-1], step)

    result = minimize(
        objective,
        np.concatenate([phases, [theta]]),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": max_iterations,
            "gtol": gradient_tolerance,
            "ftol": 1e-16,
            "maxcor": 30,
        },
    )
    edges = np.linspace(0.0, duration, pieces + 1)
    # A constant phase shift is a symmetry of the gate (paper Sec. 2.1.3), so
    # the gauge is fixed the way the paper does it: Omega(0) real and positive,
    # that is phi(0) = 0, with every phase wrapped to (-pi, pi].
    phases = _wrap_phases(np.asarray(result.x[:-1]) - result.x[0])
    return CZPulse(
        duration=duration,
        times=tuple(float(value) for value in edges),
        phases=tuple(float(value) for value in phases),
        theta=float(np.mod(result.x[-1], 2.0 * math.pi)),
        interpolation="piecewise_constant",
        method="grape",
    )


def _wrap_phases(phases: np.ndarray) -> np.ndarray:
    """Return phases wrapped to ``(-pi, pi]``."""
    return np.mod(np.asarray(phases) + math.pi, 2.0 * math.pi) - math.pi


def scan_durations(
    *,
    longest: float,
    shortest: float,
    samples: int,
    pieces: int = 99,
    seed: int | None = None,
    max_iterations: int = 2000,
) -> TimeOptimalScan:
    """Sweep pulse durations downwards with warm-started GRAPE runs.

    This reproduces the procedure behind Fig. 1(c) of the paper: the longest
    pulse is optimized from a random start, and every shorter pulse is
    initialized with the optimized phases of the previous, slightly longer
    pulse.  Continuation makes the sweep follow the global optimum instead of
    a random local minimum.

    Args:
        longest: Largest duration ``T * Omega_max`` of the sweep.
        shortest: Smallest duration ``T * Omega_max`` of the sweep.
        samples: Number of durations; at least two.
        pieces: Number of piecewise-constant phases per pulse.
        seed: Seed for the random start of the longest pulse.
        max_iterations: Iteration limit of each GRAPE run.

    Returns:
        The durations in decreasing order, the best gate error at each
        duration, and the corresponding pulses.

    Raises:
        BackendValidationError: If the duration range or the sample count is
            invalid.
    """
    _validate_setup(longest, pieces, max_iterations)
    _validate_setup(shortest, pieces, max_iterations)
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 2:
        raise BackendValidationError("samples must be an integer >= 2")
    if shortest > longest:
        raise BackendValidationError("shortest must not exceed longest")
    durations = tuple(np.linspace(longest, shortest, samples))
    pulses = []
    errors = []
    warm = None
    for duration in durations:
        pulse = optimize_cz_pulse(
            float(duration),
            pieces=pieces,
            initial=warm,
            seed=seed,
            max_iterations=max_iterations,
        )
        warm = pulse
        pulses.append(pulse)
        errors.append(pulse.gate_error())
    return TimeOptimalScan(durations, tuple(errors), tuple(pulses))


def fit_time_optimal_duration(
    scan: TimeOptimalScan,
    *,
    error_floor: float = 3.0 * CONVERGED_GATE_ERROR,
    min_samples: int = 3,
) -> TimeOptimalFit:
    """Fit ``1 - F = A (T* - T)**2`` to the converged part of a sweep.

    Args:
        scan: Result of :func:``scan_durations``.
        error_floor: Gate errors below this value are treated as numerical
            convergence noise and excluded, because the fit function predicts
            zero error at and beyond ``T*``.
        min_samples: Minimum number of retained samples; at least two are
            mathematically required.

    Returns:
        The fitted time-optimal duration and curvature.

    Raises:
        BackendValidationError: If fewer than ``min_samples`` usable samples
            remain above the error floor.
    """
    if min_samples < 2:
        raise BackendValidationError("min_samples must be at least 2")
    usable = [
        (duration, error)
        for duration, error in zip(scan.durations, scan.gate_errors)
        if error > error_floor and error > 0.0
    ]
    if len(usable) < min_samples:
        raise BackendValidationError(
            "not enough gate errors above the error floor to fit the "
            "time-optimal duration"
        )
    durations = np.array([item[0] for item in usable])
    errors = np.array([item[1] for item in usable])

    def residual(parameters: np.ndarray) -> np.ndarray:
        optimal, coefficient = parameters
        return np.sqrt(np.maximum(coefficient, 0.0)) * (optimal - durations) - np.sqrt(
            errors
        )

    start = _linear_threshold_estimate(durations, errors)
    fit = least_squares(residual, start, method="lm")
    optimal_duration, coefficient = float(fit.x[0]), float(fit.x[1])
    rms = float(np.sqrt(np.mean(fit.fun**2)))
    return TimeOptimalFit(
        optimal_duration=optimal_duration,
        coefficient=coefficient,
        residual=rms,
        samples=tuple(usable),
    )


def time_optimal_cz_pulse(
    *,
    pieces: int = 99,
    seed: int | None = None,
    window: float = 0.1,
    step: float = 0.005,
    start: float = 7.7,
    max_iterations: int = 2000,
) -> tuple[CZPulse, TimeOptimalFit]:
    """Find the time-optimal CZ pulse with the paper's sweep-and-fit procedure.

    Args:
        pieces: Number of piecewise-constant phases per pulse.
        seed: Seed for the random start of the first sweep point.
        window: Half-width of the duration window around the expected optimum.
        step: Duration step of the descending sweep.
        start: Upper end of the sweep, ``T * Omega_max``.  The paper starts at
            ``7.7``, where the gate error is already close to zero.
        max_iterations: Iteration limit of each GRAPE run.

    Returns:
        A pair ``(pulse, fit)``: the pulse optimized at the fitted optimum
        (the paper's Fig. 1(d) pulse) and the threshold fit.
    """
    if not isinstance(pieces, int) or isinstance(pieces, bool) or pieces < 1:
        raise BackendValidationError("pieces must be a positive integer")
    if step <= 0.0 or window <= 0.0 or start <= window:
        raise BackendValidationError("invalid scan window or step")
    samples = int(round(2.0 * window / step)) + 1
    scan = scan_durations(
        longest=start,
        shortest=start - 2.0 * window,
        samples=samples,
        pieces=pieces,
        seed=seed,
        max_iterations=max_iterations,
    )
    fit = fit_time_optimal_duration(scan)
    pulse = optimize_cz_pulse(
        fit.optimal_duration,
        pieces=pieces,
        initial=scan.pulses[0],
        max_iterations=max_iterations,
    )
    return pulse, fit


def _initial_parameters(
    duration: float,
    pieces: int,
    initial: CZPulse | None,
    seed: int | None,
) -> tuple[np.ndarray, float]:
    if initial is not None and len(initial.phases) == pieces:
        return np.array(initial.phases, dtype=float), float(initial.theta)
    rng = np.random.default_rng(seed)
    return (
        rng.uniform(-math.pi, math.pi, size=pieces),
        float(rng.uniform(-math.pi, math.pi)),
    )


def _objective_and_gradient(
    phases: np.ndarray, theta: float, step: float
) -> tuple[float, np.ndarray]:
    """Return the averaged gate error and its gradient for one pulse.

    The gradient uses the standard GRAPE construction: a forward sweep gives
    the states before every piece, a backward sweep gives the costate rows of
    every piece, and each derivative is one contraction of the two.  The
    single-qubit phase ``theta`` is differentiated analytically as well.
    """
    amplitudes = []
    derivatives = []
    for coupling in BLOCK_COUPLINGS:
        cosine = math.cos(coupling * step / 2.0)
        sine = math.sin(coupling * step / 2.0)
        forward = np.empty((len(phases) + 1, 2), dtype=complex)
        forward[0] = (1.0, 0.0)
        for index, phase in enumerate(phases):
            first, second = forward[index]
            raising = -1j * sine * cmath.exp(1j * phase)
            lowering = -1j * sine * cmath.exp(-1j * phase)
            forward[index + 1] = (
                cosine * first + raising * second,
                lowering * first + cosine * second,
            )
        backward = np.empty((len(phases) + 1, 2), dtype=complex)
        backward[-1] = (1.0, 0.0)
        for index in range(len(phases) - 1, -1, -1):
            first, second = backward[index + 1]
            raising = -1j * sine * cmath.exp(1j * phases[index])
            lowering = -1j * sine * cmath.exp(-1j * phases[index])
            backward[index] = (
                first * cosine + second * lowering,
                first * raising + second * cosine,
            )
        exponents = np.exp(1j * np.asarray(phases, dtype=complex))
        derivatives.append(
            sine
            * (
                backward[1:, 0] * exponents * forward[:-1, 1]
                - backward[1:, 1] * np.conj(exponents) * forward[:-1, 0]
            )
        )
        amplitudes.append(complex(forward[-1, 0]))
    one, two = amplitudes
    a_one = cmath.exp(-1j * theta) * one
    a_two = -cmath.exp(-2j * theta) * two
    total = 1.0 + 2.0 * a_one + a_two
    fidelity = (abs(total) ** 2 + 1.0 + 2.0 * abs(a_one) ** 2 + abs(a_two) ** 2) / 20.0
    weight_one = (total + a_one) * cmath.exp(1j * theta) / 10.0
    weight_two = -(total + a_two) * cmath.exp(2j * theta) / 20.0
    # The objective is 1 - F, so every derivative of F carries a minus sign.
    # For a real F of a complex amplitude, dF = 2 Re[conj(dF/dA*) dA].
    gradient = np.empty(len(phases) + 1)
    gradient[:-1] = -2.0 * np.real(
        np.conj(weight_one) * derivatives[0] + np.conj(weight_two) * derivatives[1]
    )
    gradient[-1] = -(1.0 / 5.0) * float(np.imag(np.conj(total) * (a_one + a_two)))
    return 1.0 - fidelity, gradient


def _linear_threshold_estimate(
    durations: np.ndarray, errors: np.ndarray
) -> tuple[float, float]:
    """Return a starting guess for ``(T*, A)`` from a linear fit of sqrt(1-F)."""
    slope, intercept = np.polyfit(durations, np.sqrt(errors), 1)
    if slope >= 0.0 or intercept <= 0.0:
        return float(np.max(durations) + 0.1), 1.0
    return float(-intercept / slope), float(slope**2)


def _validate_setup(duration: float, pieces: int, max_iterations: int) -> None:
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0.0
    ):
        raise BackendValidationError("duration must be finite and positive")
    if not isinstance(pieces, int) or isinstance(pieces, bool) or pieces < 1:
        raise BackendValidationError("pieces must be a positive integer")
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations < 1
    ):
        raise BackendValidationError("max_iterations must be a positive integer")
