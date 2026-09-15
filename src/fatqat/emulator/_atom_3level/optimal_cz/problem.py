"""Reduced two-atom control problem for a time-optimal Rydberg CZ gate.

This module implements the model, the gate convention, and the fidelity of the
two-atom CZ control problem solved in

    S. Jandura and G. Pupillo, "Time-Optimal Two- and Three-Qubit Gates for
    Rydberg Atoms", Quantum 6, 712 (2022), doi:10.22331/q-2022-06-20-712,
    arXiv:2202.00903,

Sections 2.1 and 3.1.  Throughout, time is measured in units of
``1 / Omega_max`` and frequencies in units of ``Omega_max``, so a pulse
duration is quoted as the dimensionless product ``T * Omega_max`` exactly as in
the paper.

A *global* pulse drives both atoms with the same complex Rabi frequency
``Omega(t) = Omega_max * exp(-1j * phi(t))``.  Two properties make the problem
small (paper Sec. 2.1.4):

* the Hamiltonian is block diagonal with respect to the number of atoms that
  are not in ``|0>``, and
* for a global pulse the blocks ``q = 01`` and ``q = 10`` are equivalent.

Only two blocks therefore determine the whole gate:

``q = 01``
    spanned by ``|01>`` and ``|0r>`` (equivalently ``|10>`` and ``|r0>``),
``q = 11``
    spanned by ``|11>`` and ``|W> = (|1r> + |r1>) / sqrt(2)``.

In each block the Hamiltonian is the two-level operator of paper Eq. (19),

    H_q(phi) = (Omega_max / 2) * sqrt(m_q) * (cos(phi) sigma_x - sin(phi) sigma_y)
             = (Omega_max / 2) * sqrt(m_q) * [[0, exp(1j phi)], [exp(-1j phi), 0]],

with ``m_q`` the number of 1s in ``q`` (``m_01 = 1``, ``m_11 = 2``); the state
``|rr>`` is decoupled by the infinite blockade.  The laser phase ``phi(t)`` is
the only control because the time-optimal pulse always keeps the amplitude at
the limit ``Omega_max`` (paper Sec. 3.1).

A pulse of duration ``T`` implements the phase gate ``U(T) |q> = exp(1j xi_q)
|q>`` with ``xi_00 = 0``, ``xi_01 = xi_10 = theta`` and ``xi_11 = 2 theta + pi``;
the single-qubit phase ``theta`` is free because it can be removed by local Z
rotations applied after the gate.  ``theta = 0`` is a strict CZ gate.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.interpolate import CubicSpline

from ....errors import BackendValidationError

#: Square roots of ``m_q`` for the blocks ``q = 01`` and ``q = 11`` (paper
#: Eq. (19)).  The first entry belongs to the single-excitation block, whose
#: coupling is ``Omega / 2``; the second one to the double-excitation block,
#: whose coupling is ``sqrt(2) Omega / 2``.
BLOCK_COUPLINGS: Final[tuple[float, float]] = (1.0, math.sqrt(2.0))

#: ``2**n * (2**n + 1)`` for ``n = 2`` qubits, the denominator of the averaged
#: fidelity in paper Eq. (5).
_AVERAGED_FIDELITY_NORMALIZATION: Final[float] = 20.0

#: Accepted values of :attr:``CZPulse.interpolation``.
INTERPOLATION_MODES: Final[tuple[str, ...]] = ("piecewise_constant", "cubic")


def phase_gate_phases(theta: float) -> tuple[float, float, float, float]:
    """Return ``(xi_00, xi_01, xi_10, xi_11)`` of the implemented phase gate.

    Args:
        theta: Single-qubit phase accumulated by ``|01>`` and ``|10>``.

    Returns:
        The four phase-gate angles of paper Sec. 3.1: ``xi_00 = 0``,
        ``xi_01 = xi_10 = theta`` and ``xi_11 = 2 theta + pi``.  A value of
        ``theta = 0`` describes a strict CZ gate.
    """
    return (0.0, theta, theta, 2.0 * theta + math.pi)


def averaged_gate_fidelity(
    single_excitation_amplitude: complex,
    double_excitation_amplitude: complex,
    theta: float,
) -> float:
    """Return the averaged gate fidelity ``F`` of paper Eq. (7).

    ``F`` is the Haar-averaged fidelity of paper Eq. (5) specialized to a
    global two-atom pulse.  It only needs the two diagonal amplitudes
    ``<01|U(T)|01>`` and ``<11|U(T)|11>`` of the propagated computational
    states, which is why a single pair of two-level propagations determines
    the whole two-qubit gate.

    Args:
        single_excitation_amplitude: ``<01|U(T)|01>`` of the ``q = 01`` block.
        double_excitation_amplitude: ``<11|U(T)|11>`` of the ``q = 11`` block.
        theta: Single-qubit phase of :func:``phase_gate_phases``.

    Returns:
        The averaged fidelity in ``[0, 1]``; ``1 - F`` is the gate error used
        as the optimization objective in Sec. 3.1 of the paper.
    """
    a_one = cmath.exp(-1j * theta) * single_excitation_amplitude
    # xi_11 = 2 theta + pi, hence exp(-1j xi_11) = -exp(-2j theta).
    a_two = -cmath.exp(-2j * theta) * double_excitation_amplitude
    return (
        abs(1.0 + 2.0 * a_one + a_two) ** 2
        + 1.0
        + 2.0 * abs(a_one) ** 2
        + abs(a_two) ** 2
    ) / _AVERAGED_FIDELITY_NORMALIZATION


@dataclass(frozen=True)
class CZPulse:
    """Constant-amplitude, phase-modulated global pulse for the two-atom CZ gate.

    A pulse is a phase function ``phi(t)`` on ``0 <= t <= duration`` with the
    amplitude fixed at ``Omega_max``.  Times are dimensionless
    (``t * Omega_max``); divide by ``Omega_max`` to obtain physical times.

    Args:
        duration: Pulse duration ``T * Omega_max``; positive and finite.
        times: Sample times, increasing, starting at exactly ``0.0`` and
            ending at ``duration``.  For ``interpolation="piecewise_constant"``
            these are the ``N + 1`` interval edges of the ``N`` constant
            pieces; for ``interpolation="cubic"`` they are the ``J`` support
            points of the interpolated phase.
        phases: Phase values.  ``N`` values for ``piecewise_constant`` (the
            value on ``[times[j], times[j + 1])``) and ``J`` values matching
            ``times`` for ``cubic``.
        theta: Single-qubit phase of the implemented phase gate, see
            :func:``phase_gate_phases``.
        interpolation: Either ``piecewise_constant`` (the ansatz used by
            GRAPE, paper Sec. 2.2.1) or ``cubic`` (a smoothed, few-parameter
            pulse such as the PMP reconstruction of paper Sec. 4).
        method: Free-form label recording how the pulse was produced, for
            example ``grape`` or ``pmp``.

    Raises:
        BackendValidationError: If the duration, the time grid, the phase
            values, or the interpolation mode are inconsistent.
    """

    duration: float
    times: tuple[float, ...]
    phases: tuple[float, ...]
    theta: float
    interpolation: str = "piecewise_constant"
    method: str = "grape"

    def __post_init__(self) -> None:
        _validate_pulse(self)

    def phase_at(self, time: float) -> float:
        """Return the laser phase ``phi(t)`` at one dimensionless time."""
        if self.interpolation == "cubic":
            return float(self._spline()(time))
        index = int(np.searchsorted(np.asarray(self.times), time, side="right")) - 1
        return float(self.phases[min(max(index, 0), len(self.phases) - 1)])

    def _spline(self) -> CubicSpline:
        return CubicSpline(np.asarray(self.times), np.asarray(self.phases))

    def block_amplitudes(self) -> tuple[complex, complex]:
        """Return ``(<01|U(T)|01>, <11|U(T)|11>)`` of this pulse."""
        return block_amplitudes(self)

    def gate_error(self) -> float:
        """Return the averaged gate error ``1 - F`` at infinite blockade."""
        return 1.0 - averaged_gate_fidelity(*self.block_amplitudes(), self.theta)

    def fidelity(self) -> float:
        """Return the averaged gate fidelity ``F`` at infinite blockade."""
        return averaged_gate_fidelity(*self.block_amplitudes(), self.theta)

    def rydberg_time(self, *, substeps: int = 8) -> float:
        """Return the average Rydberg time ``T_R * Omega_max``, paper Eq. (27)."""
        return rydberg_time(self, substeps=substeps)

    def resampled(self, sample_count: int) -> "CZPulse":
        """Return this pulse sampled on a uniform cubic grid.

        Args:
            sample_count: Number of samples; at least two.

        Returns:
            A ``cubic`` pulse that follows ``phi(t)`` on a uniform grid.  A
            piecewise-constant pulse is sampled at the piece midpoints, so the
            smoothed pulse stays close to the staircase it replaces.
        """
        if sample_count < 2:
            raise BackendValidationError("sample_count must be an integer >= 2")
        edges = np.linspace(0.0, self.duration, sample_count)
        if self.interpolation == "cubic":
            phases = self._spline()(edges)
        else:
            piece_edges = np.asarray(self.times)
            midpoints = 0.5 * (piece_edges[:-1] + piece_edges[1:])
            phases = np.interp(edges, midpoints, np.asarray(self.phases))
        return CZPulse(
            duration=self.duration,
            times=tuple(float(value) for value in edges),
            phases=tuple(float(value) for value in phases),
            theta=self.theta,
            interpolation="cubic",
            method=self.method,
        )

    def physical_times(self, omega_max: float) -> tuple[float, ...]:
        """Return the sample times in microseconds for a Rabi frequency."""
        _require_positive_omega_max(omega_max)
        return tuple(value / omega_max for value in self.times)


def block_amplitudes(pulse: CZPulse, *, substeps: int = 8) -> tuple[complex, complex]:
    """Propagate both control blocks and return the diagonal amplitudes.

    Args:
        pulse: Phase-modulated global pulse in dimensionless units.
        substeps: Number of constant sub-pieces used per authored sample
            interval when ``pulse`` is interpolated smoothly.  Piecewise
            constant pulses are propagated exactly and ignore this argument.

    Returns:
        ``(<01|U(T)|01>, <11|U(T)|11>)``.  Their phases are ``xi_01`` and
        ``xi_11`` of the implemented phase gate, and their moduli are reduced
        by leakage out of the two-level blocks.
    """
    edges, values = _constant_mesh(pulse, substeps=substeps)
    amplitudes = []
    for coupling in BLOCK_COUPLINGS:
        state = np.array([1.0, 0.0], dtype=complex)
        for start, stop, value in zip(edges[:-1], edges[1:], values):
            state = _piece_unitary(value, coupling, stop - start) @ state
        amplitudes.append(complex(state[0]))
    return amplitudes[0], amplitudes[1]


def rydberg_time(pulse: CZPulse, *, substeps: int = 8) -> float:
    """Return the average Rydberg time ``T_R * Omega_max``, paper Eq. (27).

    ``T_R`` is the state-averaged time the two atoms spend in the Rydberg
    state.  It sets the gate error caused by a finite Rydberg lifetime,
    ``1 - F = Gamma * T_R`` (paper Eq. (35) and Appendix B).

    Args:
        pulse: Phase-modulated global pulse in dimensionless units.
        substeps: Number of constant sub-pieces per sample interval; also the
            integration grid of the excited-state population.

    Returns:
        ``T_R * Omega_max``.  Because ``|00>`` never leaves the ground state,
        ``T_R`` is the average over the four computational inputs of the time
        integral of the Rydberg population ``sum_j |r>_j<r|_j``.
    """
    edges, values = _constant_mesh(pulse, substeps=substeps)
    integrals = []
    for coupling in BLOCK_COUPLINGS:
        state = np.array([1.0, 0.0], dtype=complex)
        total = 0.0
        for start, stop, value in zip(edges[:-1], edges[1:], values):
            step = stop - start
            # Simpson rule on the excited-state population of one constant piece.
            offsets = np.linspace(0.0, step, 5)
            local = np.array(
                [
                    abs((_piece_unitary(value, coupling, offset) @ state)[1]) ** 2
                    for offset in offsets
                ]
            )
            total += float(
                step
                / 12.0
                * (
                    local[0]
                    + 4.0 * local[1]
                    + 2.0 * local[2]
                    + 4.0 * local[3]
                    + local[4]
                )
            )
            state = _piece_unitary(value, coupling, step) @ state
        integrals.append(total)
    single, double = integrals
    # q = 00 contributes nothing; q = 01 and q = 10 carry one Rydberg atom each,
    # q = 11 carries one while it is in |W> (the blockade forbids |rr>).
    return float((2.0 * single + double) / 4.0)


def excited_state_populations(
    pulse: CZPulse, times: "np.ndarray | tuple[float, ...]"
) -> tuple[np.ndarray, np.ndarray]:
    """Return the excited-state populations of both control blocks over time.

    The two blocks of the global CZ pulse are two-level systems, so their
    dynamics is a Rabi oscillation between ``|01>`` and ``|0r>`` and between
    ``|11>`` and ``|W>``.  These populations are the inset of Fig. 1(d) of the
    paper and the integrand of the Rydberg time of Eq. (27).

    Args:
        pulse: Phase-modulated global pulse in dimensionless units.
        times: Increasing dimensionless sample times inside the pulse.

    Returns:
        A pair of arrays with the population of ``|0r>`` when starting from
        ``|01>`` and of ``|W>`` when starting from ``|11>``.

    Raises:
        BackendValidationError: If a time falls outside the pulse or the times
            are not increasing.
    """
    requested = np.asarray(times, dtype=float)
    if requested.ndim != 1 or requested.size == 0:
        raise BackendValidationError("times must be a non-empty sequence")
    if np.any(np.diff(requested) < 0.0):
        raise BackendValidationError("times must be increasing")
    if requested[0] < 0.0 or requested[-1] > pulse.duration:
        raise BackendValidationError("times must lie inside the pulse")
    edges, values = _constant_mesh(pulse, substeps=8)
    populations = []
    for coupling in BLOCK_COUPLINGS:
        state = np.array([1.0, 0.0], dtype=complex)
        start = 0.0
        result = np.empty(requested.size)
        index = 0
        for stop, value in zip(edges[1:], values):
            while index < requested.size and requested[index] < stop:
                offset = requested[index] - start
                local = _piece_unitary(value, coupling, offset) @ state
                result[index] = abs(local[1]) ** 2
                index += 1
            state = _piece_unitary(value, coupling, stop - start) @ state
            start = stop
        while index < requested.size:
            result[index] = abs(state[1]) ** 2
            index += 1
        populations.append(result)
    return populations[0], populations[1]


def _piece_unitary(phase: float, coupling: float, duration: float) -> np.ndarray:
    """Return ``exp(-1j H dt)`` of one constant piece of a control block."""
    angle = coupling * duration
    cosine = math.cos(angle / 2.0)
    sine = math.sin(angle / 2.0)
    return np.array(
        [
            [cosine, -1j * sine * cmath.exp(1j * phase)],
            [-1j * sine * cmath.exp(-1j * phase), cosine],
        ],
        dtype=complex,
    )


def _piece_unitary_phase_derivative(
    phase: float, coupling: float, duration: float
) -> np.ndarray:
    """Return ``d/dphi exp(-1j H(phi) dt)`` of one constant piece."""
    sine = math.sin(coupling * duration / 2.0)
    return np.array(
        [
            [0.0, sine * cmath.exp(1j * phase)],
            [-sine * cmath.exp(-1j * phase), 0.0],
        ],
        dtype=complex,
    )


def _constant_mesh(pulse: CZPulse, *, substeps: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a piecewise-constant ``(edges, values)`` representation.

    Piecewise-constant pulses are returned unchanged, so their propagation is
    exact.  Smooth pulses are sampled on a uniform refinement of their support
    with ``substeps`` intervals per authored interval, which approximates the
    exact evolution with an error of order ``(T / (J * substeps))**4``.
    """
    if pulse.interpolation == "piecewise_constant":
        edges = np.asarray(pulse.times, dtype=float)
        return edges, np.asarray(pulse.phases, dtype=float)
    if substeps < 1:
        raise BackendValidationError("substeps must be a positive integer")
    spline = pulse._spline()  # pylint: disable=protected-access
    edges = np.linspace(0.0, pulse.duration, (len(pulse.times) - 1) * substeps + 1)
    values = spline(0.5 * (edges[:-1] + edges[1:]))
    return edges, values


def _validate_pulse(pulse: CZPulse) -> None:
    if isinstance(pulse.duration, bool) or not isinstance(pulse.duration, (int, float)):
        raise BackendValidationError("pulse duration must be a finite number")
    if not math.isfinite(pulse.duration) or pulse.duration <= 0.0:
        raise BackendValidationError("pulse duration must be finite and positive")
    if pulse.interpolation not in INTERPOLATION_MODES:
        raise BackendValidationError(
            "pulse interpolation must be one of " + ", ".join(INTERPOLATION_MODES)
        )
    times = tuple(pulse.times)
    phases = tuple(pulse.phases)
    if len(times) < 2:
        raise BackendValidationError("pulse requires at least two sample times")
    if times[0] != 0.0:
        raise BackendValidationError("pulse times must start at exactly 0.0")
    if any(
        not math.isfinite(earlier) or not math.isfinite(later) or later <= earlier
        for earlier, later in zip(times[:-1], times[1:])
    ):
        raise BackendValidationError("pulse times must be strictly increasing")
    if not math.isclose(times[-1], pulse.duration, rel_tol=0.0, abs_tol=1e-9):
        raise BackendValidationError("pulse times must end at the pulse duration")
    expected = len(times) if pulse.interpolation == "cubic" else len(times) - 1
    if len(phases) != expected:
        raise BackendValidationError(
            f"a {pulse.interpolation} pulse requires {expected} phase values"
        )
    if any(not math.isfinite(value) for value in phases):
        raise BackendValidationError("pulse phases must be finite")
    if not math.isfinite(pulse.theta):
        raise BackendValidationError("pulse theta must be finite")


def _require_positive_omega_max(omega_max: float) -> None:
    if (
        isinstance(omega_max, bool)
        or not isinstance(omega_max, (int, float))
        or not math.isfinite(omega_max)
        or omega_max <= 0.0
    ):
        raise BackendValidationError("omega_max must be finite and positive")
