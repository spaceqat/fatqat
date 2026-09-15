"""Semi-analytical pulse description from Pontryagin's maximum principle.

Reference: S. Jandura and G. Pupillo, "Time-Optimal Two- and Three-Qubit Gates
for Rydberg Atoms", Quantum 6, 712 (2022), doi:10.22331/q-2022-06-20-712,
Sec. 4 and Table 1.

GRAPE returns a staircase of a few hundred phases, which is inconvenient to
report, to implement, and to reproduce.  The maximum principle shows that the
time-optimal pulse is a *smooth* solution of a small set of ordinary
differential equations that is fixed by the initial costates alone.  This
module implements the three steps of Sec. 4 of the paper:

1. :func:``extract_costates`` turns a GRAPE pulse into a starting guess for the
   initial costates with the singular-value construction of paper Eqs. (24)-(26).
2. :func:``reconstruct_pulse`` integrates the state and costate Schroedinger
   equations together with the maximization condition of paper Eqs. (20)-(23),
   which yields the smooth pulse ``phi(t)``.
3. :func:``refine_costates`` minimizes the gate error over the initial costates
   and the pulse duration, as the paper does with Powell's method.

Step 1 alone is a guess, not a result: the reconstructed pulse reacts very
sensitively to the initial costates, exactly as the paper reports, so the
refinement of step 3 is what turns the guess into a pulse of the same fidelity
as the GRAPE staircase.  The values the paper reports in Table 1 for the CZ
gate are available as :data:``PAPER_CZ_COSTATES``; they reproduce a gate error
of ``3.8e-10`` at ``T Omega_max = 7.6114828`` through
:func:``reconstruct_pulse``.

The costates are two two-component vectors.  The tangent-space gauge
``Re <chi_q(0)|psi_q(0)> = 0`` removes two of the eight real parameters and the
overall scale is irrelevant, so the pulse is described by six parameters of
which five matter, plus the duration.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from ....errors import BackendValidationError
from .problem import BLOCK_COUPLINGS, CZPulse, averaged_gate_fidelity

#: Pauli matrices in the two-dimensional control blocks.
_SIGMA_X: Final[np.ndarray] = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SIGMA_Y: Final[np.ndarray] = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)

#: Threshold on ``A**2 + B**2`` of paper Eq. (21).  The maximization condition
#: has a unique solution only while this quantity is nonzero.
_DEGENERACY_THRESHOLD: Final[float] = 1e-14


@dataclass(frozen=True)
class PMPCostates:
    """Initial costates ``|chi_q(0)>`` of the two control blocks.

    Args:
        single: Components of ``|chi_01(0)>`` in the ``(|01>, |0r>)`` basis.
        double: Components of ``|chi_11(0)>`` in the ``(|11>, |W>)`` basis.

    The overall scale of the two vectors is irrelevant, because the
    maximization condition of paper Eq. (21) is homogeneous in the costates.
    """

    single: tuple[complex, complex]
    double: tuple[complex, complex]

    def as_vector(self) -> tuple[float, ...]:
        """Return the eight real parameters ``(Re chi1, Im chi1, ...)``."""
        values: list[float] = []
        for component in (*self.single, *self.double):
            values.extend((component.real, component.imag))
        return tuple(values)

    @classmethod
    def from_vector(cls, values: "np.ndarray | tuple[float, ...]") -> "PMPCostates":
        """Build costates from the eight real parameters of :meth:``as_vector``."""
        array = np.asarray(values, dtype=float)
        if array.shape != (8,):
            raise BackendValidationError("costate vector must have eight entries")
        components = [
            complex(array[2 * index], array[2 * index + 1]) for index in range(4)
        ]
        return cls(
            single=(components[0], components[1]),
            double=(components[2], components[3]),
        )


#: Initial costates of the time-optimal CZ pulse, Table 1 of the paper.
#: The entries are ``<0|chi_1(0)>, <1|chi_1(0)>, <0|chi_2(0)>, <1|chi_2(0)>`` for
#: the blocks ``q = 01`` and ``q = 11``; the reported dimensionless duration is
#: ``T Omega_max = 7.6114828`` with gate error ``1 - F = 3.1e-10``.
PAPER_CZ_COSTATES: Final[PMPCostates] = PMPCostates(
    single=(0.11212523j, complex(-0.70546052, -0.21533291)),
    double=(-0.06056999j, complex(0.51526141, -0.41739917)),
)

#: Dimensionless duration belonging to :data:``PAPER_CZ_COSTATES``.
PAPER_CZ_COSTATE_DURATION: Final[float] = 7.6114828


@dataclass(frozen=True)
class PMPReconstruction:
    """A smooth pulse reconstructed from initial costates.

    Args:
        pulse: The reconstructed smooth pulse.
        costates: Initial costates that generated the pulse.
        theta: Optimized single-qubit phase of the implemented phase gate.
        gate_error: Averaged gate error ``1 - F`` of the reconstructed pulse.
        control_margin: Smallest value of ``A**2 + B**2`` encountered while
            integrating, in units of the squared costate norm.  The
            maximization condition of paper Eq. (21) is unique only while this
            value stays positive.
    """

    pulse: CZPulse
    costates: PMPCostates
    theta: float
    gate_error: float
    control_margin: float


def extract_costates(pulse: CZPulse, *, sample_count: int | None = None) -> PMPCostates:
    """Estimate the initial costates of a GRAPE pulse, paper Eqs. (24)-(26).

    Along a time-optimal trajectory the derivative of the maximization
    condition with respect to the phase vanishes at every time, which gives the
    linear constraint ``sum_q Im <chi_q(0)|alpha_q(t)> = 0`` with the vectors
    ``|alpha_q>`` of paper Eq. (25).  Stacking one row per sample time and
    block and taking the right-singular vector of the smallest singular value
    gives an approximate non-zero solution, which the paper uses as the
    starting point of the costate optimization.

    Args:
        pulse: Piecewise-constant GRAPE pulse at (or close to) its optimal
            duration.
        sample_count: Number of sample times; by default one sample per
            constant piece.  A few dozen samples are enough, because the
            constraint only has eight unknowns.

    Returns:
        Initial costates in the tangent-space gauge
        ``Re <chi_q(0)|psi_q(0)> = 0`` of paper Sec. 4.  They are a starting
        guess for :func:``refine_costates``, not a converged description of
        the pulse.

    Raises:
        BackendValidationError: If the pulse is not piecewise constant or the
            sample count is too small.
    """
    if pulse.interpolation != "piecewise_constant":
        raise BackendValidationError(
            "costate extraction requires a piecewise-constant GRAPE pulse"
        )
    samples = len(pulse.phases) if sample_count is None else sample_count
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 4:
        raise BackendValidationError("sample_count must be an integer >= 4")
    times = np.linspace(0.0, pulse.duration, samples + 2)[1:-1]
    alphas = [_block_alphas(pulse, coupling, times) for coupling in BLOCK_COUPLINGS]
    # Paper Eqs. (24)-(26) impose one constraint per sample time, summing the
    # two blocks.  The tangent-space gauge Re <chi_q(0)|psi_q(0)> = 0 removes
    # the two components along |q>, so six real numbers remain:
    # chi_q(0) = i a_q |q> + (b_q + i c_q) |x_q>.
    rows = np.zeros((len(times), 6))
    for block, block_alphas in enumerate(alphas):
        offset = 3 * block
        for index, alpha in enumerate(block_alphas):
            rows[index, offset + 0] = -alpha[0].real
            rows[index, offset + 1] = alpha[1].imag
            rows[index, offset + 2] = -alpha[1].real
    _, _, right = np.linalg.svd(rows)
    vector = right[-1]
    return PMPCostates(
        single=(complex(0.0, vector[0]), complex(vector[1], vector[2])),
        double=(complex(0.0, vector[3]), complex(vector[4], vector[5])),
    )


def reconstruct_pulse(
    costates: PMPCostates,
    duration: float,
    *,
    samples: int = 401,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> PMPReconstruction:
    """Integrate the PMP equations into a smooth pulse, paper Eqs. (20)-(23).

    Args:
        costates: Initial costates fixing the trajectory.
        duration: Pulse duration ``T * Omega_max``.
        samples: Number of uniform samples of the reconstructed phase.
        rtol: Relative tolerance of the ODE integration.
        atol: Absolute tolerance of the ODE integration.

    Returns:
        The reconstructed :class:``PMPReconstruction``.  Its ``pulse`` is a
        ``cubic`` :class:``~.problem.CZPulse`` on a uniform grid; call
        :meth:``~.problem.CZPulse.resampled`` to reduce the number of control
        points.

    Raises:
        BackendValidationError: If the duration or the sample count is invalid,
            if the integration fails, or if the maximization condition of
            paper Eq. (21) becomes degenerate.
    """
    _validate_duration(duration)
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 2:
        raise BackendValidationError("samples must be an integer >= 2")
    initial = np.concatenate(
        [
            np.array([1.0, 0.0], dtype=complex),
            np.asarray(costates.single, dtype=complex),
            np.array([1.0, 0.0], dtype=complex),
            np.asarray(costates.double, dtype=complex),
        ]
    )
    scale = float(np.linalg.norm(np.asarray(costates.as_vector())))
    if scale == 0.0:
        raise BackendValidationError("costates must not vanish")
    grid = np.linspace(0.0, duration, samples)

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        phase, margin = _phase_from_costates(state)
        if margin < _DEGENERACY_THRESHOLD * scale**2:
            raise BackendValidationError(
                "the PMP maximization condition is degenerate for these costates"
            )
        return _costate_derivative(state, phase)

    solution = solve_ivp(
        rhs,
        (0.0, duration),
        initial,
        method="DOP853",
        t_eval=grid,
        rtol=rtol,
        atol=atol,
    )
    if not solution.success:
        raise BackendValidationError(f"PMP integration failed: {solution.message}")
    phases = np.empty(samples)
    margins = np.empty(samples)
    for index in range(samples):
        phases[index], margin = _phase_from_costates(solution.y[:, index])
        margins[index] = margin / scale**2
    theta, error = _best_theta_and_error(
        complex(solution.y[0, -1]), complex(solution.y[4, -1])
    )
    pulse = CZPulse(
        duration=duration,
        times=tuple(float(value) for value in grid),
        phases=tuple(float(value) for value in phases),
        theta=theta,
        interpolation="cubic",
        method="pmp",
    )
    return PMPReconstruction(
        pulse=pulse,
        costates=costates,
        theta=theta,
        gate_error=error,
        control_margin=float(np.min(margins)),
    )


def refine_costates(
    costates: PMPCostates,
    duration: float,
    *,
    optimize_duration: bool = True,
    max_iterations: int = 200,
    samples: int = 101,
    rtol: float = 1e-8,
) -> PMPReconstruction:
    """Minimize the gate error over the initial costates and the duration.

    This is the last step of the paper's procedure: Powell's method in the
    eight-dimensional costate space (plus the duration) replaces the GRAPE
    staircase by a smooth pulse of the same fidelity.

    Args:
        costates: Starting guess, usually from :func:``extract_costates`` or
            from :data:``PAPER_CZ_COSTATES``.
        duration: Starting duration ``T * Omega_max``.
        optimize_duration: Whether the duration is a free parameter as well.
        max_iterations: Iteration limit of the optimizer.  Every iteration
            costs about ten integrations of the trial pulse, so the default
            trades accuracy for a runtime of roughly one minute.
        samples: Number of samples of each trial pulse; only the gate error
            enters the objective, so a coarse grid is enough.
        rtol: Relative tolerance of the trial integrations.

    Returns:
        The refined reconstruction.  Degenerate trial costates are rejected by
        the objective instead of raising.
    """
    _validate_duration(duration)
    start = list(costates.as_vector())
    if optimize_duration:
        start.append(float(duration))
    best = {"error": math.inf, "costates": costates, "duration": duration}

    def objective(parameters: np.ndarray) -> float:
        trial = PMPCostates.from_vector(parameters[:8])
        trial_duration = float(parameters[8]) if optimize_duration else duration
        if trial_duration <= 0.0:
            return 1.0
        try:
            result = reconstruct_pulse(
                trial, trial_duration, samples=samples, rtol=rtol, atol=1e-10
            )
        except BackendValidationError:
            return 1.0
        if result.gate_error < best["error"]:
            best.update(
                error=result.gate_error, costates=trial, duration=trial_duration
            )
        return result.gate_error

    minimize(
        objective,
        np.asarray(start),
        method="Powell",
        options={"maxiter": max_iterations, "xtol": 1e-10, "ftol": 1e-12},
    )
    return reconstruct_pulse(
        best["costates"], best["duration"], samples=max(samples, 401)
    )


def pmp_cz_pulse(
    pulse: CZPulse,
    *,
    refine: bool = False,
    samples: int = 401,
    max_iterations: int = 2000,
) -> PMPReconstruction:
    """Turn a GRAPE pulse into the paper's few-parameter smooth pulse.

    Args:
        pulse: GRAPE result, typically from
            :func:``~.grape.time_optimal_cz_pulse``.
        refine: Whether to run :func:``refine_costates`` after the
            singular-value guess.  Leave it ``False`` to obtain the raw guess
            quickly; set it to ``True`` for the paper's fully variational
            smooth pulse, which costs about a minute.
        samples: Number of samples of the reconstructed pulse.
        max_iterations: Iteration limit when ``refine`` is set.

    Returns:
        The smooth pulse and the initial costates describing it.
    """
    costates = extract_costates(pulse)
    if refine:
        return refine_costates(
            costates, pulse.duration, samples=samples, max_iterations=max_iterations
        )
    return reconstruct_pulse(costates, pulse.duration, samples=samples)


def costates_from_paper_table() -> PMPCostates:
    """Return the CZ costates of Table 1 of the paper.

    Returns:
        :data:``PAPER_CZ_COSTATES``, provided as a function so that callers do
        not depend on the module-level name.
    """
    return PAPER_CZ_COSTATES


def _block_alphas(
    pulse: CZPulse, coupling: float, times: np.ndarray
) -> list[tuple[complex, complex]]:
    """Return ``|alpha_q(t)>`` of paper Eq. (25) at the requested times.

    ``|alpha_q(t)> = U_q(t)^dagger (dH_q/dphi) |psi_q(t)>`` is evaluated
    exactly inside each constant piece of ``pulse``, so the result does not
    depend on the piece grid.

    Args:
        pulse: Piecewise-constant pulse.
        coupling: ``sqrt(m_q)`` of the block.
        times: Sample times inside the pulse.

    Returns:
        One pair of complex components per requested time.

    Raises:
        BackendValidationError: If a sample time falls outside the pulse.
    """
    edges = np.asarray(pulse.times, dtype=float)
    values = np.asarray(pulse.phases, dtype=float)
    if np.any(times < edges[0]) or np.any(times > edges[-1]):
        raise BackendValidationError("sample times must lie inside the pulse")
    result: list[tuple[complex, complex]] = []
    propagator = np.eye(2, dtype=complex)
    state = np.array([1.0, 0.0], dtype=complex)
    for start, stop, phase in zip(edges[:-1], edges[1:], values):
        inside = (times >= start) & (times < stop)
        for time in times[inside]:
            partial = _block_piece_unitary(phase, coupling, time - start)
            propagator_at = partial @ propagator
            state_at = partial @ state
            generator = _block_phase_derivative(phase, coupling)
            alpha = propagator_at.conj().T @ generator @ state_at
            result.append((complex(alpha[0]), complex(alpha[1])))
        step = stop - start
        propagator = _block_piece_unitary(phase, coupling, step) @ propagator
        state = _block_piece_unitary(phase, coupling, step) @ state
    if len(result) != len(times):
        raise BackendValidationError("sample times must lie inside the pulse")
    return result


def _block_piece_unitary(phase: float, coupling: float, duration: float) -> np.ndarray:
    """Return ``exp(-1j H_q dt)`` of one constant piece."""
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


def _block_phase_derivative(phase: float, coupling: float) -> np.ndarray:
    """Return ``dH_q/dphi = (g_q / 2) (-sin(phi) sigma_x - cos(phi) sigma_y)``."""
    return coupling / 2.0 * (-math.sin(phase) * _SIGMA_X - math.cos(phase) * _SIGMA_Y)


def _costate_derivative(state: np.ndarray, phase: float) -> np.ndarray:
    """Return the right-hand side of the state and costate equations.

    Both ``|psi_q>`` and ``|chi_q>`` evolve with the same Schroedinger
    equation, see the text below Eq. (14) of the paper.
    """
    derivative = np.empty_like(state)
    for block, coupling in enumerate(BLOCK_COUPLINGS):
        hamiltonian = (
            coupling / 2.0 * (math.cos(phase) * _SIGMA_X - math.sin(phase) * _SIGMA_Y)
        )
        generator = -1j * hamiltonian
        offset = 4 * block
        derivative[offset : offset + 2] = generator @ state[offset : offset + 2]
        derivative[offset + 2 : offset + 4] = generator @ state[offset + 2 : offset + 4]
    return derivative


def _phase_from_costates(state: np.ndarray) -> tuple[float, float]:
    """Return the phase of paper Eq. (21) and the margin ``A**2 + B**2``.

    ``A`` and ``B`` are the quantities of paper Eqs. (22) and (23).  With
    ``H_q = (g_q / 2) (cos(phi) sigma_x - sin(phi) sigma_y)`` the maximized
    control is ``phi = arctan2(-B, A)``, which is paper Eq. (21).
    """
    first = 0.0
    second = 0.0
    for block, coupling in enumerate(BLOCK_COUPLINGS):
        offset = 4 * block
        psi = state[offset : offset + 2]
        chi = state[offset + 2 : offset + 4]
        first += coupling * np.imag(np.vdot(chi, _SIGMA_X @ psi))
        second += coupling * np.imag(np.vdot(chi, _SIGMA_Y @ psi))
    return math.atan2(-second, first), float(abs(first) ** 2 + abs(second) ** 2)


def _project_to_tangent_space(costates: PMPCostates) -> PMPCostates:
    """Impose ``Re <chi_q(0)|psi_q(0)> = 0`` of paper Sec. 4.

    With ``|psi_q(0)> = |0>`` this removes the real part of the first costate
    component.  The removed piece is proportional to the state, which leaves
    the control of Eq. (21) unchanged.
    """
    single = (complex(0.0, costates.single[0].imag), costates.single[1])
    double = (complex(0.0, costates.double[0].imag), costates.double[1])
    return PMPCostates(single=single, double=double)


def _best_theta_and_error(one: complex, two: complex) -> tuple[float, float]:
    """Return the best single-qubit phase and the resulting gate error."""
    coarse = np.linspace(-math.pi, math.pi, 721)
    errors = [1.0 - averaged_gate_fidelity(one, two, float(theta)) for theta in coarse]
    index = int(np.argmin(errors))
    fine = np.linspace(
        coarse[max(index - 1, 0)], coarse[min(index + 1, len(coarse) - 1)], 201
    )
    fine_errors = [
        1.0 - averaged_gate_fidelity(one, two, float(theta)) for theta in fine
    ]
    best = int(np.argmin(fine_errors))
    return float(fine[best]), float(fine_errors[best])


def _validate_duration(duration: float) -> None:
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0.0
    ):
        raise BackendValidationError("duration must be finite and positive")
