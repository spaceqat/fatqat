"""Gate-error analysis of the optimized two-atom CZ pulse.

Reference: S. Jandura and G. Pupillo, "Time-Optimal Two- and Three-Qubit Gates
for Rydberg Atoms", Quantum 6, 712 (2022), doi:10.22331/q-2022-06-20-712,
Secs. 5-7 and Appendices B and E.

Two error sources limit the time-optimal CZ gate and both are analyzed here.

Decay of the Rydberg state
    The paper adds the non-Hermitian term ``-1j (Gamma / 2) |r><r|`` per atom
    to the Hamiltonian and shows that a pulse that is exact without decay has
    the gate error

        1 - F = Gamma * T_R                                        (Eq. 35)

    to first order in the decay rate, where ``T_R`` is the state-averaged time
    the atoms spend in the Rydberg state (Eq. 27).
    :func:``rydberg_decay_error`` evaluates that estimate;
    :func:``emulated_decay_fidelity`` instead runs the exact Lindblad decay of
    FATQAT's three-level atom family and reports the resulting Bell-state
    fidelity, which is not restricted to first order.

Finite blockade strength
    To first order in ``1 / B`` the finite blockade only ac-Stark shifts the
    state ``|W>`` by ``-|Omega|^2 / (2 B)`` (Eq. 31).  The gate error then
    grows quadratically in ``1 / B`` (Eq. 32) and is written as

        1 - F = alpha * Omega_max^2 / (B^2 (T Omega_max)^2)        (Eq. 35)

    with the dimensionless quality factor ``alpha = (1 - F) B^2 T^2``.
    :func:``quadratic_blockade_coefficient`` evaluates ``alpha`` from Eqs. (32)
    and (33), and :func:``emulated_cz_metrics`` measures the same error
    exactly, without the ``1 / B`` expansion, by running the full two-atom
    three-level propagator of the emulator at a finite ``C6 / d^6``.

Both contributions add up in the budget of Eq. (35), which
:func:``gate_error_budget`` evaluates.  Because the two terms scale as
``1 / Omega_max`` and ``Omega_max^2``, the total error has a minimum at an
optimal Rabi frequency, which :func:``optimal_rabi_frequency`` returns and
which reproduces Fig. 7 of the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.linalg import expm

import fatqat as fq
import fatqat.operations as ops
from ....errors import BackendValidationError
from ...._pulse_values import PulseControl
from ...._waveforms import SampledWaveform
from ....noise import NoiseModel, TransitionRelaxation
from ..backend import Atom3LevelEmulator
from ..calibration import default_atom_3level_calibration
from ..model import Atom3LevelModel
from .problem import CZPulse, _constant_mesh, rydberg_time
from .pulses import DEFAULT_SAMPLE_COUNT, cz_gate_implementation_map

#: Indices of the two-qubit computational subspace inside the nine-dimensional
#: two-qutrit space of the emulator (level order ``|0>, |1>, |r>``).
COMPUTATIONAL_INDICES: Final[tuple[int, ...]] = (0, 1, 3, 4)

#: Block coupling of the ``q = 11`` block, ``sqrt(m_q)`` of paper Eq. (19).
_DOUBLE_BLOCK_COUPLING: Final[float] = math.sqrt(2.0)


@dataclass(frozen=True)
class GateErrorBudget:
    """Gate error of one pulse at a given decay rate and blockade strength.

    Args:
        decay_error: Contribution ``Gamma T_R`` of the Rydberg lifetime.
        blockade_error: Contribution of the finite blockade strength.
        optimal_rabi_frequency: Rabi frequency that minimizes the total error.
        minimal_error: Total gate error at ``optimal_rabi_frequency``.
    """

    decay_error: float
    blockade_error: float
    optimal_rabi_frequency: float
    minimal_error: float

    @property
    def total_error(self) -> float:
        """Return ``decay_error + blockade_error`` at the given frequency."""
        return self.decay_error + self.blockade_error


@dataclass(frozen=True)
class CZGateMetrics:
    """Metrics of one two-atom CZ propagator.

    The definitions match the reference analysis of the emulator test suite:
    ``process_fidelity`` is ``|Tr(CZ^dagger U)|^2 / 16`` on the computational
    subspace, ``survival`` is ``Tr(U^dagger U) / 4`` there, and
    ``average_fidelity`` is ``(4 process + survival) / 5``.

    Args:
        process_fidelity: Process fidelity with an ideal CZ gate.
        average_fidelity: Averaged gate fidelity.
        survival: Population that remains in the computational subspace.
        leakage: ``1 - survival``, population outside the subspace.
        entangling_phase: Conditional phase; ``pi`` for an ideal CZ gate.
        local_phases: Single-qubit phases of ``|01>`` and ``|10>``.
        blockade: Blockade strength ``C6 / d^6`` in rad/us used for the run.
    """

    process_fidelity: float
    average_fidelity: float
    survival: float
    leakage: float
    entangling_phase: float
    local_phases: tuple[float, float]
    blockade: float

    @property
    def gate_error(self) -> float:
        """Return ``1 - average_fidelity``."""
        return 1.0 - self.average_fidelity


@dataclass(frozen=True)
class DecayFidelity:
    """Bell-state fidelity of the optimized pulse under Rydberg decay.

    Args:
        bell_fidelity: Fidelity of the prepared Bell state with the ideal
            output of the CZ gate.
        survival: Population that remains in the computational subspace.
        leakage: ``1 - survival``.
        decay_rate: Rydberg decay rate in rad/us used for the run.
        perturbative_error: First-order estimate ``Gamma T_R`` of the paper.
    """

    bell_fidelity: float
    survival: float
    leakage: float
    decay_rate: float
    perturbative_error: float

    @property
    def emulated_error(self) -> float:
        """Return the exact emulated gate error ``1 - bell_fidelity``."""
        return 1.0 - self.bell_fidelity


def quadratic_blockade_coefficient(pulse: CZPulse, *, substeps: int = 8) -> float:
    """Return ``alpha = (1 - F) B^2 T^2`` of paper Sec. 6.2.

    The coefficient follows from the first-order expansion of paper
    Eqs. (31)-(33): the perturbation ``H1 = -(1/2) |W><W|`` (in units of
    ``Omega_max``) is propagated together with the unperturbed state, and

        ``(1 - F) B^2 = <psi1|psi1> / 4 - |<11|psi1>|^2 / 10``

    gives the quadratic growth of the gate error with ``1 / B``.

    Args:
        pulse: Pulse in the dimensionless units of the paper.
        substeps: Number of constant sub-pieces per sample interval.

    Returns:
        Dimensionless ``alpha``.  The time-optimal CZ pulse of the paper has
        ``alpha = 35.9``, and the pulse that minimizes it at long durations
        approaches ``alpha = 28``.
    """
    edges, values = _constant_mesh(pulse, substeps=substeps)
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0
    for start, stop, phase in zip(edges[:-1], edges[1:], values):
        state = expm(_blockade_generator(phase) * (stop - start)) @ state
    first_order = state[2:]
    norm = float(np.vdot(first_order, first_order).real)
    return pulse.duration**2 * (0.25 * norm - 0.1 * abs(first_order[0]) ** 2)


def rydberg_decay_error(
    pulse: CZPulse, *, decay_rate: float, omega_max: float
) -> float:
    """Return the first-order decay error ``Gamma T_R``, paper Eq. (35).

    Args:
        pulse: Pulse in the dimensionless units of the paper.
        decay_rate: Rydberg decay rate ``Gamma`` in rad/us, that is
            ``1 / lifetime``.
        omega_max: Rabi frequency in rad/us that scales the pulse.

    Returns:
        The gate error caused by the finite Rydberg lifetime, valid for
        ``Gamma T_R << 1``.

    Raises:
        BackendValidationError: If a rate is not finite and positive.
    """
    _require_positive(decay_rate, "decay_rate")
    _require_positive(omega_max, "omega_max")
    return decay_rate * rydberg_time(pulse) / omega_max


def finite_blockade_error(
    pulse: CZPulse, *, blockade: float, omega_max: float
) -> float:
    """Return the finite-blockade error of paper Eq. (35).

    Args:
        pulse: Pulse in the dimensionless units of the paper.
        blockade: Blockade strength ``B = C6 / d^6`` in rad/us.
        omega_max: Rabi frequency in rad/us that scales the pulse.

    Returns:
        ``alpha Omega_max^2 / (B^2 (T Omega_max)^2)``, the gate error caused by
        the finite blockade to leading order in ``1 / B``.

    Raises:
        BackendValidationError: If a rate is not finite and positive.
    """
    _require_positive(blockade, "blockade")
    _require_positive(omega_max, "omega_max")
    alpha = quadratic_blockade_coefficient(pulse)
    return alpha * omega_max**2 / (blockade**2 * pulse.duration**2)


def gate_error_budget(
    pulse: CZPulse,
    *,
    decay_rate: float,
    blockade: float,
    omega_max: float,
) -> GateErrorBudget:
    """Return the error budget of paper Eq. (35) at one Rabi frequency.

    Args:
        pulse: Pulse in the dimensionless units of the paper.
        decay_rate: Rydberg decay rate ``Gamma`` in rad/us.
        blockade: Blockade strength ``B`` in rad/us.
        omega_max: Rabi frequency in rad/us.

    Returns:
        The two error contributions and the frequency that minimizes their
        sum, which is ``(Gamma T_R B^2 (T Omega_max)^2 / (2 alpha))^(1/3)``.
    """
    decay_error = rydberg_decay_error(pulse, decay_rate=decay_rate, omega_max=omega_max)
    blockade_error = finite_blockade_error(
        pulse, blockade=blockade, omega_max=omega_max
    )
    best_frequency, best_error = optimal_rabi_frequency(
        pulse, decay_rate=decay_rate, blockade=blockade
    )
    return GateErrorBudget(
        decay_error=decay_error,
        blockade_error=blockade_error,
        optimal_rabi_frequency=best_frequency,
        minimal_error=best_error,
    )


def optimal_rabi_frequency(
    pulse: CZPulse, *, decay_rate: float, blockade: float
) -> tuple[float, float]:
    """Return the Rabi frequency that minimizes the error of Eq. (35).

    Args:
        pulse: Pulse in the dimensionless units of the paper.
        decay_rate: Rydberg decay rate ``Gamma`` in rad/us.
        blockade: Blockade strength ``B`` in rad/us.

    Returns:
        A pair ``(omega_max, minimal_error)`` in rad/us and dimensionless
        units.  The pulse shape is kept fixed, as in Fig. 7 of the paper,
        where the gate error is minimized over the Rabi frequency alone.
    """
    _require_positive(decay_rate, "decay_rate")
    _require_positive(blockade, "blockade")
    alpha = quadratic_blockade_coefficient(pulse)
    rydberg = rydberg_time(pulse)
    frequency = (
        decay_rate * rydberg * blockade**2 * pulse.duration**2 / (2.0 * alpha)
    ) ** (1.0 / 3.0)
    error = decay_rate * rydberg / frequency + alpha * frequency**2 / (
        blockade**2 * pulse.duration**2
    )
    return frequency, error


def emulated_cz_metrics(
    model: Atom3LevelModel,
    pulse: CZPulse,
    *,
    omega_max: float,
    blockade: float,
    sample_count: int | None = None,
) -> CZGateMetrics:
    """Measure the exact finite-blockade gate error with the emulator.

    Instead of expanding the dynamics in ``1 / B``, this function runs the full
    two-atom three-level propagator of
    :class:``~..backend.Atom3LevelEmulator`` with the optimized pulse and
    reports the metrics of the resulting 4x4 computational block.  The
    emulator models the blockade as ``C6 / d^6`` and does not project the state
    ``|rr>`` out, so the result is exact for the requested blockade strength.

    Args:
        model: Three-level atom model, used for its Rydberg control addresses
            and its ``C6`` coefficient.
        pulse: Pulse in the dimensionless units of the paper.
        omega_max: Rabi frequency in rad/us that scales the pulse.
        blockade: Requested blockade strength ``C6 / d^6`` in rad/us.  The
            atom spacing is chosen as ``(|C6| / B)^(1/6)``.
        sample_count: Number of waveform samples; see
            :func:``~.pulses.cz_waveform``.

    Returns:
        The CZ metrics of the emulated propagator.

    Raises:
        BackendValidationError: If the model, the pulse, or the blockade is
            invalid, or if the emulator reports an execution failure.
    """
    _require_positive(blockade, "blockade")
    spacing = (abs(model.c6_angular_per_us_um6) / blockade) ** (1.0 / 6.0)
    propagator = _emulated_propagator(
        model,
        pulse,
        omega_max=omega_max,
        spacing=spacing,
        sample_count=sample_count,
    )
    return _cz_metrics(propagator, blockade=blockade)


def emulated_decay_fidelity(
    model: Atom3LevelModel,
    pulse: CZPulse,
    *,
    decay_rate: float,
    omega_max: float,
    blockade: float,
    branching_to_zero: float = 0.0,
    sample_count: int | None = None,
) -> DecayFidelity:
    """Measure the effect of a finite Rydberg lifetime with the emulator.

    The optimized pulse is applied to ``|++>`` with a Lindblad decay of the
    Rydberg state attached to the CZ operation only, and the fidelity of the
    resulting density matrix with the ideal output of the CZ gate is reported.
    Unlike the first-order estimate of :func:``rydberg_decay_error``, the
    emulator integrates the full master equation, so the result is exact for
    the modeled channel.

    The emulator keeps the population inside the three-level space: the decay
    either repopulates ``|1>`` or ``|0>``, while the non-Hermitian model of the
    paper removes it from the space.  The two descriptions therefore agree only
    to leading order in ``Gamma``, and the emulated error is the more
    conservative statement about a real experiment.

    Args:
        model: Three-level atom model.
        pulse: Pulse in the dimensionless units of the paper.
        decay_rate: Rydberg decay rate ``Gamma`` in rad/us.
        omega_max: Rabi frequency in rad/us that scales the pulse.
        blockade: Blockade strength ``C6 / d^6`` in rad/us, fixing the spacing.
        branching_to_zero: Fraction of the decay that reaches ``|0>`` instead
            of ``|1>``; ``0.0`` describes decay into the qubit state ``|1>``.
        sample_count: Number of waveform samples; see
            :func:``~.pulses.cz_waveform``.

    Returns:
        The Bell-state fidelity, the computational survival, and the
        first-order estimate of the paper for comparison.

    Raises:
        BackendValidationError: If an argument is invalid or the emulator
            reports an execution failure.
    """
    _require_positive(decay_rate, "decay_rate")
    _require_positive(blockade, "blockade")
    if not 0.0 <= branching_to_zero <= 1.0:
        raise BackendValidationError("branching_to_zero must lie in [0, 1]")
    spacing = (abs(model.c6_angular_per_us_um6) / blockade) ** (1.0 / 6.0)
    backend = Atom3LevelEmulator(
        model,
        arrangement=fq.emulator.AtomArrangement.chain(2, spacing),
        method="density_matrix",
        noise=_decay_noise(decay_rate, branching_to_zero),
    )
    program = fq.Program(2)
    program.add(
        _prepared_cz_operation(
            model, pulse, omega_max=omega_max, sample_count=sample_count
        )
    )
    job = backend.run(
        program,
        shots=1,
        result_config={"counts": False, "final_state": True},
    )
    density = job.result().get_density_matrix()
    # State and density-matrix results are reported before the virtual frame
    # ledger of the pulse is applied, so the reference state is the phase gate
    # that the pulse implements, not its CZ correction.
    ideal = _ideal_phase_gate_output(pulse.theta)
    fidelity = float(np.real(np.vdot(ideal, density @ ideal)))
    survival = float(np.real(np.trace(density)) - _rydberg_population(density))
    return DecayFidelity(
        bell_fidelity=fidelity,
        survival=survival,
        leakage=1.0 - survival,
        decay_rate=decay_rate,
        perturbative_error=rydberg_decay_error(
            pulse, decay_rate=decay_rate, omega_max=omega_max
        ),
    )


def _blockade_generator(phase: float) -> np.ndarray:
    """Return the generator of paper Eq. (33) for one constant piece.

    The state is ordered as ``(|psi0_11>, |psi1_11>)`` with two components
    each, where ``|psi0>`` is the unperturbed state and ``|psi1>`` its
    first-order correction in ``1 / B``.  In units of ``Omega_max`` the block
    Hamiltonian is ``H0 = (sqrt(2) / 2) [[0, e^{1j phi}], [e^{-1j phi}, 0]]``
    and the blockade perturbation is ``H1 = -(1/2) |W><W|``, see paper
    Eq. (31).
    """
    coupling = _DOUBLE_BLOCK_COUPLING / 2.0
    hamiltonian = coupling * np.array(
        [
            [0.0, np.exp(1j * phase)],
            [np.exp(-1j * phase), 0.0],
        ],
        dtype=complex,
    )
    perturbation = -0.5 * np.diag([0.0, 1.0]).astype(complex)
    generator = np.zeros((4, 4), dtype=complex)
    generator[:2, :2] = hamiltonian
    generator[2:, 2:] = hamiltonian
    generator[2:, :2] = perturbation
    return -1j * generator


def _decay_noise(decay_rate: float, branching_to_zero: float) -> NoiseModel:
    """Return background decay of the Rydberg state on both sites.

    The jump operators act on the Rydberg component only, so the background
    scope is equivalent to attaching the decay to the CZ operation alone: the
    state preparation never populates ``|r>``.
    """
    noise = NoiseModel()
    for site in (0, 1):
        noise.add(
            TransitionRelaxation(
                rate=decay_rate * (1.0 - branching_to_zero),
                coefficients={(2, 1): 1.0},
            ),
            targets=site,
        )
        if branching_to_zero > 0.0:
            noise.add(
                TransitionRelaxation(
                    rate=decay_rate * branching_to_zero,
                    coefficients={(2, 0): 1.0},
                ),
                targets=site,
            )
    return noise


def _prepared_cz_operation(
    model: Atom3LevelModel,
    pulse: CZPulse,
    *,
    omega_max: float,
    sample_count: int | None,
) -> ops.PulseOperation:
    """Return the ``|++>`` preparation and the optimized CZ in one pulse block.

    The operation drives the Raman transition of both sites for a quarter of a
    Rabi period and applies the optimized Rydberg pulse afterwards.  Both
    waveforms cover the whole block with explicit samples, including their zero
    segments, because the emulator evaluates every authored waveform over the
    entire continuous pulse run.
    """
    omega_01 = default_atom_3level_calibration().omega_01_angular_per_us
    preparation = 0.5 * math.pi / omega_01
    total = preparation + pulse.duration / omega_max
    count = DEFAULT_SAMPLE_COUNT if sample_count is None else sample_count
    grid = np.linspace(0.0, total, count)
    raman = np.where(
        grid <= preparation, omega_01 * np.exp(-1j * (-0.5 * math.pi)), 0.0
    )
    offsets = np.clip((grid - preparation) * omega_max, 0.0, pulse.duration)
    phases = np.array([pulse.phase_at(float(value)) for value in offsets])
    rydberg = np.where(grid >= preparation, omega_max * np.exp(-1j * phases), 0.0)
    return ops.PulseOperation(
        total,
        (
            PulseControl(model.control.raman(0), SampledWaveform(grid, raman)),
            PulseControl(model.control.raman(1), SampledWaveform(grid, raman)),
            PulseControl(model.control.rydberg(0), SampledWaveform(grid, rydberg)),
            PulseControl(model.control.rydberg(1), SampledWaveform(grid, rydberg)),
        ),
    )


def _emulated_propagator(
    model: Atom3LevelModel,
    pulse: CZPulse,
    *,
    omega_max: float,
    spacing: float,
    sample_count: int | None,
) -> np.ndarray:
    """Return the 9x9 two-qutrit propagator of the optimized CZ pulse."""
    backend = Atom3LevelEmulator(
        model,
        arrangement=fq.emulator.AtomArrangement.chain(2, spacing),
        method="unitary",
        gate_implementation_map=cz_gate_implementation_map(
            model, pulse, omega_max=omega_max, sample_count=sample_count
        ),
    )
    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))
    return backend.run(program).result().get_unitary()


def _cz_metrics(propagator: np.ndarray, *, blockade: float) -> CZGateMetrics:
    """Return the CZ metrics of a 9x9 physical propagator."""
    if propagator.shape != (9, 9):
        raise BackendValidationError("CZ analysis requires a 9 x 9 propagator")
    indices = np.asarray(COMPUTATIONAL_INDICES)
    block = propagator[np.ix_(indices, indices)]
    ideal = np.diag([1.0, 1.0, 1.0, -1.0])
    process = float(abs(np.trace(ideal.conj().T @ block)) ** 2 / 16.0)
    survival = float(np.trace(block.conj().T @ block).real / 4.0)
    phases = (
        _principal(float(np.angle(block[1, 1]))),
        _principal(float(np.angle(block[2, 2]))),
    )
    entangling = _principal(
        float(
            np.angle(block[3, 3])
            - np.angle(block[2, 2])
            - np.angle(block[1, 1])
            + np.angle(block[0, 0])
        )
    )
    return CZGateMetrics(
        process_fidelity=process,
        average_fidelity=float((4.0 * process + survival) / 5.0),
        survival=survival,
        leakage=1.0 - survival,
        entangling_phase=entangling,
        local_phases=phases,
        blockade=blockade,
    )


def _ideal_phase_gate_output(theta: float) -> np.ndarray:
    """Return the ideal image of ``|++>`` under the implemented phase gate.

    The pulse implements the phase gate with ``xi_01 = xi_10 = theta`` and
    ``xi_11 = 2 theta + pi`` (paper Sec. 3.1), so the reference state is
    ``(|00> + e^{1j theta} (|01> + |10>) - e^{2j theta} |11>) / 2``.
    """
    state = np.zeros(9, dtype=complex)
    state[0] = 0.5
    state[1] = 0.5 * np.exp(1j * theta)
    state[3] = 0.5 * np.exp(1j * theta)
    state[4] = -0.5 * np.exp(2j * theta)
    return state


def _rydberg_population(density: np.ndarray) -> float:
    """Return the population of the states that contain a Rydberg excitation."""
    total = 0.0
    for row in range(9):
        for column in range(9):
            if row == column and (row // 3 == 2 or row % 3 == 2):
                total += float(np.real(density[row, column]))
    return total


def _principal(angle: float) -> float:
    """Return an angle folded into ``(-pi, pi]``."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _require_positive(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise BackendValidationError(f"{name} must be finite and positive")
