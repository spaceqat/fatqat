"""Optimal-phase-control contracts for the two-atom Rydberg CZ gate.

The reference for every number asserted here is

    S. Jandura and G. Pupillo, "Time-Optimal Two- and Three-Qubit Gates for
    Rydberg Atoms", Quantum 6, 712 (2022), doi:10.22331/q-2022-06-20-712.

The paper values used as expectations are the time-optimal duration
``T* Omega_max = 7.612``, the threshold curvature ``A = 0.0544``, the average
Rydberg time ``T_R Omega_max = 2.957``, the blockade quality factor
``alpha = 35.9``, and the gate errors of Fig. 7 at ``B / 2 pi = 180 MHz``.
"""

import math

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.emulator._atom_3level import Atom3LevelEmulator
from fatqat.emulator._atom_3level.optimal_cz import (
    PAPER_CZ_COSTATES,
    PAPER_CZ_COSTATE_DURATION,
    CZPulse,
    averaged_gate_fidelity,
    block_amplitudes,
    cz_gate_implementation_map,
    emulated_cz_metrics,
    emulated_decay_fidelity,
    finite_blockade_error,
    fit_time_optimal_duration,
    optimize_cz_pulse,
    quadratic_blockade_coefficient,
    reconstruct_pulse,
    rydberg_decay_error,
    rydberg_time,
    scan_durations,
)

PAPER_TIME_OPTIMAL_DURATION = 7.612
PAPER_RYDBERG_TIME = 2.957
PAPER_BLOCKADE_COEFFICIENT = 35.9
PAPER_CZ_ERROR_AT_180MHZ = 4.6e-4
LIFETIME_US = 540.0


def _constant_pulse(duration, pieces, phase=0.0, theta=0.0):
    edges = np.linspace(0.0, duration, pieces + 1)
    return CZPulse(
        duration=duration,
        times=tuple(float(value) for value in edges),
        phases=tuple([phase] * pieces),
        theta=theta,
    )


def test_constant_phase_pulse_matches_the_analytic_rabi_solution():
    coupling, duration, pieces = 1.0, 3.0, 4000
    pulse = _constant_pulse(duration, pieces)
    amplitude, _ = block_amplitudes(pulse)

    assert amplitude == pytest.approx(math.cos(coupling * duration / 2.0), abs=1e-6)
    # The excited-state population is sin^2(g t / 2); T_R averages it over the
    # two single-excitation blocks and the double-excitation block.
    single = duration / 2 - math.sin(coupling * duration) / (2 * coupling)
    double_coupling = math.sqrt(2.0)
    double = duration / 2 - math.sin(double_coupling * duration) / (2 * double_coupling)
    assert rydberg_time(pulse) == pytest.approx((2 * single + double) / 4, abs=1e-6)


def test_identity_pulse_reproduces_the_paper_identity_gate_error():
    # Paper Sec. 3.1: at T = 0 the gate error is 0.4, the identity with
    # theta = pi / 2.
    pulse = _constant_pulse(1e-3, 1, theta=math.pi / 2.0)

    assert averaged_gate_fidelity(1.0, 1.0, math.pi / 2.0) == pytest.approx(0.6)
    assert pulse.gate_error() == pytest.approx(0.4, abs=1e-6)


def test_pulse_validation_rejects_inconsistent_grids():
    with pytest.raises(Exception, match="phase values"):
        CZPulse(duration=1.0, times=(0.0, 1.0), phases=(0.0, 0.0), theta=0.0)
    with pytest.raises(Exception, match="strictly increasing"):
        CZPulse(duration=1.0, times=(0.0, 0.5, 0.5), phases=(0.0, 0.0), theta=0.0)


def test_grape_reaches_the_time_optimal_pulse_of_the_paper():
    pulse = optimize_cz_pulse(
        PAPER_TIME_OPTIMAL_DURATION, pieces=99, seed=1, max_iterations=4000
    )

    assert pulse.gate_error() < 1e-9
    # The paper reports T_R Omega_max = 2.957 for this pulse, and the phase
    # profile of its Fig. 1(d) rises to about 1.0, dips to about -0.4 and ends
    # near 0.7.
    assert pulse.rydberg_time() == pytest.approx(PAPER_RYDBERG_TIME, abs=0.05)
    profile = [pulse.phase_at(value) for value in np.linspace(0.0, 7.5, 30)]
    assert max(profile) == pytest.approx(1.0, abs=0.15)
    assert min(profile) == pytest.approx(-0.4, abs=0.15)


def test_duration_sweep_reproduces_the_time_optimal_threshold():
    scan = scan_durations(longest=7.7, shortest=7.55, samples=16, pieces=49, seed=3)
    fit = fit_time_optimal_duration(scan)

    # The paper fits T* Omega_max = 7.612 with A = 0.0544 for 99 pieces.
    assert fit.optimal_duration == pytest.approx(PAPER_TIME_OPTIMAL_DURATION, abs=0.02)
    assert fit.coefficient == pytest.approx(0.0544, rel=0.35)
    assert scan.gate_errors[0] < 1e-6
    assert min(scan.gate_errors) < 1e-9


def test_pmp_reconstruction_of_the_paper_costates_is_time_optimal():
    result = reconstruct_pulse(
        PAPER_CZ_COSTATES, PAPER_CZ_COSTATE_DURATION, samples=401
    )

    # The paper reports 1 - F = 3.1e-10 for these costates.
    assert result.gate_error < 1e-8
    assert result.control_margin > 0.0
    assert result.pulse.rydberg_time() == pytest.approx(PAPER_RYDBERG_TIME, abs=0.05)


def test_quadratic_blockade_coefficient_matches_the_paper():
    pulse = optimize_cz_pulse(
        PAPER_TIME_OPTIMAL_DURATION, pieces=99, seed=1, max_iterations=4000
    )

    assert quadratic_blockade_coefficient(pulse) == pytest.approx(
        PAPER_BLOCKADE_COEFFICIENT, rel=0.02
    )


def test_error_budget_reproduces_figure_seven():
    pulse = optimize_cz_pulse(
        PAPER_TIME_OPTIMAL_DURATION, pieces=99, seed=1, max_iterations=4000
    )
    decay_rate = 1.0 / LIFETIME_US
    blockade = 2.0 * math.pi * 180.0

    frequency, error = gate_error_minimum(pulse, decay_rate, blockade)
    assert error == pytest.approx(PAPER_CZ_ERROR_AT_180MHZ, rel=0.1)
    assert frequency / (2.0 * math.pi) == pytest.approx(2.8e6 * 1e-6, rel=0.15)


def gate_error_minimum(pulse, decay_rate, blockade):
    from fatqat.emulator._atom_3level.optimal_cz import optimal_rabi_frequency

    return optimal_rabi_frequency(pulse, decay_rate=decay_rate, blockade=blockade)


def test_exported_pulse_implements_the_cz_gate_in_the_emulator(atom_3level_model):
    pulse = optimize_cz_pulse(7.612, pieces=49, seed=1, max_iterations=2000)
    omega_max = 2.0 * math.pi * 5.0

    metrics = emulated_cz_metrics(
        atom_3level_model,
        pulse,
        omega_max=omega_max,
        blockade=100.0 * omega_max,
        sample_count=401,
    )

    assert metrics.average_fidelity > 0.9999
    assert metrics.leakage < 1e-4
    assert abs(metrics.entangling_phase) == pytest.approx(math.pi, abs=0.03)
    assert abs(metrics.local_phases[0] - metrics.local_phases[1]) < 1e-3


def test_emulated_blockade_error_follows_the_inverse_square_law(atom_3level_model):
    pulse = optimize_cz_pulse(7.612, pieces=49, seed=1, max_iterations=2000)
    omega_max = 2.0 * math.pi * 5.0
    blockade = 50.0 * omega_max

    metrics = emulated_cz_metrics(
        atom_3level_model,
        pulse,
        omega_max=omega_max,
        blockade=blockade,
        sample_count=401,
    )
    predicted = finite_blockade_error(pulse, blockade=blockade, omega_max=omega_max)

    assert metrics.gate_error == pytest.approx(predicted, rel=0.15)


def test_emulated_decay_error_tracks_the_first_order_estimate(atom_3level_model):
    pulse = optimize_cz_pulse(7.612, pieces=49, seed=1, max_iterations=2000)
    omega_max = 2.0 * math.pi * 5.0
    decay_rate = 2.0 * math.pi * 0.01

    result = emulated_decay_fidelity(
        atom_3level_model,
        pulse,
        decay_rate=decay_rate,
        omega_max=omega_max,
        blockade=100.0 * omega_max,
        sample_count=801,
    )
    predicted = rydberg_decay_error(pulse, decay_rate=decay_rate, omega_max=omega_max)

    # The emulator integrates the exact Lindblad decay inside the three-level
    # space, so it agrees with Gamma T_R only up to higher orders in Gamma.
    assert result.emulated_error == pytest.approx(predicted, rel=0.6)
    assert 0.0 < result.leakage < 1e-2
    assert result.perturbative_error == pytest.approx(predicted)


def test_exported_gate_map_runs_a_cz_program(atom_3level_model):
    pulse = optimize_cz_pulse(7.612, pieces=25, seed=5, max_iterations=1000)
    omega_max = 2.0 * math.pi * 5.0
    spacing = (abs(atom_3level_model.c6_angular_per_us_um6) / (100.0 * omega_max)) ** (
        1.0 / 6.0
    )
    backend = Atom3LevelEmulator(
        atom_3level_model,
        arrangement=fq.emulator.AtomArrangement.chain(2, spacing),
        method="unitary",
        gate_implementation_map=cz_gate_implementation_map(
            atom_3level_model, pulse, omega_max=omega_max, sample_count=401
        ),
    )
    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))

    unitary = backend.run(program).result().get_unitary()

    assert unitary.shape == (9, 9)
    assert abs(unitary[0, 0]) == pytest.approx(1.0, abs=1e-6)
    assert np.angle(unitary[4, 4]) == pytest.approx(-math.pi, abs=0.05)
