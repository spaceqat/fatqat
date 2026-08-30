"""Three-level atom calibrated-CZ oracle, convergence, and regression.

The four-test two-atom module covers the shipped calibrated ratio only. Its
paired six-test physics gate reported 35.66 seconds under contention; the
parallel module records the corresponding propagation timing and CI allowance.
"""

import pytest

import fatqat as fq
import fatqat.operations as ops
from fatqat.emulator.atom_3level.realization import _CZ_SAMPLE_POINT_COUNT
from fatqat._pulse_values import PulseControl
from tests.emulator.atom_3level.reference.atom_3level_cz_reference import (
    analyze_cz_propagator,
    analytic_cz_oracle,
    principal,
    sampled_cz_reference,
)

CALIBRATED_V_OVER_OMEGA = 100
CALIBRATED_PROCESS_FIDELITY = 0.999908213
CALIBRATED_AVERAGE_FIDELITY = 0.999922925
CALIBRATED_SURVIVAL = 0.999981773
CALIBRATED_LEAKAGE = 1.823e-5
CALIBRATED_ENTANGLING_PHASE = -3.121785
PRODUCTION_TABLE_METRIC_TOLERANCE = 1e-7
PRODUCTION_TABLE_PHASE_TOLERANCE = 1e-6
ANALYTIC_VS_PRODUCTION_METRIC_TOLERANCE = 1e-7
ANALYTIC_VS_PRODUCTION_PHASE_TOLERANCE = 1e-6
PRODUCTION_VS_SAMPLED_METRIC_TOLERANCE = 1e-7
PRODUCTION_VS_SAMPLED_PHASE_TOLERANCE = 1e-6


def _spacing(model, calibration, ratio):
    spacing = (
        abs(model.c6_angular_per_us_um6) / (ratio * calibration.omega_1r_angular_per_us)
    ) ** (1 / 6)
    reconstructed = abs(model.c6_angular_per_us_um6) / (
        spacing**6 * calibration.omega_1r_angular_per_us
    )
    assert reconstructed == pytest.approx(ratio, rel=1e-13)
    return spacing


def _physical_interaction(model, calibration, ratio):
    spacing = _spacing(model, calibration, ratio)
    return abs(model.c6_angular_per_us_um6) / spacing**6


def _program():
    program = fq.Program(2)
    program.add(ops.CZ, (0, 1))
    return program


def _backend(atom_3level_model, atom_3level_calibration, ratio):
    return fq.emulator.Atom3LevelEmulator(
        atom_3level_model,
        arrangement=fq.emulator.AtomArrangement.rectangular(
            1, 2, _spacing(atom_3level_model, atom_3level_calibration, ratio)
        ),
        method="unitary",
    )


def test_analytic_dimensionless_oracle_rederives_the_calibrated_reference(
    atom_3level_calibration,
):
    raw, metrics = analytic_cz_oracle(atom_3level_calibration, CALIBRATED_V_OVER_OMEGA)
    assert metrics.process_fidelity == pytest.approx(
        CALIBRATED_PROCESS_FIDELITY, abs=1e-9
    )
    assert metrics.average_fidelity == pytest.approx(
        CALIBRATED_AVERAGE_FIDELITY, abs=1e-9
    )
    assert metrics.survival == pytest.approx(CALIBRATED_SURVIVAL, abs=1e-9)
    assert metrics.leakage == pytest.approx(CALIBRATED_LEAKAGE, abs=1e-8)
    assert metrics.entangling_phase == pytest.approx(
        CALIBRATED_ENTANGLING_PHASE, abs=1e-6
    )
    assert analyze_cz_propagator(
        raw, atom_3level_calibration, apply_correction=False
    ).local_phases == pytest.approx((-2.099085629, -2.099085629), abs=1e-9)


def test_production_matches_the_calibrated_analytic_oracle(
    atom_3level_model, atom_3level_calibration
):
    backend = _backend(
        atom_3level_model, atom_3level_calibration, CALIBRATED_V_OVER_OMEGA
    )
    produced = backend.run(_program()).result().get_unitary()
    production = analyze_cz_propagator(
        produced, atom_3level_calibration, apply_correction=False
    )
    _raw, analytic = analytic_cz_oracle(
        atom_3level_calibration,
        CALIBRATED_V_OVER_OMEGA,
        physical_interaction_angular_per_us=_physical_interaction(
            atom_3level_model, atom_3level_calibration, CALIBRATED_V_OVER_OMEGA
        ),
    )
    assert production.process_fidelity == pytest.approx(
        CALIBRATED_PROCESS_FIDELITY, abs=PRODUCTION_TABLE_METRIC_TOLERANCE
    )
    assert production.average_fidelity == pytest.approx(
        CALIBRATED_AVERAGE_FIDELITY, abs=PRODUCTION_TABLE_METRIC_TOLERANCE
    )
    assert production.survival == pytest.approx(
        CALIBRATED_SURVIVAL, abs=PRODUCTION_TABLE_METRIC_TOLERANCE
    )
    assert production.leakage == pytest.approx(
        CALIBRATED_LEAKAGE, abs=PRODUCTION_TABLE_METRIC_TOLERANCE
    )
    assert production.entangling_phase == pytest.approx(
        CALIBRATED_ENTANGLING_PHASE, abs=PRODUCTION_TABLE_PHASE_TOLERANCE
    )
    assert production.local_phases == pytest.approx((0.0, 0.0), abs=1e-6)
    double_corrected = analyze_cz_propagator(
        produced, atom_3level_calibration, apply_correction=True
    )
    assert abs(principal(double_corrected.local_phases[0])) > 1.0
    for name in ("process_fidelity", "average_fidelity", "survival", "leakage"):
        delta = getattr(analytic, name) - getattr(production, name)
        assert (
            abs(delta) <= ANALYTIC_VS_PRODUCTION_METRIC_TOLERANCE
        ), f"analytic-production {name} delta={delta:+.3e}"
    phase_delta = principal(analytic.entangling_phase - production.entangling_phase)
    assert (
        abs(phase_delta) <= ANALYTIC_VS_PRODUCTION_PHASE_TOLERANCE
    ), f"analytic-production phase delta={phase_delta:+.3e}"


def test_production_grid_metadata_sampled_reference_and_convergence(
    atom_3level_model, atom_3level_calibration
):
    backend = _backend(
        atom_3level_model, atom_3level_calibration, CALIBRATED_V_OVER_OMEGA
    )
    program = _program()
    points = _CZ_SAMPLE_POINT_COUNT
    plan = backend._prepare_program(program).plan
    controls = [
        control for control in plan[0].controls if isinstance(control, PulseControl)
    ]
    assert {len(control.waveform.times) for control in controls} == {points}
    production = analyze_cz_propagator(
        backend.run(program).result().get_unitary(),
        atom_3level_calibration,
        apply_correction=False,
    )
    _sampled_unitary, sampled = sampled_cz_reference(
        atom_3level_calibration, CALIBRATED_V_OVER_OMEGA, points
    )
    _sampled_double_unitary, doubled = sampled_cz_reference(
        atom_3level_calibration, CALIBRATED_V_OVER_OMEGA, 2 * points
    )
    for name in ("process_fidelity", "average_fidelity", "survival", "leakage"):
        delta = getattr(production, name) - getattr(sampled, name)
        assert (
            abs(delta) <= PRODUCTION_VS_SAMPLED_METRIC_TOLERANCE
        ), f"production-sampled({points}) {name} delta={delta:+.3e}"
    assert (
        abs(principal(production.entangling_phase - sampled.entangling_phase))
        <= PRODUCTION_VS_SAMPLED_PHASE_TOLERANCE
    )
    assert abs(sampled.process_fidelity - doubled.process_fidelity) < 1e-6
    assert abs(sampled.leakage - doubled.leakage) < 1e-6
    assert sampled.local_phases == pytest.approx((0.0, 0.0), abs=1e-6)
    assert doubled.local_phases == pytest.approx((0.0, 0.0), abs=1e-6)
