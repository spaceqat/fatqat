"""Typed continuous-noise selection and qutrit lowering tests."""

import json
from pathlib import Path

import numpy as np
import pytest

import fatqat as fq
from fatqat.backends import SimulatorBackend
from fatqat.backends.pulse.backend import PulseBackend
from fatqat.backends.pulse.qutip_adapter import SCQutipAdapter
from fatqat.backends.pulse.superconducting import (
    load_calibration_spec,
    load_physics_model,
)
from fatqat.errors import BackendValidationError
from fatqat.noise import (
    ContinuousNoise,
    Depolarizing,
    NoiseModel,
    ThermalRelaxation,
)
from fatqat.resource_layout import ResourceLayout

_FIXTURES = Path(__file__).parent / "fixtures"


def _model_and_calibration():
    model = load_physics_model(
        json.loads((_FIXTURES / "sc_transmon_exchange.json").read_text())
    )
    calibration = load_calibration_spec(
        json.loads((_FIXTURES / "sc_transmon_exchange_calibration.json").read_text()),
        model,
    )
    return model, calibration


def test_thermal_relaxation_validates_finite_t1_t2_bounds():
    assert ThermalRelaxation(10, 20) == ThermalRelaxation(10.0, 20.0)
    for values in ((0, 1), (1, 0), (1, 2.1), (np.inf, 1), (True, 1)):
        with pytest.raises(ValueError):
            ThermalRelaxation(*values)


def test_continuous_selection_specifics_replace_defaults_and_accumulate():
    program = fq.Program(1)
    ref = program.qreg[0][0]
    layout = ResourceLayout({ref: "q0"})
    default = ThermalRelaxation(100, 150)
    physical = ThermalRelaxation(200, 300)
    logical = ThermalRelaxation(400, 600)
    noise = NoiseModel()
    noise.add_continuous_noise(default)
    noise.add_continuous_noise(physical, target="q0")
    noise.add_continuous_noise(logical, target=ref)

    noise.validate_for(program, layout)
    assert noise.continuous_noise_for(ref, "q0") == (physical, logical)
    assert noise.continuous_noise_for(None, "q1") == (default,)

    illegal = NoiseModel()
    illegal.add_continuous_noise(default, target="q1")
    with pytest.raises(BackendValidationError, match="effective resource layout"):
        illegal.validate_for(program, layout)


def test_empty_legacy_shim_is_harmless_but_nonempty_requires_migration():
    program = fq.Program(1)
    ref = program.qreg[0][0]
    layout = ResourceLayout({ref: "q0"})
    noise = NoiseModel()
    noise.validate_for(program, layout)
    noise.qubit_noise["q0"] = object()
    with pytest.raises(BackendValidationError, match="add_continuous_noise"):
        noise.validate_for(program, layout)


class _UnsupportedContinuous(ContinuousNoise):
    pass


def test_support_reports_reject_unknown_continuous_sources_without_spoofing_zz():
    model, calibration = _model_and_calibration()
    noise = NoiseModel()
    noise.add_continuous_noise(_UnsupportedContinuous())
    report = PulseBackend(model, calibration).validate_noise(noise)
    assert report.rejected_sources == ("_UnsupportedContinuous",)
    assert not report.supported


def test_matrix_backend_keeps_gate_channels_and_rejects_typed_continuous_noise():
    noise = NoiseModel()
    noise.add_noise(fq.ops.X, Depolarizing(p=0.1))
    noise.add_continuous_noise(ThermalRelaxation(100, 150))
    report = SimulatorBackend().validate_noise(noise)
    assert "Depolarizing" in report.accepted_sources
    assert "ThermalRelaxation" in report.rejected_sources


def test_qutrit_collapse_coefficients_and_t2_limit_are_exact():
    model, _ = _model_and_calibration()
    source = ThermalRelaxation(100, 120)
    adapter = SCQutipAdapter(model, continuous_noise=((source,), ()))
    collapse = adapter._collapse_operators
    assert len(collapse) == 2
    expected_t1 = np.sqrt(1 / source.T1_ns) * adapter._annihilation[0]
    expected_phi = (
        np.sqrt(2 * (1 / source.T2_ns - 1 / (2 * source.T1_ns))) * adapter._number[0]
    )
    assert np.allclose(collapse[0](0).full(), expected_t1.full())
    assert np.allclose(collapse[1](0).full(), expected_phi.full())

    limited = ThermalRelaxation(100, 200)
    adapter = SCQutipAdapter(model, continuous_noise=((limited,), ()))
    assert len(adapter._collapse_operators) == 1


def test_pulse_backend_accepts_and_executes_thermal_relaxation():
    model, calibration = _model_and_calibration()
    noise = NoiseModel()
    noise.add_continuous_noise(ThermalRelaxation(100, 150))
    backend = PulseBackend(model, calibration, noise=noise)
    report = backend.validate_noise(noise)
    assert report.supported
    assert report.accepted_sources == ("ThermalRelaxation",)

    program = fq.Program(1)
    program.add(fq.ops.RX(0.3), 0)
    result = backend.run(
        program, result_config={"counts": False, "final_state": True}
    ).result()
    assert result.get_density_matrix().shape == (9, 9)


def test_default_noise_covers_unused_model_subsystems_but_specific_replaces_it():
    model, calibration = _model_and_calibration()
    program = fq.Program(1)
    backend = PulseBackend(model, calibration)
    layout = backend._resolve_resource_layout(program)
    default = ThermalRelaxation(100, 200)
    specific = ThermalRelaxation(50, 100)
    backend.noise.add_continuous_noise(default)
    backend.noise.add_continuous_noise(specific, target=program.qreg[0][0])

    selected = backend._continuous_noise(program, layout)
    assert selected == ((specific,), (default,))
