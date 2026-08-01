"""Always-on channel selection and qutrit lowering tests."""

import numpy as np
import pytest

import fatqat as fq
from fatqat.simulator import Simulator
from fatqat.emulator.backend import Emulator
from fatqat.emulator.qutip_adapter import SCQutipAdapter
from fatqat.errors import BackendValidationError
from fatqat.noise import (
    Channel,
    Depolarizing,
    NoiseModel,
    ThermalRelaxation,
)
from fatqat.resource_layout import ResourceLayout


def test_thermal_relaxation_validates_finite_t1_t2_bounds():
    assert ThermalRelaxation(t1=10, t2=20) == ThermalRelaxation(t1=10.0, t2=20.0)
    for values in ((0, 1), (1, 0), (1, 2.1), (np.inf, 1), (True, 1)):
        with pytest.raises(ValueError):
            ThermalRelaxation(t1=values[0], t2=values[1])


def test_always_on_selection_specifics_replace_defaults_and_accumulate():
    program = fq.Program(1)
    ref = program.quantum_registers[0][0]
    layout = ResourceLayout({ref: "q0"})
    default = ThermalRelaxation(t1=100, t2=150)
    physical = ThermalRelaxation(t1=200, t2=300)
    logical = ThermalRelaxation(t1=400, t2=600)
    noise = NoiseModel()
    noise.add_channel(default)
    noise.add_channel(physical, targets="q0")
    noise.add_channel(logical, targets=ref)

    noise.validate_for(program, layout)
    assert noise.always_on_channels_for(ref, "q0") == (physical, logical)
    assert noise.always_on_channels_for(None, "q1") == (default,)

    illegal = NoiseModel()
    illegal.add_channel(default, targets="q1")
    with pytest.raises(BackendValidationError, match="effective resource layout"):
        illegal.validate_for(program, layout)


class _UnsupportedAlwaysOn(Channel):
    _num_subsystems = 1


def test_support_reports_reject_unknown_always_on_sources(model, calibration):
    noise = NoiseModel()
    noise.add_channel(_UnsupportedAlwaysOn())
    report = Emulator(model, calibration).validate_noise(noise)
    assert report.rejected_sources == ("_UnsupportedAlwaysOn(always-on)",)
    assert not report.supported


def test_matrix_backend_keeps_gate_channels_and_rejects_always_on_noise():
    noise = NoiseModel()
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.X)
    noise.add_channel(ThermalRelaxation(t1=100, t2=150))
    report = Simulator().validate_noise(noise)
    assert "Depolarizing" in report.accepted_sources
    assert "ThermalRelaxation(always-on)" in report.rejected_sources


def test_pulse_backend_names_each_rejected_gate_channel_source(model, calibration):
    noise = NoiseModel()
    noise.add_channel(Depolarizing(p=0.1), operation=fq.ops.X)
    report = Emulator(model, calibration).validate_noise(noise)
    assert report.rejected_sources == ("Depolarizing",)


def test_qutrit_collapse_coefficients_and_t2_limit_are_exact(model, calibration):
    noise = NoiseModel()
    source = ThermalRelaxation(t1=100, t2=120)
    noise.add_channel(source, targets="q0")
    backend = Emulator(model, calibration, noise=noise)
    adapter = SCQutipAdapter(
        model,
        always_on_noise=backend._always_on_noise(fq.Program(0), ResourceLayout({})),
    )
    collapse = adapter._collapse_operators
    assert len(collapse) == 2
    expected_t1 = np.sqrt(source.amplitude_rate) * adapter._annihilation[0]
    expected_phi = np.sqrt(2 * source.pure_dephasing_rate) * adapter._number[0]
    assert np.allclose(collapse[0](0).full(), expected_t1.full())
    assert np.allclose(collapse[1](0).full(), expected_phi.full())

    limited_noise = NoiseModel()
    limited_noise.add_channel(ThermalRelaxation(t1=100, t2=200), targets="q0")
    limited_backend = Emulator(model, calibration, noise=limited_noise)
    adapter = SCQutipAdapter(
        model,
        always_on_noise=limited_backend._always_on_noise(
            fq.Program(0), ResourceLayout({})
        ),
    )
    assert len(adapter._collapse_operators) == 1


def test_pulse_backend_accepts_and_executes_thermal_relaxation(model, calibration):
    noise = NoiseModel()
    noise.add_channel(ThermalRelaxation(t1=100, t2=150))
    backend = Emulator(model, calibration, noise=noise)
    report = backend.validate_noise(noise)
    assert report.supported
    assert report.accepted_sources == ("ThermalRelaxation(always-on)",)

    program = fq.Program(1)
    program.add(fq.ops.RX(0.3), 0)
    result = backend.run(
        program, result_config={"counts": False, "final_state": True}
    ).result()
    assert result.get_density_matrix().shape == (9, 9)


def test_default_noise_covers_unused_model_subsystems_but_specific_replaces_it(
    model, calibration
):
    program = fq.Program(1)
    backend = Emulator(model, calibration)
    layout = backend._resolve_resource_layout(program)
    default = ThermalRelaxation(t1=100, t2=200)
    specific = ThermalRelaxation(t1=50, t2=100)
    backend._noise_model.add_channel(default)
    backend._noise_model.add_channel(specific, targets=program.quantum_registers[0][0])

    selected = backend._always_on_noise(program, layout)
    assert [term.model_ordinals for term in selected] == [(0,), (1,)]
    assert np.allclose(
        selected[0].local_operator,
        [[0.0, np.sqrt(0.02), 0.0], [0.0, 0.0, np.sqrt(0.04)], [0.0, 0.0, 0.0]],
    )
    assert np.allclose(
        selected[1].local_operator,
        [[0.0, np.sqrt(0.01), 0.0], [0.0, 0.0, np.sqrt(0.02)], [0.0, 0.0, 0.0]],
    )
