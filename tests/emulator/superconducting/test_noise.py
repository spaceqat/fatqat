"""Always-on channel selection and qutrit lowering tests."""

import numpy as np
import pytest

import fatqat as fq
from fatqat._index_allocation import _EngineAllocation
from fatqat.simulator import Simulator
from fatqat.emulator.superconducting.backend import TransmonEmulator
from fatqat.emulator.superconducting.qutip_adapter import _TransmonQutipAdapter
from fatqat.emulator.superconducting.target import _TransmonTarget
from fatqat.errors import BackendValidationError
from fatqat.noise import (
    Channel,
    Depolarizing,
    NoiseModel,
    ThermalRelaxation,
)
from fatqat.resource_layout import ResourceLayout


def _adapter(model, **kwargs):
    target = _TransmonTarget(model)
    return _TransmonQutipAdapter(
        target,
        engine_allocation=_EngineAllocation(
            target.device_labels,
            (target.local_dimension,) * len(target.device_labels),
        ),
        **kwargs,
    )


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
    by_ref = ThermalRelaxation(t1=400, t2=600)
    noise = NoiseModel()
    noise.add_channel(default)
    noise.add_channel(physical, targets="q0")
    noise.add_channel(by_ref, targets=ref)

    noise.validate_for(program, layout.device_labels)
    assert noise.always_on_channels_for(ref, "q0") == (physical, by_ref)
    assert noise.always_on_channels_for(None, "q1") == (default,)

    illegal = NoiseModel()
    illegal.add_channel(default, targets="q1")
    with pytest.raises(BackendValidationError, match="legal device universe"):
        illegal.validate_for(program, layout.device_labels)


class _UnsupportedAlwaysOn(Channel):
    _num_subsystems = 1


def test_support_reports_reject_unknown_always_on_sources(model, calibration):
    noise = NoiseModel()
    noise.add_channel(_UnsupportedAlwaysOn())
    report = TransmonEmulator(model).validate_noise(noise)
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
    report = TransmonEmulator(model).validate_noise(noise)
    assert report.rejected_sources == ("Depolarizing",)


def test_qutrit_collapse_coefficients_and_t2_limit_are_exact(model, calibration):
    noise = NoiseModel()
    source = ThermalRelaxation(t1=100, t2=120)
    noise.add_channel(source, targets="q0")
    backend = TransmonEmulator(model, noise=noise)
    adapter = _adapter(
        model,
        always_on_noise=backend._prepare_program(fq.Program(1)).always_on_noise,
    )
    collapse = adapter._always_on_collapse_operators()
    assert len(collapse) == 2
    expected_t1 = np.sqrt(source.amplitude_rate) * adapter._annihilation[0]
    expected_phi = np.sqrt(2 * source.pure_dephasing_rate) * adapter._number[0]
    assert np.allclose(collapse[0](0).full(), expected_t1.full())
    assert np.allclose(collapse[1](0).full(), expected_phi.full())

    limited_noise = NoiseModel()
    limited_noise.add_channel(ThermalRelaxation(t1=100, t2=200), targets="q0")
    limited_backend = TransmonEmulator(model, noise=limited_noise)
    adapter = _adapter(
        model,
        always_on_noise=limited_backend._prepare_program(fq.Program(1)).always_on_noise,
    )
    assert len(adapter._always_on_collapse_operators()) == 1


def test_pulse_backend_accepts_and_executes_thermal_relaxation(model, calibration):
    noise = NoiseModel()
    noise.add_channel(ThermalRelaxation(t1=100, t2=150))
    backend = TransmonEmulator(model, noise=noise)
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
    backend = TransmonEmulator(model)
    default = ThermalRelaxation(t1=100, t2=200)
    specific = ThermalRelaxation(t1=50, t2=100)
    backend._noise_model.add_channel(default)
    backend._noise_model.add_channel(specific, targets=program.quantum_registers[0][0])

    selected = backend._prepare_program(program).always_on_noise
    assert [term.engine_indices for term in selected] == [(0,), (1,)]
    assert np.allclose(
        selected[0].local_operator,
        [[0.0, np.sqrt(0.02), 0.0], [0.0, 0.0, np.sqrt(0.04)], [0.0, 0.0, 0.0]],
    )
    assert np.allclose(
        selected[1].local_operator,
        [[0.0, np.sqrt(0.01), 0.0], [0.0, 0.0, np.sqrt(0.02)], [0.0, 0.0, 0.0]],
    )
