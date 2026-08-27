"""Background generator selection and qutrit lowering tests."""

import numpy as np
import pytest

import fatqat as fq
import fatqat.operations as ops
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


def test_background_logical_physical_alias_rejects_on_actual_match():
    program = fq.Program(1)
    ref = program.quantum_registers[0][0]
    layout = ResourceLayout({ref: "q0"})
    physical = ThermalRelaxation(t1=200, t2=300)
    by_ref = ThermalRelaxation(t1=400, t2=600)
    noise = NoiseModel()
    noise.add(physical, targets="q0")
    noise.add(by_ref, targets=ref)

    noise._validate_for(program, layout.device_labels)
    with pytest.raises(BackendValidationError, match="both match"):
        noise._background_noise_for(ref, "q0")
    assert noise._background_noise_for(None, "q1") == ()

    illegal = NoiseModel()
    illegal.add(physical, targets="q1")
    with pytest.raises(BackendValidationError, match="legal universe"):
        illegal._validate_for(program, layout.device_labels)


class _UnsupportedAlwaysOn(Channel):
    num_subsystems = 1


def test_support_reports_reject_unknown_background_sources(model, calibration):
    noise = NoiseModel()
    noise.add(_UnsupportedAlwaysOn(), targets="q0")
    report = TransmonEmulator(model).check_noise_support(noise)
    assert report.rejected_sources == ("_UnsupportedAlwaysOn(background)",)
    assert not report.supported


def test_matrix_backend_keeps_gate_channels_and_rejects_background_noise():
    noise = NoiseModel()
    noise.add(Depolarizing(p=0.1), operation=ops.X)
    noise.add(ThermalRelaxation(t1=100, t2=150), targets=0)
    report = Simulator().check_noise_support(noise)
    assert "Depolarizing(p)" in report.accepted_sources
    assert "ThermalRelaxation(background)" in report.rejected_sources


def test_pulse_backend_names_each_rejected_gate_channel_source(model, calibration):
    noise = NoiseModel()
    noise.add(Depolarizing(p=0.1), operation=ops.X)
    report = TransmonEmulator(model).check_noise_support(noise)
    assert report.rejected_sources == ("Depolarizing(p)",)


def test_qutrit_collapse_coefficients_and_t2_limit_are_exact(model, calibration):
    noise = NoiseModel()
    source = ThermalRelaxation(t1=100, t2=120)
    noise.add(source, targets="q0")
    backend = TransmonEmulator(model, noise=noise)
    adapter = _adapter(
        model,
        background_noise=backend._prepare_program(fq.Program(1)).background_noise,
    )
    collapse = adapter._background_collapse_operators()
    assert len(collapse) == 2
    expected_t1 = np.sqrt(source.amplitude_rate) * adapter._annihilation[0]
    expected_phi = np.sqrt(2 * source.pure_dephasing_rate) * adapter._number[0]
    assert np.allclose(collapse[0](0).full(), expected_t1.full())
    assert np.allclose(collapse[1](0).full(), expected_phi.full())

    limited_noise = NoiseModel()
    limited_noise.add(ThermalRelaxation(t1=100, t2=200), targets="q0")
    limited_backend = TransmonEmulator(model, noise=limited_noise)
    adapter = _adapter(
        model,
        background_noise=limited_backend._prepare_program(
            fq.Program(1)
        ).background_noise,
    )
    assert len(adapter._background_collapse_operators()) == 1


def test_pulse_backend_accepts_and_executes_thermal_relaxation(model, calibration):
    noise = NoiseModel()
    noise.add(ThermalRelaxation(t1=100, t2=150), targets="q0")
    backend = TransmonEmulator(model, noise=noise)
    report = backend.check_noise_support(noise)
    assert report.supported
    assert report.accepted_sources == ("ThermalRelaxation(background)",)

    program = fq.Program(1)
    program.add(ops.RX(0.3), 0)
    result = backend.run(
        program, result_config={"counts": False, "final_state": True}
    ).result()
    assert result.get_density_matrix().shape == (9, 9)


def test_explicit_background_noise_covers_referenced_and_unused_subsystems(
    model, calibration
):
    program = fq.Program(1)
    noise = NoiseModel()
    noise.add(ThermalRelaxation(t1=50, t2=100), targets="q0")
    noise.add(ThermalRelaxation(t1=100, t2=200), targets="q1")
    backend = TransmonEmulator(model, noise=noise)

    selected = backend._prepare_program(program).background_noise
    assert [term.engine_indices for term in selected] == [(0,), (1,)]
    assert np.allclose(
        selected[0].local_operator,
        [[0.0, np.sqrt(0.02), 0.0], [0.0, 0.0, np.sqrt(0.04)], [0.0, 0.0, 0.0]],
    )
    assert np.allclose(
        selected[1].local_operator,
        [[0.0, np.sqrt(0.01), 0.0], [0.0, 0.0, np.sqrt(0.02)], [0.0, 0.0, 0.0]],
    )
