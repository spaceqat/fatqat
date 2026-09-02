"""Background generator selection and qutrit lowering tests."""

import numpy as np
import pytest
from qutip import Qobj

import fatqat as fq
import fatqat.operations as ops
from fatqat._index_allocation import _EngineAllocation
from fatqat._pulse_values import PulseControl
from fatqat.emulator import SampledWaveform
from fatqat.emulator.superconducting.backend import TransmonEmulator
from fatqat.emulator.superconducting.qutip_adapter import _TransmonQutipAdapter
from fatqat.emulator.superconducting.target import _TransmonTarget
from fatqat.errors import BackendValidationError
from fatqat.noise import (
    AmplitudeDamping,
    Channel,
    Depolarizing,
    NoiseModel,
    ReadoutConfusion,
    ThermalRelaxation,
    TransitionRelaxation,
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


def test_background_logical_physical_alias_rejects_on_actual_match():
    program = fq.Program(1)
    ref = program.quantum_registers[0][0]
    layout = ResourceLayout({ref: "q0"})
    physical = Depolarizing(rate=0.01)
    by_ref = Depolarizing(rate=0.02)
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


def test_noise_validation_rejects_unknown_background_sources(model, calibration):
    noise = NoiseModel()
    noise.add(_UnsupportedAlwaysOn(), targets="q0")
    with pytest.raises(
        BackendValidationError,
        match="_UnsupportedAlwaysOn.*no registered Lindblad implementation",
    ):
        TransmonEmulator(model).validate_noise_model(noise)


def test_transmon_accepts_binary_and_eagerly_rejects_nonbinary_readout_confusion(
    model,
):
    valid = NoiseModel()
    valid.add(ReadoutConfusion(np.eye(2)))
    assert TransmonEmulator(model, noise=valid).validate_noise_model(valid) is None

    invalid = NoiseModel()
    invalid.add(ReadoutConfusion(np.eye(3)))
    with pytest.raises(BackendValidationError, match="2 x 2"):
        TransmonEmulator(model).validate_noise_model(invalid)
    with pytest.raises(BackendValidationError, match="2 x 2"):
        TransmonEmulator(model, noise=invalid)


def test_rate_depolarization_populates_the_full_qutrit_space(model):
    rate = 0.7
    duration = 0.4
    noise = NoiseModel()
    noise.add(Depolarizing(rate=rate), targets="q0")
    backend = TransmonEmulator(model, method="density_matrix", noise=noise)
    program = fq.Program(1)
    program.add(
        ops.PulseOperation(
            duration,
            (
                PulseControl(
                    model.control.drive("q0"),
                    SampledWaveform((0.0, duration), (0.0, 0.0)),
                ),
            ),
        )
    )

    density = (
        backend.run(
            program,
            result_config={"counts": False, "final_state": True},
        )
        .result()
        .get_density_matrix()
    )
    q0_state = Qobj(density, dims=[[3, 3], [3, 3]]).ptrace(0)

    expected_population = (1.0 - np.exp(-rate * duration)) / 3.0
    assert q0_state.diag()[2].real == pytest.approx(expected_population, abs=2e-7)


@pytest.mark.parametrize(
    "channel",
    [
        AmplitudeDamping(rate=0.01),
        ThermalRelaxation(t1=100, t2=150),
    ],
)
def test_transmon_rejects_qubit_only_relaxation(model, calibration, channel):
    noise = NoiseModel()
    noise.add(channel, targets="q0")
    with pytest.raises(
        BackendValidationError,
        match=f"{type(channel).__name__}.*no registered Lindblad implementation",
    ):
        TransmonEmulator(model, noise=noise)


def test_explicit_background_noise_covers_referenced_and_unused_subsystems(
    model, calibration
):
    program = fq.Program(1)
    noise = NoiseModel()
    noise.add(
        TransitionRelaxation(
            rate=0.02,
            coefficients={(1, 0): 1, (2, 1): np.sqrt(2)},
        ),
        targets="q0",
    )
    noise.add(
        TransitionRelaxation(
            rate=0.01,
            coefficients={(1, 0): 1, (2, 1): np.sqrt(2)},
        ),
        targets="q1",
    )
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
