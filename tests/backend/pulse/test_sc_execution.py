"""Physical qutrit boundaries, guards, and dynamic replay tests."""

import json
from math import pi
from pathlib import Path

import numpy as np
from qutip import Qobj, basis, ket2dm, tensor

import fatqat as fq
from fatqat.backends import MeasurementStep, ResetStep
from fatqat.backends.pulse.backend import PulseBackend
from fatqat.backends.pulse.engine import PulseEngine, _ShotContext
from fatqat.backends.pulse.qutip_adapter import SCQutipAdapter
from fatqat.backends.pulse.resolved import PhaseShift, PulseBlock, SampledControl
from fatqat.backends.pulse.superconducting import (
    load_calibration_spec,
    load_physics_model,
)
from fatqat.noise import NoiseModel, ThermalRelaxation

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


def _backend(noise=None):
    model, calibration = _model_and_calibration()
    return PulseBackend(model, calibration, noise=noise)


def _context(adapter, state, *, classical=(0,), seed=1):
    return _ShotContext(
        state=state,
        classical_memory=list(classical),
        rng=np.random.default_rng(seed),
    )


def test_partial_entangled_measurement_collapses_the_physical_posterior():
    model, _ = _model_and_calibration()
    adapter = SCQutipAdapter(model)
    ket = (tensor(basis(3, 0), basis(3, 0)) + tensor(basis(3, 1), basis(3, 2))).unit()
    context = _context(adapter, ket2dm(ket), seed=4)
    adapter.execute_boundary(
        MeasurementStep((0,), (0,), reported_digit_maps=((0, 1, 2),)),
        context,
    )

    outcome = context.classical_memory[0]
    assert outcome in (0, 1)
    posterior = context.state.ptrace(1)
    expected = ket2dm(basis(3, 0 if outcome == 0 else 2))
    assert np.allclose(posterior.full(), expected.full())


def test_grouped_measurement_preserves_declared_outcome_order():
    model, _ = _model_and_calibration()
    adapter = SCQutipAdapter(model)
    context = _context(
        adapter,
        ket2dm(tensor(basis(3, 1), basis(3, 2))),
        classical=(0, 0),
    )
    adapter.execute_boundary(
        MeasurementStep(
            (1, 0),
            (0, 1),
            reported_digit_maps=((0, 1, 2), (0, 1, 2)),
        ),
        context,
    )
    assert context.classical_memory == [2, 1]


def test_leakage_reports_one_then_confusion_changes_only_classical_value():
    model, _ = _model_and_calibration()
    adapter = SCQutipAdapter(model)
    leaked = ket2dm(tensor(basis(3, 2), basis(3, 0)))
    context = _context(adapter, leaked)
    adapter.execute_boundary(
        MeasurementStep(
            (0,),
            (0,),
            reported_digit_maps=((0, 1, 1),),
            confusions=(np.array([[0.0, 1.0], [1.0, 0.0]]),),
        ),
        context,
    )
    assert context.classical_memory == [0]
    assert np.allclose(context.state.full(), leaked.full())


def test_reset_reprepares_only_target_and_guard_can_skip_it():
    model, _ = _model_and_calibration()
    adapter = SCQutipAdapter(model)
    entangled = (
        tensor(basis(3, 1), basis(3, 0)) + tensor(basis(3, 2), basis(3, 2))
    ).unit()
    context = _context(adapter, ket2dm(entangled))
    adapter.execute_boundary(ResetStep((0,), condition=((0, 1),)), context)
    assert np.allclose(context.state.full(), ket2dm(entangled).full())

    context.classical_memory[0] = 1
    adapter.execute_boundary(ResetStep((0,), condition=((0, 1),)), context)
    expected_other = (ket2dm(basis(3, 0)) + ket2dm(basis(3, 2))) / 2
    expected = tensor(ket2dm(basis(3, 0)), expected_other)
    assert np.allclose(context.state.full(), expected.full())


def test_confused_reported_value_drives_later_guarded_pulse():
    noise = NoiseModel()
    noise.add_readout_error(np.array([[0.0, 1.0], [1.0, 0.0]]), target="q0")
    backend = _backend(noise)
    program = fq.Program(2, 1)
    program.add_measurement(0, 0)
    program.add(fq.ops.RX(pi), 1, condition=(0, 1))
    result = backend.run(
        program,
        shots=1,
        result_config={"counts": True, "final_state": True},
    ).result()

    assert result.get_counts() == {"1": 1}
    density = Qobj(result.get_density_matrix(), dims=[[3, 3], [3, 3]])
    assert density.ptrace(1).diag()[1].real > 0.8
    assert np.allclose(density.ptrace(0).full(), ket2dm(basis(3, 0)).full())


def test_seeded_dynamic_replay_is_reproducible():
    backend = _backend()
    program = fq.Program(1, 1)
    program.add(fq.ops.RX(pi / 2), 0)
    program.add_measurement(0, 0)
    config = {"counts": True, "final_state": False}
    first = backend.run(
        program, shots=40, simulation_config={"seed": 19}, result_config=config
    ).result()
    second = backend.run(
        program, shots=40, simulation_config={"seed": 19}, result_config=config
    ).result()
    assert first.get_counts_as_tuples() == second.get_counts_as_tuples()


def test_real_boundary_preserves_frame_ledger_for_later_drive():
    backend = _backend()
    with_boundary = fq.Program(2, 1)
    with_boundary.add(fq.ops.RZ(0.3), 0)
    with_boundary.add_measurement(1, 0)
    with_boundary.add(fq.ops.RX(0.7), 0)
    boundary_state = (
        backend.run(
            with_boundary,
            shots=1,
            result_config={"counts": False, "final_state": True},
        )
        .result()
        .get_density_matrix()
    )

    continuous = fq.Program(2)
    continuous.add(fq.ops.RZ(0.3), 0)
    continuous.add(fq.ops.RX(0.7), 0)
    continuous_state = (
        backend.run(
            continuous,
            result_config={"counts": False, "final_state": True},
        )
        .result()
        .get_density_matrix()
    )
    assert np.allclose(boundary_state, continuous_state, atol=2e-7)


class _ExcitedAdapter(SCQutipAdapter):
    def initial_state(self):
        return ket2dm(tensor(basis(3, 2), basis(3, 0)))

    def finish_shot(self, context):
        result = super().finish_shot(context)
        return result, dict(context.frame_angles)


def test_false_guard_reserves_noisy_idle_and_skips_controls_and_frames():
    model, _ = _model_and_calibration()
    thermal = ThermalRelaxation(5, 10)
    adapter = _ExcitedAdapter(model, continuous_noise=((thermal,), ()))
    frame = model.frame("q0")
    block = PulseBlock(
        model,
        20.0,
        (
            SampledControl(
                model.drive_control("q0"),
                [0.0, 20.0],
                [10.0, 10.0],
            ),
        ),
        (model.resource("q0"),),
        post_actions=(PhaseShift(frame, 0.7),),
        condition=((0, 1),),
    )
    (outcome,) = PulseEngine(adapter).execute(
        (block,), shots=1, n_clbits=1, rng=np.random.default_rng(5)
    )
    shot, frames = outcome
    density = Qobj(shot.density_matrix, dims=[[3, 3], [3, 3]])
    assert density.ptrace(0).diag()[2].real < 0.1
    assert frames == {}
