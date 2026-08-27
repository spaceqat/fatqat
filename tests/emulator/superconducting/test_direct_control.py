"""End-to-end superconducting direct-control regression tests."""

import numpy as np
import pytest
from qutip import Qobj, qeye, tensor

import fatqat as fq
import fatqat.operations as ops
from fatqat._pulse_values import PulseControl
from fatqat._index_allocation import _EngineAllocation
from fatqat.emulator._core.engine import PulseEngine
from fatqat.emulator.superconducting.qutip_adapter import _TransmonQutipAdapter
from fatqat.emulator.superconducting.target import _TransmonTarget
from fatqat.errors import BackendValidationError
from fatqat.emulator import SampledWaveform


def _control(channel, values, *, duration=1.0, offset=0.0):
    return PulseControl(
        channel,
        SampledWaveform((0.0, duration), values),
        start_offset=offset,
    )


def test_direct_detuning_and_exchange_have_exact_engine_targets(backend):
    detuning = ops.PulseOperation(
        1.0,
        (_control(backend.model.control.detuning("q0"), (0.1, 0.1)),),
    )
    exchange = ops.PulseOperation(
        1.0,
        (_control(backend.model.control.exchange("q1", "q0"), (0.02, 0.02)),),
    )
    program = fq.Program(2)
    program.add(detuning)
    program.add(exchange)

    plan = backend._prepare_program(program).plan

    assert plan[0].target_indices == (0,)
    assert plan[1].target_indices == (0, 1)
    assert plan[1].controls[0].channel == backend.model.control.exchange("q0", "q1")


def test_concurrent_drive_and_offset_exchange_lower_into_one_atomic_block(backend):
    operation = ops.PulseOperation(
        2.0,
        (
            _control(
                backend.model.control.drive("q0"),
                (0.0, 0.04 + 0.01j),
                duration=2.0,
            ),
            _control(
                backend.model.control.exchange("q0", "q1"),
                (0.0, 0.015),
                duration=1.0,
                offset=0.5,
            ),
        ),
    )
    program = fq.Program(2)
    program.add(operation)
    program.add(ops.RX(0.1), 0)

    plan = backend._prepare_program(program).plan

    assert len(plan) == 2
    assert plan[0].controls == operation.controls
    assert plan[0].target_indices == (0, 1)


def test_condition_is_preserved_on_direct_control(backend):
    operation = ops.PulseOperation(
        1.0,
        (_control(backend.model.control.drive("q1"), (0.0, 0.1j)),),
    )
    program = fq.Program(2, 1)
    program.add(operation, condition=(0, 1))

    (block,) = backend._prepare_program(program).plan

    assert block.condition == ((0, 1),)
    assert block.target_indices == (1,)


@pytest.mark.parametrize("kind", ("detuning", "exchange"))
def test_real_only_direct_controls_reject_complex_envelopes(backend, kind):
    channel = (
        backend.model.control.detuning("q0")
        if kind == "detuning"
        else backend.model.control.exchange("q0", "q1")
    )
    program = fq.Program(2)
    program.add(
        ops.PulseOperation(
            1.0,
            (_control(channel, (0.1 + 0.2j, 0.1 + 0.2j)),),
        )
    )

    with pytest.raises(BackendValidationError, match=f"{kind}.*must be real"):
        backend._prepare_program(program)


def test_reversed_exchange_handles_are_duplicate_channels(backend):
    forward = _control(backend.model.control.exchange("q0", "q1"), (0.0, 0.01))
    reverse = _control(backend.model.control.exchange("q1", "q0"), (0.0, 0.01))

    with pytest.raises(ValueError, match="one channel"):
        ops.PulseOperation(1.0, (forward, reverse))


def test_direct_drive_propagator_matches_independent_full_hamiltonian(backend):
    duration = 0.4
    amplitude = 0.07
    operation = ops.PulseOperation(
        duration,
        (
            _control(
                backend.model.control.drive("q0"),
                (amplitude, amplitude),
                duration=duration,
            ),
        ),
    )
    program = fq.Program(2)
    program.add(operation)

    actual = backend.propagator(program)
    target = _TransmonTarget(backend.model)
    adapter = _TransmonQutipAdapter(
        target,
        engine_allocation=_EngineAllocation(target.device_labels, (3, 3)),
    )
    annihilation = Qobj(backend.model.annihilation)
    drift = adapter._drift.get_ideal_qobjevo([3, 3])(0.0)
    drive = amplitude * tensor(qeye(3), annihilation + annihilation.dag())
    expected = (-1j * (drift + drive) * duration).expm().full()

    assert np.allclose(actual, expected, atol=2e-7)


def test_direct_condition_changes_actual_execution(backend):
    operation = ops.PulseOperation(
        0.5,
        (
            _control(
                backend.model.control.drive("q0"),
                (0.8, 0.8),
                duration=0.5,
            ),
        ),
    )
    enabled = fq.Program(2, 1)
    enabled.add(operation, condition=(0, 0))
    disabled = fq.Program(2, 1)
    disabled.add(operation, condition=(0, 1))

    enabled_state = (
        backend.run(enabled, result_config={"counts": False, "final_state": True})
        .result()
        .get_density_matrix()
    )
    disabled_state = (
        backend.run(disabled, result_config={"counts": False, "final_state": True})
        .result()
        .get_density_matrix()
    )

    assert not np.allclose(enabled_state, disabled_state)
    assert np.isclose(disabled_state[0, 0], 1.0)


def test_disjoint_direct_control_after_measurement_retains_fast_path(backend):
    operation = ops.PulseOperation(
        1.0,
        (_control(backend.model.control.drive("q1"), (0.0, 0.1)),),
    )
    program = fq.Program(2, 1)
    program.measure(0, 0)
    program.add(operation)

    plan = backend._prepare_program(program).plan
    is_dynamic, terminal_measurements = PulseEngine._analyze_plan(tuple(plan))

    assert not is_dynamic
    assert len(terminal_measurements) == 1
    assert plan[-1].target_indices == (1,)
