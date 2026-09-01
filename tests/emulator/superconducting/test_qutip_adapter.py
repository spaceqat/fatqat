"""Private QuTiP contracts for a bound transmon target."""

from dataclasses import replace
from math import pi

import numpy as np
import pytest
from qutip import Qobj, basis, destroy, ket2dm, mesolve, num, qeye, tensor
from scipy.interpolate import CubicSpline

from fatqat._backends.steps import MeasurementStep, ResetStep
from fatqat._index_allocation import _EngineAllocation
from fatqat._pulse_values import PulseControl
from fatqat.emulator._core.adapter_common import _BoundFrames
from fatqat.emulator._core.engine import PulseEngine, _ShotContext
from fatqat.emulator._core.outcome import _PulseShotOutcome
from fatqat.emulator._core.pulse import (
    PhaseShift,
    PhaseSwap,
    PulseBlock,
    _invoke_pulse_rule,
)
from fatqat.emulator._core.target import _PreparedControlBinding
from fatqat.emulator._core.scheduling import _ScheduledPulseRun, schedule_pulse_run
from fatqat.emulator.superconducting.qutip_adapter import _TransmonQutipAdapter
from fatqat.emulator.superconducting.target import _TransmonTarget
from fatqat.emulator.superconducting.realization import (
    default_transmon_gate_implementation_map,
)
from fatqat.errors import BackendValidationError
from fatqat.operations import CZ, RZ, iSwap
from fatqat.emulator import SampledWaveform


def _target(model):
    return _TransmonTarget(model)


def _engine_allocation(target):
    return _EngineAllocation(
        target.device_labels,
        (target.local_dimension,) * len(target.device_labels),
    )


def _adapter(model, **kwargs):
    target = _target(model)
    return _TransmonQutipAdapter(
        target,
        engine_allocation=_engine_allocation(target),
        **kwargs,
    )


def _qutip_tensor(*canonical_factors):
    """Construct a QuTiP value from canonical least-significant-first factors."""
    return tensor(*reversed(canonical_factors))


def _block(target, controls, duration, *, post_actions=(), condition=None):
    controls = tuple(controls)
    target_bindings = tuple(
        target.bind_control(control.channel) for control in controls
    )
    bindings = tuple(
        _PreparedControlBinding(
            binding.kind,
            tuple(
                target.device_labels.index(value) for value in binding.device_operands
            ),
        )
        for binding in target_bindings
    )
    claims = [claim for binding in target_bindings for claim in binding.claims]
    for action in post_actions:
        claims.extend(target.bind_frame(action.frame).claims)
    target.validate_pulse_controls(controls, bindings, duration)
    return PulseBlock(
        duration,
        controls,
        bindings,
        tuple(dict.fromkeys(claims)),
        post_actions=post_actions,
        condition=condition,
    )


def _drive_block(
    target,
    subsystem_id,
    *,
    duration=1.0,
    coefficients=(0.1, 0.1),
    tlist=None,
    condition=None,
    post_actions=(),
):
    if tlist is None:
        tlist = (0.0, duration)
    return _block(
        target,
        (
            PulseControl(
                target.model.control.drive(subsystem_id),
                SampledWaveform(tlist, coefficients),
            ),
        ),
        duration,
        post_actions=post_actions,
        condition=condition,
    )


def _realize(operation, device_operands, *, target, calibration):
    implementation_map = default_transmon_gate_implementation_map(
        model=target.model,
        calibration=calibration,
    )
    rule = implementation_map.implementation_for(
        operation,
        device_operands=device_operands,
    )
    definition = _invoke_pulse_rule(
        rule,
        operation,
        device_operands=device_operands,
    )
    target_bindings = tuple(
        target.bind_control(control.channel) for control in definition.controls
    )
    bindings = tuple(
        _PreparedControlBinding(
            binding.kind,
            tuple(
                target.device_labels.index(value) for value in binding.device_operands
            ),
        )
        for binding in target_bindings
    )
    claims = list(target.bind_gate_operands(device_operands).claims)
    claims.extend(claim for binding in target_bindings for claim in binding.claims)
    for action in definition.post_actions:
        if isinstance(action, PhaseShift):
            frames = (action.frame,)
        elif isinstance(action, PhaseSwap):
            frames = (action.first, action.second)
        else:
            raise AssertionError(f"unknown frame action {action!r}")
        for frame in frames:
            claims.extend(target.bind_frame(frame).claims)
    target.validate_pulse_controls(
        definition.controls,
        target_bindings,
        definition.duration,
    )
    return PulseBlock(
        definition.duration,
        definition.controls,
        bindings,
        tuple(dict.fromkeys(claims)),
        post_actions=definition.post_actions,
    )


def _context(adapter, state=None, *, memory=None):
    return _ShotContext(
        state=adapter.initial_state() if state is None else state,
        classical_memory=[] if memory is None else list(memory),
        rng=np.random.default_rng(1),
    )


def _evolve(adapter, blocks, context=None, *, boundary=0.0):
    context = _context(adapter) if context is None else context
    run = schedule_pulse_run(blocks, boundary_time=boundary)
    adapter.evolve(run, context, (True,) * len(run.blocks))
    context.time = run.end_time
    return context


def test_completed_outcome_can_skip_final_state_copy(model):
    adapter = _adapter(model, retain_final_state=False)
    outcome = adapter.finish_shot(_context(adapter))
    assert isinstance(outcome, _PulseShotOutcome)
    assert outcome.final_state is None
    assert outcome.final_state_kind == "density_matrix"


def test_child_binding_uses_one_cubic_qip_pulse_and_native_endpoints(model):
    adapter = _adapter(model)
    child = PulseControl(
        model.control.drive("q0"),
        SampledWaveform(
            (0.0, 0.3, 0.7, 1.0),
            (1.0 + 2.0j, 3.0 + 4.0j, -2.0 + 1.0j, 5.0 + 6.0j),
        ),
        start_offset=0.5,
    )
    phase = 0.2
    target_binding = adapter._target.bind_control(child.channel)
    binding = _PreparedControlBinding(
        target_binding.kind,
        tuple(
            adapter._engine_allocation.engine_index(value)
            for value in target_binding.device_operands
        ),
    )
    pulse = adapter._bind_child(child, binding, 4.0, {model.frame("q0"): phase})
    expected = 0.5 * np.exp(-1j * phase) * np.asarray(child.waveform.values)

    assert type(pulse).__module__ == "qutip_qip.pulse"
    assert pulse.spline_kind == "cubic"
    assert np.array_equal(pulse.tlist, np.array([4.5, 4.8, 5.2, 5.5]))
    assert np.allclose(pulse.coeff, expected.real)
    assert len(pulse.coherent_noise) == 1
    assert np.allclose(pulse.coherent_noise[0].coeff, expected.imag)

    evolution, collapse = pulse.get_noisy_qobjevo([3, 3])
    assert collapse == []
    assert np.allclose(evolution(4.0).full(), evolution(4.5).full())
    assert np.allclose(evolution(6.0).full(), evolution(5.5).full())
    time = 4.95
    spline = CubicSpline(pulse.tlist, expected)
    annihilation = destroy(len(model.basis_order))
    x_operator = annihilation + annihilation.dag()
    y_operator = -1j * (annihilation - annihilation.dag())
    expected_hamiltonian = spline(time).real * _qutip_tensor(
        x_operator, qeye(3)
    ) + spline(time).imag * _qutip_tensor(y_operator, qeye(3))
    assert np.allclose(evolution(time).full(), expected_hamiltonian.full())


def test_constant_drive_matches_an_independent_full_model_hamiltonian(model):
    adapter = _adapter(model)
    duration = 0.8
    amplitude = 0.07
    context = _evolve(
        adapter,
        (
            _drive_block(
                adapter._target,
                "q0",
                duration=duration,
                coefficients=(amplitude,) * 2,
            ),
        ),
    )

    annihilation = destroy(len(model.basis_order))
    drift = adapter._drift.get_ideal_qobjevo([3, 3])(0.0)
    hamiltonian = drift + 0.5 * amplitude * _qutip_tensor(
        annihilation + annihilation.dag(), qeye(3)
    )
    initial = adapter.initial_state()
    unitary = (-1j * hamiltonian * duration).expm()
    expected = unitary * initial * unitary.dag()
    assert np.allclose(context.state.full(), expected.full(), atol=2e-7)


def test_exchange_keeps_both_qutrit_leakage_paths_and_matches_reference(model):
    adapter = _adapter(model)
    amplitude = 0.12
    duration = 0.4
    exchange = _block(
        adapter._target,
        (
            PulseControl(
                model.control.exchange("q0", "q1"),
                SampledWaveform(
                    (0.0, duration),
                    (amplitude, amplitude),
                ),
            ),
        ),
        duration,
    )
    initial = ket2dm(tensor(basis(3, 1), basis(3, 1)))
    context = _evolve(adapter, (exchange,), _context(adapter, initial))

    annihilation = destroy(len(model.basis_order))
    exchange_operator = tensor(annihilation.dag(), annihilation) + tensor(
        annihilation, annihilation.dag()
    )
    drift = adapter._drift.get_ideal_qobjevo([3, 3])(0.0)
    unitary = (-1j * (drift + amplitude * exchange_operator) * duration).expm()
    expected = unitary * initial * unitary.dag()
    density = context.state.full()
    assert density[6, 6].real > 1e-4
    assert density[2, 2].real > 1e-4
    assert np.allclose(density, expected.full(), atol=2e-7)


def test_realized_iswap_matches_the_public_positive_i_phase_convention(
    model, calibration
):
    adapter = _adapter(model)
    block = _realize(
        iSwap,
        ("q0", "q1"),
        target=adapter._target,
        calibration=calibration,
    )
    ket_00 = _qutip_tensor(basis(3, 0), basis(3, 0))
    ket_01 = _qutip_tensor(basis(3, 0), basis(3, 1))
    ket_10 = _qutip_tensor(basis(3, 1), basis(3, 0))
    initial = ket2dm((ket_00 + ket_01 + 0.3 * ket_10).unit())
    actual = _evolve(adapter, (block,), _context(adapter, initial)).state
    expected = ket2dm((ket_00 + 1j * ket_10 + 0.3j * ket_01).unit())
    assert np.allclose(actual.full(), expected.full(), atol=2e-7)


def test_drift_and_detuning_match_independent_qutrit_phase_facts(model, model_document):
    adapter = _adapter(model)
    duration = 0.17
    ket_00 = _qutip_tensor(basis(3, 0), basis(3, 0))
    ket_20 = _qutip_tensor(basis(3, 2), basis(3, 0))
    initial = ket2dm((ket_00 + ket_20).unit())
    idle = _drive_block(
        adapter._target,
        "q0",
        duration=duration,
        coefficients=(0.0, 0.0),
    )
    actual = _evolve(adapter, (idle,), _context(adapter, initial)).state
    anharmonicity = model_document["parameters"]["subsystems"]["q0"]["anharmonicity"]
    phase = np.exp(-1j * 2 * pi * anharmonicity * duration)
    expected = ket2dm((ket_00 + phase * ket_20).unit())
    assert np.allclose(actual.full(), expected.full(), atol=2e-7)

    detuning = 0.09
    detuned = _block(
        adapter._target,
        (
            PulseControl(
                model.control.detuning("q0"),
                SampledWaveform((0.0, duration), (detuning, detuning)),
            ),
        ),
        duration,
    )
    actual = _evolve(adapter, (detuned,), _context(adapter, initial)).state
    phase = np.exp(-1j * (2 * pi * anharmonicity + 2 * detuning) * duration)
    expected = ket2dm((ket_00 + phase * ket_20).unit())
    assert np.allclose(actual.full(), expected.full(), atol=2e-7)


def test_control_on_q1_keeps_complete_model_canonical_state_order(model):
    adapter = _adapter(model)
    outcomes = PulseEngine(adapter).run(
        (_drive_block(adapter._target, "q1"),),
        shots=1,
        n_clbits=0,
        rng=np.random.default_rng(3),
    )
    physical = Qobj(outcomes[0].final_state, dims=[[3, 3], [3, 3]])
    assert np.allclose(physical.ptrace(1).full(), ket2dm(basis(3, 0)).full())
    assert physical.ptrace(0).diag()[0].real < 0.999


def test_drift_covers_leading_internal_and_trailing_idle_intervals(model):
    adapter = _adapter(model)
    first = replace(
        _drive_block(
            adapter._target,
            "q0",
            duration=1.0,
            coefficients=(0.0, 0.0),
        ),
        start_time=1.0,
    )
    second = replace(
        _drive_block(
            adapter._target,
            "q1",
            duration=1.0,
            coefficients=(0.0, 0.0),
        ),
        start_time=3.0,
    )
    ket = _qutip_tensor((basis(3, 0) + basis(3, 2)).unit(), basis(3, 0))
    initial = ket2dm(ket)
    context = _context(adapter, initial)
    run = schedule_pulse_run((first, second), boundary_time=0.0)
    adapter.evolve(run, context, (True, True))

    drift = adapter._drift.get_ideal_qobjevo([3, 3])(0.0)
    unitary = (-1j * drift * 4.0).expm()
    assert np.allclose(
        context.state.full(),
        (unitary * initial * unitary.dag()).full(),
        atol=2e-7,
    )


def test_local_frame_fixes_nominal_cz_crossing_but_calibration_remains_data(
    model, calibration, model_document
):
    adapter = _adapter(model)
    recipe = calibration._cz_recipe("q0", "q1")
    assert recipe is not None
    detuning_ghz = recipe.park_detuning_ghz
    ramp_duration_ns = recipe.ramp_duration_ns
    assert (
        detuning_ghz
        == -model_document["parameters"]["subsystems"]["q0"]["anharmonicity"]
    )

    drift = adapter._drift.get_ideal_qobjevo([3, 3])(0.0)
    parked = drift + 2 * pi * detuning_ghz * adapter._number[0]
    state_20 = _qutip_tensor(basis(3, 2), basis(3, 0))
    state_11 = _qutip_tensor(basis(3, 1), basis(3, 1))
    energy_20 = complex(state_20.dag() * parked * state_20).real
    energy_11 = complex(state_11.dag() * parked * state_11).real
    assert np.isclose(energy_20, energy_11)

    cz = _realize(
        CZ,
        ("q0", "q1"),
        target=adapter._target,
        calibration=calibration,
    )
    assert cz.controls[1].start_offset == ramp_duration_ns


def test_realized_cz_matches_an_independent_synchronized_hamiltonian(
    model, calibration, model_document
):
    adapter = _adapter(model)
    block = _realize(
        CZ,
        ("q0", "q1"),
        target=adapter._target,
        calibration=calibration,
    )
    detuning, exchange = block.controls
    detuning_spline = CubicSpline(
        detuning.waveform.times,
        np.asarray(detuning.waveform.values).real,
    )
    exchange_spline = CubicSpline(
        exchange.start_offset + np.asarray(exchange.waveform.times),
        np.asarray(exchange.waveform.values).real,
    )

    def parked_exchange(time, _args=None):
        if time < exchange.start_offset or time > (
            exchange.start_offset + exchange.waveform.duration
        ):
            return 0.0
        return float(exchange_spline(time))

    assert parked_exchange(exchange.start_offset / 2) == 0.0
    assert parked_exchange(block.duration - exchange.start_offset / 2) == 0.0
    dimension = len(model.basis_order)
    number = num(dimension)
    annihilation = destroy(dimension)
    identity = qeye(3)
    drift = sum(
        2
        * pi
        * model_document["parameters"]["subsystems"][subsystem_id]["anharmonicity"]
        * _qutip_tensor(
            number * (number - identity) / 2 if ordinal == 0 else identity,
            number * (number - identity) / 2 if ordinal == 1 else identity,
        )
        for ordinal, subsystem_id in enumerate(model.subsystem_ids)
    )
    exchange_operator = _qutip_tensor(annihilation.dag(), annihilation) + _qutip_tensor(
        annihilation, annihilation.dag()
    )
    initial = ket2dm(_qutip_tensor(basis(3, 1), basis(3, 1)))
    expected = mesolve(
        [
            drift,
            [_qutip_tensor(number, identity), detuning_spline],
            [exchange_operator, parked_exchange],
        ],
        initial,
        [0.0, block.duration],
        options={"atol": 1e-11, "rtol": 1e-9, "nsteps": 100000},
    ).states[-1]
    actual = _evolve(adapter, (block,), _context(adapter, initial)).state
    assert np.allclose(actual.full(), expected.full(), atol=2e-7)


class _NoOpBoundaryAdapter(_TransmonQutipAdapter):
    def execute_boundary(self, step, context):
        del step, context


class _RecordingAdapter(_NoOpBoundaryAdapter):
    def __init__(self, target):
        super().__init__(
            target,
            engine_allocation=_engine_allocation(target),
        )
        self.bound_phases = []

    def _bind_child(self, child, binding, block_start_time, frames):
        if binding.kind == "drive":
            engine_index = binding.engine_indices[0]
            frame = self._target.model.frame(
                self._engine_allocation.device_operands[engine_index]
            )
            self.bound_phases.append((engine_index, frames.get(frame, 0.0)))
        return super()._bind_child(child, binding, block_start_time, frames)


def test_frame_ledger_survives_boundary_and_respects_post_action_time(
    model, calibration
):
    target = _target(model)
    adapter = _RecordingAdapter(target)
    frame = model.frame("q0")
    rz = _realize(
        RZ(0.2),
        ("q0",),
        target=target,
        calibration=calibration,
    )
    overlapping = _drive_block(
        target,
        "q1",
        duration=2.0,
        coefficients=(0.0, 0.0),
        post_actions=(PhaseShift(frame, 0.3),),
    )
    q0_first = _drive_block(target, "q0", coefficients=(0.0, 0.0))
    q0_second = _drive_block(target, "q0", coefficients=(0.0, 0.0))
    q0_after = _drive_block(target, "q0", coefficients=(0.0, 0.0))
    context = _context(adapter, memory=(0,))
    first_run = schedule_pulse_run((rz,), boundary_time=0.0)
    adapter.evolve(first_run, context, (True,))
    adapter.execute_boundary(
        MeasurementStep((0,), (0,), reported_digit_maps=((0, 1, 1),)),
        context,
    )
    second_run = _ScheduledPulseRun(
        (overlapping, q0_first, q0_second, q0_after),
        (0.0, 0.0, 1.0, 2.0),
        0.0,
        3.0,
    )
    adapter.evolve(second_run, context, (True, True, True, True))
    q0_phases = [phase for ordinal, phase in adapter.bound_phases if ordinal == 0]
    assert np.allclose(q0_phases, [0.2, 0.2, 0.5])


def test_drive_and_exchange_bindings_use_carried_target_facts(model):
    adapter = _adapter(model)
    target = adapter._target
    drive = target.bind_control(model.control.drive("q0"))
    exchange = target.bind_control(model.control.exchange("q0", "q1"))
    assert drive.kind == "drive"
    assert drive.device_operands == ("q0",)
    assert exchange.kind == "exchange"
    assert exchange.allows_additional_claims
    assert exchange.device_operands == ("q0", "q1")


def test_frame_only_run_returns_frames_without_constructing_dynamics(model):
    adapter = _adapter(model)
    target = adapter._target
    action = PhaseShift(model.frame("q0"), 0.4)
    frame_binding = target.bind_frame(action.frame)
    block = PulseBlock(
        0.0,
        (),
        (),
        frame_binding.claims,
        post_actions=(action,),
    )
    run = schedule_pulse_run((block,), boundary_time=0.0)
    bound = adapter._bind_run(
        run,
        enabled=(True,),
        input_time=0.0,
        input_frames={},
    )
    assert isinstance(bound, _BoundFrames)
    assert bound.output_frames[model.frame("q0")] == pytest.approx(0.4)
    expected = _qutip_tensor(
        Qobj(np.diag(np.exp(1j * 0.4 * np.arange(3)))), Qobj(np.eye(3))
    )
    assert np.allclose(adapter.propagator(run).full(), expected.full())
    assert np.allclose(
        adapter.propagator(run, apply_final_frame=False).full(), np.eye(9)
    )


def test_disabled_block_suppresses_control_and_post_frame_but_advances_time(model):
    adapter = _adapter(model)
    target = adapter._target
    block = _drive_block(
        target,
        "q0",
        post_actions=(PhaseShift(model.frame("q0"), 0.3),),
    )
    run = schedule_pulse_run((block,), boundary_time=0.0)
    context = _context(adapter)
    adapter.evolve(run, context, (False,))
    assert context.frame_angles == {}
    assert np.allclose(context.state.full(), adapter.initial_state().full())


def test_measurement_and_reset_indices_are_canonical_physical_axes(model):
    adapter = _adapter(model)
    state = ket2dm(_qutip_tensor(basis(3, 0), basis(3, 2)))
    context = _context(adapter, state, memory=(0,))
    step = MeasurementStep((0,), (0,), reported_digit_maps=((0, 1, 1),))
    adapter.execute_boundary(step, context)
    assert context.classical_memory == [0]
    adapter.execute_boundary(ResetStep((1,)), context)
    expected = ket2dm(_qutip_tensor(basis(3, 0), basis(3, 0)))
    assert np.allclose(context.state.full(), expected.full())


def test_engine_allocation_must_cover_the_complete_model(model):
    target = _target(model)
    with pytest.raises(BackendValidationError, match="complete model"):
        _TransmonQutipAdapter(
            target,
            engine_allocation=_EngineAllocation(("q1",), (3,)),
        )


def test_measurement_uses_supplied_confusion_and_binary_digit_map(model):
    adapter = _adapter(model)
    state = ket2dm(_qutip_tensor(basis(3, 2), basis(3, 0)))
    context = _context(adapter, state, memory=(0,))
    adapter.execute_boundary(
        MeasurementStep(
            (0,),
            (0,),
            confusions=(np.array([[0.0, 1.0], [1.0, 0.0]]),),
            reported_digit_maps=((0, 1, 1),),
        ),
        context,
    )
    assert context.classical_memory == [0]


def test_initial_copy_and_propagator_shapes_cover_full_qutrit_model(model):
    adapter = _adapter(model)
    initial = adapter.initial_state()
    copied = adapter.copy_state(initial)
    assert copied is not initial
    assert initial.shape == (9, 9)
    block = _drive_block(adapter._target, "q0", duration=0.2)
    unitary = adapter.propagator(schedule_pulse_run((block,), boundary_time=0.0))
    assert unitary.shape == (9, 9)


def test_coherent_statevector_execution_returns_a_flat_full_qutrit_ket(model):
    adapter = _adapter(model, execution_mode="statevector")
    initial = adapter.initial_state()
    context = _evolve(
        adapter,
        (_drive_block(adapter._target, "q0", duration=0.2, coefficients=(2.0, 2.0)),),
    )
    outcome = adapter.finish_shot(context)

    assert initial.shape == (9, 1)
    assert outcome.final_state_kind == "statevector"
    assert outcome.final_state.shape == (9,)
    assert np.linalg.norm(outcome.final_state) == pytest.approx(1.0)
    assert abs(outcome.final_state[0]) < 0.999


def test_sampled_ket_reset_matches_the_exact_qutrit_channel_ensemble(model):
    entangled = (
        _qutip_tensor(basis(3, 1), basis(3, 0))
        + _qutip_tensor(basis(3, 2), basis(3, 2))
    ).unit()
    exact_adapter = _adapter(model, execution_mode="density_matrix")
    exact_context = _context(exact_adapter, ket2dm(entangled))
    exact_adapter.execute_boundary(ResetStep((0,)), exact_context)

    trajectory_adapter = _adapter(model, execution_mode="statevector")
    projectors = []
    for seed in range(400):
        context = _ShotContext(entangled.copy(), [], np.random.default_rng(seed))
        trajectory_adapter.execute_boundary(ResetStep((0,)), context)
        vector = np.asarray(context.state.full()).reshape(-1)
        projectors.append(np.outer(vector, vector.conj()))

    sampled = np.mean(projectors, axis=0)
    assert sampled == pytest.approx(exact_context.state.full(), abs=0.08)
