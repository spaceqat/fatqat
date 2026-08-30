"""Private qutrit numerical-adapter contracts for a bound atom target."""

from copy import deepcopy

import numpy as np
import pytest
from qutip import basis, ket2dm, mesolve, tensor

from fatqat.emulator import AtomArrangement
from fatqat._backends.steps import MeasurementStep, ResetStep
from fatqat._index_allocation import _EngineAllocation
from fatqat._pulse_values import PulseControl
from fatqat.emulator._core.adapter_common import _BoundDynamics
from fatqat.emulator._core.engine import _ShotContext
from fatqat.emulator._core.outcome import _PulseShotOutcome
from fatqat.emulator._core.pulse import PhaseShift, PulseBlock
from fatqat.emulator._core.target import _PreparedControlBinding
from fatqat.emulator._core.scheduling import _ScheduledPulseRun
from fatqat.emulator.atom_3level import Atom3LevelModel
from fatqat.emulator.atom_3level.qutip_adapter import _Atom3LevelQutipAdapter
from fatqat.emulator.atom_3level.target import _Atom3LevelTarget
from fatqat.errors import BackendValidationError
from fatqat.emulator import SampledWaveform


def _target(model, coordinates):
    coordinates = tuple(coordinates)
    spacing = coordinates[1][0] - coordinates[0][0] if len(coordinates) > 1 else 1.0
    return _Atom3LevelTarget(
        model,
        AtomArrangement.rectangular(1, len(coordinates), spacing),
    )


def _adapter(model, coordinates, **kwargs):
    target = _target(model, coordinates)
    return _Atom3LevelQutipAdapter(
        target,
        engine_allocation=_EngineAllocation(
            target.device_labels, (3,) * len(target.device_labels)
        ),
        **kwargs,
    )


def _constant(channel, rate, duration):
    return PulseControl(channel, SampledWaveform((0.0, duration), (rate, rate)))


def _block(adapter, controls, duration, *, post_actions=()):
    controls = tuple(controls)
    target_bindings = tuple(
        adapter._target.bind_control(control.channel) for control in controls
    )
    bindings = tuple(
        _PreparedControlBinding(
            binding.kind,
            tuple(
                adapter._engine_allocation.engine_index(value)
                for value in binding.device_operands
            ),
        )
        for binding in target_bindings
    )
    claims = tuple(
        dict.fromkeys(claim for binding in target_bindings for claim in binding.claims)
    )
    return PulseBlock(
        duration,
        controls,
        bindings,
        claims,
        post_actions=post_actions,
    )


def _run(adapter, controls, duration):
    block = _block(adapter, controls, duration)
    return _ScheduledPulseRun((block,), (0.0,), 0.0, duration)


def _two_block_run(first, second):
    return _ScheduledPulseRun((first, second), (0.0, 0.0), 0.0, second.duration)


def _frame_shift_block(adapter, angle, *, site=0):
    action = PhaseShift(adapter._target.model.frame(site), angle)
    gate_binding = adapter._target.bind_frame(action.frame)
    return PulseBlock(
        0.0,
        (),
        (),
        gate_binding.claims,
        post_actions=(action,),
        target_indices=(site,),
    )


def _population(unitary, initial_levels, final_levels):
    initial = tensor(*(basis(3, level) for level in reversed(initial_levels)))
    final = tensor(*(basis(3, level) for level in reversed(final_levels)))
    return abs(final.overlap(unitary * initial)) ** 2


def test_local_operators_and_fixed_signed_all_pair_drift(atom_3level_model):
    adapter = _adapter(
        atom_3level_model,
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )
    assert np.array_equal(
        adapter.local_raman_raising.full(),
        np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
    )
    assert np.array_equal(
        adapter.local_rydberg_raising.full(),
        np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
    )
    assert np.array_equal(adapter.local_rydberg_number.full(), np.diag([0, 0, 1]))
    assert adapter._target.interactions[0].signed_strength_rad_per_us > 0
    assert adapter.interaction_drift().shape == (9, 9)


@pytest.mark.parametrize("transition", ["raman", "rydberg"])
def test_bare_pi_area_transfers_the_selected_transition(atom_3level_model, transition):
    adapter = _adapter(atom_3level_model, ((0.0, 0.0, 0.0),))
    omega = 7.0
    duration = np.pi / omega
    channel = (
        adapter._target.model.control.raman(0)
        if transition == "raman"
        else adapter._target.model.control.rydberg(0)
    )
    unitary = adapter.propagator(
        _run(adapter, (_constant(channel, omega, duration),), duration)
    )
    initial, destination = (0, 1) if transition == "raman" else (1, 2)
    assert _population(unitary, (initial,), (destination,)) == pytest.approx(
        1.0, abs=2e-7
    )
    if transition == "rydberg":
        assert _population(unitary, (0,), (0,)) == pytest.approx(1.0, abs=2e-7)


def test_simultaneous_rydberg_drives_couple_one_one_symmetrically(
    atom_3level_model,
):
    adapter = _adapter(
        atom_3level_model,
        ((0.0, 0.0, 0.0), (100.0, 0.0, 0.0)),
    )
    omega = 7.0
    duration = np.pi / (2.0 * omega)
    controls = tuple(
        _constant(adapter._target.model.control.rydberg(site), omega, duration)
        for site in (0, 1)
    )
    unitary = adapter.propagator(_run(adapter, controls, duration))
    first = _population(unitary, (1, 1), (2, 1))
    second = _population(unitary, (1, 1), (1, 2))
    assert first == pytest.approx(second, abs=2e-7)
    assert first > 0.1


def _max_double_rydberg_population(adapter, *, omega, duration):
    controls = tuple(
        _constant(adapter._target.model.control.rydberg(site), omega, duration)
        for site in (0, 1)
    )
    bound = adapter._bind_run(
        _run(adapter, controls, duration),
        enabled=(True,),
        input_time=0.0,
        input_frames={},
    )
    assert isinstance(bound, _BoundDynamics)
    initial = ket2dm(tensor(basis(3, 1), basis(3, 1)))
    rr = tensor(basis(3, 2), basis(3, 2))
    evolution = mesolve(bound.hamiltonian, initial, np.linspace(0.0, duration, 101))
    return max(abs(rr.overlap(state * rr)) for state in evolution.states)


def test_finite_blockade_suppresses_double_rydberg_population(
    atom_3level_model_document,
):
    omega = 10.0
    duration = np.pi / omega
    negligible = deepcopy(atom_3level_model_document)
    negligible["parameters"]["c6"] = 1e-300
    finite = deepcopy(atom_3level_model_document)
    finite["parameters"]["c6"] = 100.0 * omega
    coordinates = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    no_interaction = _adapter(Atom3LevelModel.from_document(negligible), coordinates)
    finite_blockade = _adapter(Atom3LevelModel.from_document(finite), coordinates)

    assert finite_blockade._target.interactions[
        0
    ].signed_strength_rad_per_us / omega == pytest.approx(100.0)
    assert _max_double_rydberg_population(
        finite_blockade, omega=omega, duration=duration
    ) < _max_double_rydberg_population(no_interaction, omega=omega, duration=duration)


def test_state_copy_measurement_reset_and_completed_payload(atom_3level_model):
    adapter = _adapter(atom_3level_model, ((0.0, 0.0, 0.0),))
    initial = adapter.initial_state()
    copied = adapter.copy_state(initial)
    assert copied is not initial
    context = _ShotContext(ket2dm(basis(3, 2)), [0], np.random.default_rng(2))

    adapter.execute_boundary(MeasurementStep((0,), (0,)), context)
    assert context.classical_memory == [1]
    assert np.allclose(context.state.full(), ket2dm(basis(3, 2)).full())
    adapter.execute_boundary(ResetStep((0,)), context)
    assert np.allclose(context.state.full(), ket2dm(basis(3, 0)).full())
    outcome = adapter.finish_shot(context)
    assert isinstance(outcome, _PulseShotOutcome)
    assert outcome.classical_digits == (1,)
    assert outcome.final_state_kind == "density_matrix"
    assert np.allclose(outcome.final_state, ket2dm(basis(3, 0)).full())


def test_measurement_indices_are_canonical_axes(atom_3level_model):
    adapter = _adapter(
        atom_3level_model,
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )
    context = _ShotContext(
        ket2dm(tensor(basis(3, 2), basis(3, 0))),
        [0],
        np.random.default_rng(4),
    )
    adapter.execute_boundary(MeasurementStep((0,), (0,)), context)
    assert context.classical_memory == [0]
    adapter.execute_boundary(ResetStep((1,)), context)
    assert np.allclose(
        context.state.full(),
        ket2dm(tensor(basis(3, 0), basis(3, 0))).full(),
    )


def test_engine_allocation_must_match_the_complete_target(atom_3level_model):
    target = _target(
        atom_3level_model,
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    )
    with pytest.raises(BackendValidationError, match="complete target"):
        _Atom3LevelQutipAdapter(
            target,
            engine_allocation=_EngineAllocation((0,), (3,)),
        )


def test_atom3_adapter_has_no_continuous_trajectory_mode(atom_3level_model):
    with pytest.raises(BackendValidationError, match="execution mode"):
        _adapter(
            atom_3level_model,
            ((0.0, 0.0, 0.0),),
            execution_mode="trajectory",
        )


def test_frame_only_run_preserves_virtual_frame_semantics(atom_3level_model):
    adapter = _adapter(atom_3level_model, ((0.0, 0.0, 0.0),))
    theta = 0.37
    shift = _frame_shift_block(adapter, theta)
    run = _ScheduledPulseRun((shift,), (0.0,), 0.0, 0.0)
    assert np.allclose(
        adapter.propagator(run).full(),
        np.diag((1.0, np.exp(1j * theta), 1.0)),
    )
    assert np.allclose(
        adapter.propagator(run, apply_final_frame=False).full(), np.eye(3)
    )


@pytest.mark.parametrize(
    ("transition", "initial", "destination", "expected_phase"),
    (
        ("raman", 0, 1, lambda theta: np.exp(-1j * theta)),
        ("rydberg", 1, 2, lambda theta: np.exp(1j * theta)),
    ),
)
def test_positive_frame_shift_rotates_later_controls_in_execution_and_propagator(
    atom_3level_model, transition, initial, destination, expected_phase
):
    adapter = _adapter(atom_3level_model, ((0.0, 0.0, 0.0),))
    theta = 0.37
    omega = 7.0
    duration = np.pi / (2.0 * omega)
    channel = (
        adapter._target.model.control.raman(0)
        if transition == "raman"
        else adapter._target.model.control.rydberg(0)
    )
    driven = _block(
        adapter,
        (_constant(channel, omega, duration),),
        duration,
    )
    run = _two_block_run(_frame_shift_block(adapter, theta), driven)
    context = _ShotContext(
        ket2dm(basis(3, initial)),
        [],
        np.random.default_rng(5),
    )
    adapter.evolve(run, context, (True, True))
    execution_amplitude = context.state.full()[destination, initial]
    unitary = adapter.propagator(run, apply_final_frame=False)
    propagator_amplitude = basis(3, destination).dag() * unitary * basis(3, initial)
    expected = -1j * expected_phase(theta)
    assert execution_amplitude == pytest.approx(expected / 2.0, abs=2e-7)
    assert propagator_amplitude == pytest.approx(expected / np.sqrt(2.0), abs=2e-7)


def test_completed_outcome_can_skip_final_state_copy(atom_3level_model):
    adapter = _adapter(
        atom_3level_model,
        ((0.0, 0.0, 0.0),),
        retain_final_state=False,
    )
    context = _ShotContext(adapter.initial_state(), [], np.random.default_rng(1))
    assert adapter.finish_shot(context).final_state is None
