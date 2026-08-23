"""Three-level-atom direct Raman/Rydberg control regressions."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.linalg import expm

import fatqat as fq
from fatqat._pulse_values import PulseControl
from fatqat.emulator.atom_3level import Atom3LevelCalibration, Atom3LevelModel
from fatqat.emulator._core.engine import PulseEngine
from fatqat.errors import BackendValidationError
from fatqat.waveforms import SampledWaveform


def _backend(model, calibration, sites=2):
    return fq.emulator.Atom3LevelEmulator(
        model,
        arrangement=fq.emulator.AtomArrangement.rectangular(1, sites, 2.0),
    )


def _control(channel, value, duration=0.3):
    return PulseControl(channel, SampledWaveform((0.0, duration), (value, value)))


@pytest.mark.parametrize(
    ("factory_name", "raising"),
    (
        ("raman", np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])),
        ("rydberg", np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])),
    ),
)
def test_local_direct_propagator_matches_independent_qutrit_reference(
    atom_3level_model, atom_3level_calibration, factory_name, raising
):
    backend = _backend(atom_3level_model, atom_3level_calibration, sites=1)
    duration = 0.3
    envelope = 0.4 + 0.2j
    channel = getattr(atom_3level_model.control, factory_name)(0)
    operation = fq.ops.PulseOperation(
        duration, (_control(channel, envelope, duration),)
    )
    program = fq.Program(1)
    program.add(operation)

    actual = backend.propagator(program)
    raising = raising.astype(complex)
    hamiltonian = 0.5 * (envelope * raising + envelope.conjugate() * raising.T)
    expected = expm(-1j * hamiltonian * duration)

    assert np.allclose(actual, expected, atol=2e-7)


def test_concurrent_disjoint_raman_and_rydberg_controls_share_one_block(
    atom_3level_model, atom_3level_calibration
):
    backend = _backend(atom_3level_model, atom_3level_calibration)
    operation = fq.ops.PulseOperation(
        0.3,
        (
            _control(atom_3level_model.control.raman(0), 0.2 + 0.1j),
            _control(atom_3level_model.control.rydberg(1), -0.3j),
        ),
    )
    program = fq.Program(2)
    program.add(operation)
    program.add(fq.ops.RX(0.1), 0)

    plan = backend._prepare_program(program).plan

    assert len(plan) == 2
    assert plan[0].controls == operation.controls
    assert plan[0].target_indices == (0, 1)
    assert tuple(binding.engine_indices for binding in plan[0].control_bindings) == (
        (0,),
        (1,),
    )


def test_controls_sharing_a_site_deduplicate_target_and_claim(
    atom_3level_model, atom_3level_calibration
):
    backend = _backend(atom_3level_model, atom_3level_calibration, sites=1)
    operation = fq.ops.PulseOperation(
        0.3,
        (
            _control(atom_3level_model.control.raman(0), 0.2),
            _control(atom_3level_model.control.rydberg(0), 0.3j),
        ),
    )
    program = fq.Program(1)
    program.add(operation)

    plan = backend._prepare_program(program).plan

    assert plan[0].target_indices == (0,)
    assert tuple(binding.engine_indices for binding in plan[0].control_bindings) == (
        (0,),
        (0,),
    )


def test_structural_controls_reuse_across_compatible_arrangements(
    atom_3level_model, atom_3level_calibration
):
    operation = fq.ops.PulseOperation(
        0.3, (_control(atom_3level_model.control.rydberg(1), 0.2j),)
    )
    program = fq.Program(2)
    program.add(operation)

    first = _backend(atom_3level_model, atom_3level_calibration, sites=2)
    second = fq.emulator.Atom3LevelEmulator(
        atom_3level_model,
        arrangement=fq.emulator.AtomArrangement.rectangular(2, 1, 3.0),
    )

    first_plan = first._prepare_program(program).plan
    second_plan = second._prepare_program(program).plan
    assert first_plan[0].controls == second_plan[0].controls == operation.controls
    assert first_plan[0].target_indices == second_plan[0].target_indices == (1,)


def test_direct_rydberg_drives_coexist_with_signed_all_pair_drift(
    atom_3level_model_document, atom_3level_calibration_document
):
    positive_model = Atom3LevelModel.from_document(deepcopy(atom_3level_model_document))
    negative_document = deepcopy(atom_3level_model_document)
    negative_document["parameters"]["c6"] *= -1
    negative_model = Atom3LevelModel.from_document(negative_document)
    positive_calibration = Atom3LevelCalibration(
        deepcopy(atom_3level_calibration_document)
    )
    negative_calibration = Atom3LevelCalibration(
        deepcopy(atom_3level_calibration_document)
    )
    controls = (
        _control(positive_model.control.rydberg(0), 0.2, 0.1),
        _control(positive_model.control.rydberg(1), 0.2, 0.1),
    )
    program = fq.Program(2)
    program.add(fq.ops.PulseOperation(0.1, controls))
    positive = _backend(positive_model, positive_calibration)
    negative = _backend(negative_model, negative_calibration)

    positive_values = positive._target.interactions
    negative_values = negative._target.interactions

    assert (
        positive_values[0].signed_strength_rad_per_us
        == -negative_values[0].signed_strength_rad_per_us
    )
    assert not np.allclose(positive.propagator(program), negative.propagator(program))


def test_condition_changes_actual_direct_atom_execution(
    atom_3level_model, atom_3level_calibration
):
    backend = _backend(atom_3level_model, atom_3level_calibration, sites=1)
    operation = fq.ops.PulseOperation(
        0.4, (_control(atom_3level_model.control.raman(0), 1.0, 0.4),)
    )
    enabled = fq.Program(1, 1)
    enabled.add(operation, condition=(0, 0))
    disabled = fq.Program(1, 1)
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


def test_disjoint_post_measurement_direct_atom_drive_retains_fast_path(
    atom_3level_model, atom_3level_calibration
):
    backend = _backend(atom_3level_model, atom_3level_calibration)
    program = fq.Program(2, 1)
    program.measure(0, 0)
    program.add(
        fq.ops.PulseOperation(0.3, (_control(atom_3level_model.control.raman(1), 0.1),))
    )

    plan = backend._prepare_program(program).plan
    dynamic, terminal = PulseEngine._analyze_plan(tuple(plan))

    assert not dynamic
    assert len(terminal) == 1
    assert plan[-1].target_indices == (1,)


def test_direct_control_rejects_wrong_family_and_absent_site(
    atom_3level_model, atom_3level_calibration, model
):
    backend = _backend(atom_3level_model, atom_3level_calibration)
    for channel, match in (
        (model.control.drive("q0"), "foreign atom control"),
        (atom_3level_model.control.rydberg(2), "unknown atom site"),
    ):
        program = fq.Program(2)
        program.add(fq.ops.PulseOperation(0.3, (_control(channel, 0.1),)))
        with pytest.raises(BackendValidationError, match=match):
            backend._prepare_program(program)
