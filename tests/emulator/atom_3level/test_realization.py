"""Compilation tests for the standard portable Atom3Level pulse map."""

from copy import deepcopy
import inspect

import numpy as np
import pytest

from fatqat import ops
from fatqat.emulator._core.pulse import _invoke_pulse_rule
from fatqat.emulator.atom_3level import (
    Atom3LevelCalibration,
    Atom3LevelModel,
    default_atom_3level_gate_implementation_map,
)
from fatqat.errors import BackendValidationError


def _map(model, calibration):
    return default_atom_3level_gate_implementation_map(
        model=model, calibration=calibration
    )


def _resolve(implementations, operation, operands):
    rule = implementations.implementation_for(operation, device_operands=operands)
    return _invoke_pulse_rule(rule, operation, device_operands=operands)


def _assert_same_definition(first, second):
    assert first.duration == second.duration
    assert first.post_actions == second.post_actions
    assert len(first.controls) == len(second.controls)
    for left, right in zip(first.controls, second.controls, strict=True):
        assert left.channel == right.channel
        assert left.start_offset == right.start_offset
        assert np.array_equal(left.waveform.times, right.waveform.times)
        assert np.array_equal(left.waveform.values, right.waveform.values)


def test_builder_signature_and_runtime_types_are_exact(
    atom_3level_model, atom_3level_calibration
):
    parameters = inspect.signature(
        default_atom_3level_gate_implementation_map
    ).parameters
    assert tuple(parameters) == ("model", "calibration")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in parameters.values()
    )
    with pytest.raises(BackendValidationError, match="Atom3LevelModel"):
        default_atom_3level_gate_implementation_map(
            model=object(), calibration=atom_3level_calibration
        )
    with pytest.raises(BackendValidationError, match="Atom3LevelCalibration"):
        default_atom_3level_gate_implementation_map(
            model=atom_3level_model, calibration=object()
        )


def test_source_c6_and_identity_do_not_redesign_v1_definitions(
    atom_3level_model_document, atom_3level_calibration
):
    first_document = deepcopy(atom_3level_model_document)
    second_document = deepcopy(atom_3level_model_document)
    second_document["model"]["revision"] = "different-source"
    second_document["parameters"]["c6"] *= -3
    first = _map(Atom3LevelModel.from_document(first_document), atom_3level_calibration)
    second = _map(
        Atom3LevelModel.from_document(second_document), atom_3level_calibration
    )
    for operation, operands in (
        (ops.RX(0.7), (0,)),
        (ops.RY(-0.2), (1,)),
        (ops.RZ(0.4), (1,)),
        (ops.CZ, (0, 1)),
    ):
        _assert_same_definition(
            _resolve(first, operation, operands),
            _resolve(second, operation, operands),
        )


def test_rules_use_plain_ordinals_and_return_structural_claim_free_values(
    atom_3level_model, atom_3level_calibration
):
    implementations = _map(atom_3level_model, atom_3level_calibration)
    rx = _resolve(implementations, ops.RX(0.3), (2,))
    assert rx.controls[0].channel == atom_3level_model.control.raman(2)
    assert not hasattr(rx, "resource_claims")
    assert rx.controls[0].channel.operands == (2,)

    rz = _resolve(implementations, ops.RZ(0.4), (1,))
    assert rz.post_actions[0].frame == atom_3level_model.frame(1)
    assert rz.post_actions[0].frame.operands == (1,)

    cz = _resolve(implementations, ops.CZ, (2, 0))
    assert [control.channel for control in cz.controls] == [
        atom_3level_model.control.rydberg(2),
        atom_3level_model.control.rydberg(0),
    ]
    assert tuple(control.channel.operands for control in cz.controls) == ((2,), (0,))


def test_calibration_values_are_compiled_without_runtime_dependencies(
    atom_3level_model, atom_3level_calibration_document
):
    baseline_calibration = Atom3LevelCalibration(atom_3level_calibration_document)
    changed_document = deepcopy(atom_3level_calibration_document)
    changed_document["recipes"]["rx_ry"]["omega_01"] *= 2
    changed_document["recipes"]["cz"]["omega_1r"] *= 1.5
    changed_calibration = Atom3LevelCalibration(changed_document)
    baseline = _map(atom_3level_model, baseline_calibration)
    changed = _map(atom_3level_model, changed_calibration)

    baseline_rx = _resolve(baseline, ops.RX(0.6), (0,))
    changed_rx = _resolve(changed, ops.RX(0.6), (0,))
    assert changed_rx.duration == pytest.approx(baseline_rx.duration / 2)

    baseline_cz = _resolve(baseline, ops.CZ, (0, 1))
    changed_cz = _resolve(changed, ops.CZ, (0, 1))
    assert not np.array_equal(
        baseline_cz.controls[0].waveform.values,
        changed_cz.controls[0].waveform.values,
    )


@pytest.mark.parametrize("operation", (ops.RX(0.2), ops.RY(0.3), ops.RZ(0.4), ops.CZ))
def test_standard_rules_are_unconstrained_and_operand_aware(
    atom_3level_model, atom_3level_calibration, operation
):
    implementations = _map(atom_3level_model, atom_3level_calibration)
    assert implementations.implementation_for(operation) is not None
    arity = 2 if operation is ops.CZ else 1
    operands = tuple(range(arity))
    assert _resolve(implementations, operation, operands) is not None


def test_invalid_ordinals_are_rejected_by_the_compiled_rule(
    atom_3level_model, atom_3level_calibration
):
    implementations = _map(atom_3level_model, atom_3level_calibration)
    for operands in ((-1,), (True,), (0, 0)):
        operation = ops.CZ if len(operands) == 2 else ops.RX(0.2)
        with pytest.raises(BackendValidationError):
            _resolve(implementations, operation, operands)
