"""Compilation tests for the standard portable transmon pulse map."""

from copy import deepcopy
import inspect

import numpy as np
import pytest

import fatqat.operations as ops
from fatqat.emulator._core.pulse import _invoke_pulse_rule
from fatqat.emulator.superconducting import (
    TransmonCalibration,
    TransmonModel,
    default_transmon_gate_implementation_map,
)
from fatqat.errors import BackendValidationError


def _map(model, calibration):
    return default_transmon_gate_implementation_map(
        model=model, calibration=calibration
    )


def _resolve(operation, operands, *, model, calibration):
    rule = _map(model, calibration).implementation_for(
        operation, device_operands=operands
    )
    return _invoke_pulse_rule(rule, operation, device_operands=operands)


def test_builder_signature_and_runtime_types_are_exact(model, calibration):
    parameters = inspect.signature(default_transmon_gate_implementation_map).parameters
    assert tuple(parameters) == ("model", "calibration")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
        for parameter in parameters.values()
    )
    with pytest.raises(BackendValidationError, match="TransmonModel"):
        default_transmon_gate_implementation_map(
            model=object(), calibration=calibration
        )
    with pytest.raises(BackendValidationError, match="TransmonCalibration"):
        default_transmon_gate_implementation_map(model=model, calibration=object())


def test_drag_uses_compiled_source_anharmonicity_and_calibration(
    model_document, calibration_document
):
    source = TransmonModel.from_document(model_document)
    calibration = TransmonCalibration(calibration_document)
    baseline = _resolve(ops.RX(0.7), ("q0",), model=source, calibration=calibration)

    changed_model = deepcopy(model_document)
    changed_model["parameters"]["subsystems"]["q0"]["anharmonicity"] = -0.4
    changed_branch = deepcopy(calibration_document)
    changed_branch["recipes"]["cz"]["edges"][0]["recipe"]["park_detuning_ghz"] = 0.4
    redesigned = _resolve(
        ops.RX(0.7),
        ("q0",),
        model=TransmonModel.from_document(changed_model),
        calibration=TransmonCalibration(changed_branch),
    )
    assert not np.allclose(
        baseline.controls[0].waveform.values,
        redesigned.controls[0].waveform.values,
    )

    changed_calibration = deepcopy(calibration_document)
    changed_calibration["recipes"]["rx_ry"]["duration"] = 24.0
    changed_calibration["recipes"]["rx_ry"]["drag_coefficient"] = 0.5
    recalibrated = _resolve(
        ops.RX(0.7),
        ("q0",),
        model=source,
        calibration=TransmonCalibration(changed_calibration),
    )
    assert recalibrated.duration == 24.0
    assert not np.allclose(
        baseline.controls[0].waveform.values,
        recalibrated.controls[0].waveform.values,
    )


def test_single_qubit_rules_are_unconstrained_operand_aware(model, calibration):
    implementations = _map(model, calibration)
    for operation in (ops.RX(0.2), ops.RY(0.3), ops.RZ(0.4)):
        assert implementations.implementation_for(operation) is not None
        first = _invoke_pulse_rule(
            implementations.implementation_for(operation, device_operands=("q0",)),
            operation,
            device_operands=("q0",),
        )
        second = _invoke_pulse_rule(
            implementations.implementation_for(operation, device_operands=("q1",)),
            operation,
            device_operands=("q1",),
        )
        assert first != second


def test_two_body_rules_cover_both_orders_and_share_one_physical_cz(model, calibration):
    implementations = _map(model, calibration)
    expected_keys = frozenset({("q0", "q1"), ("q1", "q0")})
    assert implementations.device_operands_for(ops.CZ) == expected_keys
    assert implementations.device_operands_for(ops.iSwap) == expected_keys

    forward_rule = implementations.implementation_for(
        ops.CZ, device_operands=("q0", "q1")
    )
    reverse_rule = implementations.implementation_for(
        ops.CZ, device_operands=("q1", "q0")
    )
    forward = _invoke_pulse_rule(forward_rule, ops.CZ, device_operands=("q0", "q1"))
    reverse = _invoke_pulse_rule(reverse_rule, ops.CZ, device_operands=("q1", "q0"))
    assert forward.duration == reverse.duration == 60.0
    assert forward.controls[0].channel == model.control.detuning("q0")
    assert reverse.controls[0].channel == model.control.detuning("q0")
    assert tuple(control.channel for control in forward.controls) == tuple(
        control.channel for control in reverse.controls
    )
    for first, second in zip(forward.controls, reverse.controls):
        assert np.array_equal(first.waveform.times, second.waveform.times)
        assert np.array_equal(first.waveform.values, second.waveform.values)
    assert forward.post_actions == reverse.post_actions

    iswap_forward = _resolve(
        ops.iSwap, ("q0", "q1"), model=model, calibration=calibration
    )
    iswap_reverse = _resolve(
        ops.iSwap, ("q1", "q0"), model=model, calibration=calibration
    )
    assert np.array_equal(
        iswap_forward.controls[0].waveform.values,
        iswap_reverse.controls[0].waveform.values,
    )


def test_unused_canonical_edge_is_accepted_but_source_domain_does_not_expand(
    model, calibration_document
):
    document = deepcopy(calibration_document)
    extra = deepcopy(document["recipes"]["cz"]["edges"][0])
    extra["canonical_edge"] = ["q8", "q9"]
    extra["recipe"]["detuned_subsystem"] = "q8"
    document["recipes"]["cz"]["edges"].append(extra)
    calibration = TransmonCalibration(document)
    implementations = _map(model, calibration)
    assert calibration._cz_recipe("q9", "q8") is not None
    assert (
        implementations.implementation_for(ops.CZ, device_operands=("q8", "q9")) is None
    )
    with pytest.raises(BackendValidationError, match="no subsystem"):
        _invoke_pulse_rule(
            implementations.implementation_for(ops.RX(0.2), device_operands=("q8",)),
            ops.RX(0.2),
            device_operands=("q8",),
        )


def test_builder_requires_every_model_edge(model, calibration_document):
    document = deepcopy(calibration_document)
    document["recipes"]["cz"]["edges"].clear()
    with pytest.raises(BackendValidationError, match="no CZ recipe.*q0.*q1"):
        _map(model, TransmonCalibration(document))


def test_builder_checks_the_selected_endpoint_branch_with_inclusive_tolerance(
    model, calibration_document
):
    document = deepcopy(calibration_document)
    recipe = document["recipes"]["cz"]["edges"][0]["recipe"]
    recipe["park_detuning_ghz"] = 0.23
    branch_error = abs(0.23 - 0.22)
    recipe["branch_tolerance_ghz"] = branch_error
    _map(model, TransmonCalibration(document))

    recipe["branch_tolerance_ghz"] = np.nextafter(branch_error, 0.0)
    with pytest.raises(BackendValidationError, match="expected.*tolerance"):
        _map(model, TransmonCalibration(document))


def test_standard_map_modes_require_remove_before_switching(model, calibration):
    implementations = _map(model, calibration)
    with pytest.raises(ValueError):
        implementations.add(
            ops.CZ,
            lambda operation, *, device_operands: None,
        )
    implementations.remove(ops.CZ)
    implementations.add(
        ops.CZ,
        lambda operation, *, device_operands: None,
    )

    with pytest.raises(ValueError):
        implementations.add(
            ops.RX,
            lambda operation: None,
            device_operands=("q0",),
        )
    implementations.remove(ops.RX)
    implementations.add(
        ops.RX,
        lambda operation: None,
        device_operands=("q0",),
    )
