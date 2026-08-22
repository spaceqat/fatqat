"""Compilation tests for the standard portable transmon pulse map."""

from copy import deepcopy
import inspect

import numpy as np
import pytest

from fatqat import ops
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
    redesigned = _resolve(
        ops.RX(0.7),
        ("q0",),
        model=TransmonModel.from_document(changed_model),
        calibration=calibration,
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


def test_two_body_rules_cover_both_orders_and_select_ordered_cz_override(
    model, calibration_document
):
    document = deepcopy(calibration_document)
    override = document["recipes"]["cz"]["overrides"][0]["recipe"]
    override.update(
        {
            "detuning_operand": 1,
            "duration": 64.0,
            "ramp_duration": 4.0,
            "detuning": 0.25,
        }
    )
    calibration = TransmonCalibration(document)
    implementations = _map(model, calibration)
    expected_keys = frozenset({("q0", "q1"), ("q1", "q0")})
    assert implementations.device_operands_for(ops.CZ) == expected_keys
    assert implementations.device_operands_for(ops.iSwap) == expected_keys

    forward = _resolve(ops.CZ, ("q0", "q1"), model=model, calibration=calibration)
    reverse = _resolve(ops.CZ, ("q1", "q0"), model=model, calibration=calibration)
    assert forward.duration == 64.0
    assert forward.controls[0].channel == model.control.detuning("q1")
    assert reverse.duration == 60.0
    assert reverse.controls[0].channel == model.control.detuning("q1")

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


def test_unused_override_is_accepted_but_source_domain_does_not_expand(
    model, calibration_document
):
    document = deepcopy(calibration_document)
    document["recipes"]["cz"]["overrides"].append(
        {
            "device_operands": ["q8", "q9"],
            "recipe": deepcopy(document["recipes"]["cz"]["default"]),
        }
    )
    implementations = _map(model, TransmonCalibration(document))
    assert (
        implementations.implementation_for(ops.CZ, device_operands=("q8", "q9")) is None
    )
    with pytest.raises(BackendValidationError, match="no subsystem"):
        _invoke_pulse_rule(
            implementations.implementation_for(ops.RX(0.2), device_operands=("q8",)),
            ops.RX(0.2),
            device_operands=("q8",),
        )


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
