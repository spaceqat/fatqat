"""Registration mechanics and locked error policy for PulseImplementationMap."""

import json
from pathlib import Path

import pytest

from fatqat import ops
from fatqat.emulator.pulse import (
    PulseBlock,
    PulseDefinition,
    PulseImplementationMap,
    SampledControl,
    _invoke_pulse_rule,
)
from fatqat.emulator.superconducting import load_calibration_spec, load_physics_model
from fatqat.errors import (
    BackendValidationError,
    PulseImplementationError,
    UnsupportedOperationError,
)

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


def _definition(model):
    return PulseDefinition(
        1.0,
        (SampledControl(model.drive_control("q0"), [0.0, 1.0], [0.0, 0.0]),),
        (model.resource("q0"),),
    )


def _dummy_rule(operation, *, targets, model, calibration):
    return _definition(model)


# --- registration mechanics: mirrors ImplementationMap's own coverage -------


def test_add_accepts_operation_instance_and_class_key():
    m = PulseImplementationMap()
    m.add(ops.CZ, _dummy_rule)
    assert m.supports(ops.CZ)
    assert m.supports(type(ops.CZ))

    m2 = PulseImplementationMap()
    m2.add(type(ops.CZ), _dummy_rule)
    assert m2.supports(ops.CZ)


def test_add_rejects_variable_arity_operation():
    class VariableGate(ops.Operation):
        name = "VariableGate"
        _num_subsystems = None

    m = PulseImplementationMap()
    with pytest.raises(TypeError, match="variable arity"):
        m.add(VariableGate, _dummy_rule)


def test_add_rejects_non_callable_rule():
    m = PulseImplementationMap()
    with pytest.raises(TypeError, match="callable"):
        m.add(ops.CZ, "not a rule")


def test_add_resolves_by_device_operands():
    m = PulseImplementationMap()
    m.add(ops.CZ, _dummy_rule, device_operands=(0, 1))

    assert m.supports(ops.CZ)
    assert m.implementation_for(ops.CZ, device_operands=(0, 1)) is not None
    assert m.implementation_for(ops.CZ, device_operands=(1, 0)) is None
    assert m.device_operands_for(ops.CZ) == frozenset({(0, 1)})
    assert m.implementation_for(ops.CZ) is None


def test_add_rejects_wrong_device_operand_arity():
    m = PulseImplementationMap()
    with pytest.raises(ValueError, match="expects 2 device operand"):
        m.add(ops.CZ, _dummy_rule, device_operands=(0,))


def test_add_rejects_unconstrained_after_device_specific_additions():
    m = PulseImplementationMap()
    m.add(ops.CZ, _dummy_rule, device_operands=(0, 1))
    with pytest.raises(ValueError, match="device-specific implementations"):
        m.add(ops.CZ, _dummy_rule)


def test_add_rejects_device_specific_after_unconstrained_addition():
    m = PulseImplementationMap()
    m.add(ops.CZ, _dummy_rule)
    with pytest.raises(ValueError, match="unconstrained rule"):
        m.add(ops.CZ, _dummy_rule, device_operands=(0, 1))


def test_add_replaces_previous_unconstrained_rule_for_same_operation():
    m = PulseImplementationMap()
    m.add(ops.CZ, _dummy_rule)
    first = m.implementation_for(ops.CZ)

    def other_rule(operation, *, targets, model, calibration):
        return _definition(model)

    m.add(ops.CZ, other_rule)
    second = m.implementation_for(ops.CZ)
    assert first is not second


def test_device_operand_order_is_significant():
    m = PulseImplementationMap()
    m.add(ops.CZ, _dummy_rule, device_operands=(0, 1))
    assert m.implementation_for(ops.CZ, device_operands=(0, 1)) is not None
    assert m.implementation_for(ops.CZ, device_operands=(1, 0)) is None


def test_remove_removes_by_instance_or_class():
    m = PulseImplementationMap()
    m.add(ops.CZ, _dummy_rule)
    m.remove(ops.CZ)
    assert m.implementation_for(ops.CZ) is None
    assert not m.supports(ops.CZ)


def test_supported_operations_enumerates_registered_families():
    m = PulseImplementationMap()
    m.add(ops.CZ, _dummy_rule)
    m.add(ops.RX, _dummy_rule)
    assert m.supported_operations() == frozenset({type(ops.CZ), ops.RX})


def test_copy_is_independent_of_original():
    m = PulseImplementationMap()
    m.add(ops.CZ, _dummy_rule, device_operands=(0, 1))

    clone = m.copy()
    clone.add(ops.CZ, _dummy_rule, device_operands=(1, 0))

    assert m.device_operands_for(ops.CZ) == frozenset({(0, 1)})
    assert clone.device_operands_for(ops.CZ) == frozenset({(0, 1), (1, 0)})


# --- locked error policy: _invoke_pulse_rule --------------------------------


def test_invoke_pulse_rule_returns_the_definition_on_success():
    model, calibration = _model_and_calibration()
    m = PulseImplementationMap()
    m.add(ops.CZ, _dummy_rule)
    rule = m.implementation_for(ops.CZ)

    result = _invoke_pulse_rule(
        rule, ops.CZ, targets=(), model=model, calibration=calibration
    )
    assert isinstance(result, PulseDefinition)


def test_invoke_pulse_rule_wraps_unexpected_exceptions():
    model, calibration = _model_and_calibration()

    def failing_rule(operation, *, targets, model, calibration):
        raise ValueError("bad recipe")

    m = PulseImplementationMap()
    m.add(ops.CZ, failing_rule)
    rule = m.implementation_for(ops.CZ)

    with pytest.raises(PulseImplementationError, match="CZGate") as excinfo:
        _invoke_pulse_rule(
            rule, ops.CZ, targets=(), model=model, calibration=calibration
        )
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_invoke_pulse_rule_propagates_backend_validation_error_unwrapped():
    model, calibration = _model_and_calibration()

    def rejecting_rule(operation, *, targets, model, calibration):
        raise BackendValidationError("target order contradicts orientation")

    m = PulseImplementationMap()
    m.add(ops.CZ, rejecting_rule)
    rule = m.implementation_for(ops.CZ)

    with pytest.raises(BackendValidationError, match="orientation"):
        _invoke_pulse_rule(
            rule, ops.CZ, targets=(), model=model, calibration=calibration
        )


def test_invoke_pulse_rule_propagates_unsupported_operation_error_unwrapped():
    model, calibration = _model_and_calibration()

    def unsupported_rule(operation, *, targets, model, calibration):
        raise UnsupportedOperationError("no recipe for this edge")

    m = PulseImplementationMap()
    m.add(ops.CZ, unsupported_rule)
    rule = m.implementation_for(ops.CZ)

    with pytest.raises(UnsupportedOperationError):
        _invoke_pulse_rule(
            rule, ops.CZ, targets=(), model=model, calibration=calibration
        )


@pytest.mark.parametrize(
    "bad_result",
    [None, (1, 2, 3), "not a definition"],
    ids=["none", "tuple", "string"],
)
def test_invoke_pulse_rule_rejects_non_pulse_definition_return(bad_result):
    model, calibration = _model_and_calibration()

    def wrong_type_rule(operation, *, targets, model, calibration):
        return bad_result

    m = PulseImplementationMap()
    m.add(ops.CZ, wrong_type_rule)
    rule = m.implementation_for(ops.CZ)

    with pytest.raises(PulseImplementationError, match="PulseDefinition"):
        _invoke_pulse_rule(
            rule, ops.CZ, targets=(), model=model, calibration=calibration
        )


def test_invoke_pulse_rule_rejects_a_pulse_block_return():
    # A PulseBlock is the occurrence-bound execution value, not the reusable
    # definition a rule must return; returning one is a rule-authoring bug.
    model, calibration = _model_and_calibration()

    def block_returning_rule(operation, *, targets, model, calibration):
        return PulseBlock(
            model,
            1.0,
            (SampledControl(model.drive_control("q0"), [0.0, 1.0], [0.0, 0.0]),),
            (model.resource("q0"),),
        )

    m = PulseImplementationMap()
    m.add(ops.CZ, block_returning_rule)
    rule = m.implementation_for(ops.CZ)

    with pytest.raises(PulseImplementationError, match="PulseDefinition"):
        _invoke_pulse_rule(
            rule, ops.CZ, targets=(), model=model, calibration=calibration
        )
