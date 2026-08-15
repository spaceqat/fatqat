"""Pulse-only implementation-map normalization and invocation contracts."""

import pytest

from fatqat import ops
from fatqat._pulse_values import PulseControl
from fatqat.emulator._core.pulse import (
    PulseBlock,
    PulseDefinition,
    PulseImplementationMap,
    _invoke_pulse_rule,
)
from fatqat.errors import (
    BackendValidationError,
    PulseImplementationError,
    UnsupportedOperationError,
)
from fatqat.waveforms import SampledWaveform


def _definition(model, subsystem_id="q0"):
    return PulseDefinition(
        1.0,
        (
            PulseControl(
                model.drive_control(subsystem_id),
                SampledWaveform((0.0, 1.0), (0.0, 0.0)),
            ),
        ),
    )


def test_direct_definition_requires_and_uses_explicit_ordered_operands(model):
    definition = _definition(model)
    implementations = PulseImplementationMap()
    with pytest.raises(ValueError, match="requires explicit device_operands"):
        implementations.add(ops.X, definition)

    implementations.add(ops.X, definition, device_operands=("q0",))
    rule = implementations.implementation_for(ops.X, device_operands=("q0",))
    assert _invoke_pulse_rule(rule, ops.X, device_operands=("q0",)) is definition


@pytest.mark.parametrize(
    "implementation",
    [
        lambda operation: operation,
        lambda operation, **kwargs: (operation, kwargs),
    ],
    ids=("operation-only", "kwargs-only"),
)
def test_operand_unaware_callable_requires_explicit_registration(implementation):
    implementations = PulseImplementationMap()
    with pytest.raises(ValueError, match="requires explicit device_operands"):
        implementations.add(ops.X, implementation)


def test_positional_only_device_operands_is_operand_unaware():
    def positional(operation, device_operands, /):
        return operation, device_operands

    implementations = PulseImplementationMap()
    with pytest.raises(ValueError, match="requires explicit device_operands"):
        implementations.add(ops.X, positional)


def test_uninspectable_callable_is_operand_unaware():
    class Uninspectable:
        __signature__ = object()

        def __call__(self, operation):
            return operation

    implementations = PulseImplementationMap()
    with pytest.raises(ValueError, match="requires explicit device_operands"):
        implementations.add(ops.X, Uninspectable())


def test_operand_aware_function_and_callable_object_receive_exact_tuple(model):
    received = []

    def function(operation, *, device_operands):
        del operation
        received.append(device_operands)
        return _definition(model, device_operands[0])

    class CallableObject:
        def __call__(self, operation, *, device_operands):
            return function(operation, device_operands=device_operands)

    for implementation in (function, CallableObject()):
        implementations = PulseImplementationMap()
        implementations.add(ops.X, implementation)
        rule = implementations.implementation_for(ops.X, device_operands=("q1",))
        result = _invoke_pulse_rule(rule, ops.X, device_operands=("q1",))
        assert result.controls[0].channel == model.drive_control("q1")

    assert received == [("q1",), ("q1",)]


def test_explicit_registration_allows_operation_only_callable(model):
    definition = _definition(model)
    implementations = PulseImplementationMap()
    implementations.add(
        ops.X,
        lambda operation: definition,
        device_operands=("q0",),
    )
    rule = implementations.implementation_for(ops.X, device_operands=("q0",))
    assert _invoke_pulse_rule(rule, ops.X, device_operands=("q0",)) is definition


def test_registration_and_copy_preserve_rule_identity_and_captured_state(model):
    captured = []

    def rule(operation, *, device_operands):
        del operation
        captured.append(device_operands)
        return _definition(model, device_operands[0])

    implementations = PulseImplementationMap()
    implementations.add(ops.X, rule)
    clone = implementations.copy()
    original_rule = implementations.implementation_for(ops.X, device_operands=("q0",))
    clone_rule = clone.implementation_for(ops.X, device_operands=("q1",))
    assert original_rule is clone_rule

    _invoke_pulse_rule(original_rule, ops.X, device_operands=("q0",))
    _invoke_pulse_rule(clone_rule, ops.X, device_operands=("q1",))
    assert captured == [("q0",), ("q1",)]

    clone.remove(ops.X)
    assert implementations.supports(ops.X)
    assert not clone.supports(ops.X)


def test_wrong_callable_arity_fails_at_use(model):
    def wrong(operation, extra, *, device_operands):
        del operation, extra, device_operands
        return _definition(model)

    implementations = PulseImplementationMap()
    implementations.add(ops.X, wrong)
    rule = implementations.implementation_for(ops.X, device_operands=("q0",))
    with pytest.raises(PulseImplementationError, match="XGate") as excinfo:
        _invoke_pulse_rule(rule, ops.X, device_operands=("q0",))
    assert isinstance(excinfo.value.__cause__, TypeError)


@pytest.mark.parametrize(
    ("error", "error_type"),
    [
        (BackendValidationError("bad target"), BackendValidationError),
        (UnsupportedOperationError("unsupported"), UnsupportedOperationError),
    ],
)
def test_deliberate_backend_errors_propagate_unchanged(error, error_type):
    def rejecting(operation, *, device_operands):
        del operation, device_operands
        raise error

    implementations = PulseImplementationMap()
    implementations.add(ops.X, rejecting)
    rule = implementations.implementation_for(ops.X, device_operands=("q0",))
    with pytest.raises(error_type) as excinfo:
        _invoke_pulse_rule(rule, ops.X, device_operands=("q0",))
    assert excinfo.value is error


@pytest.mark.parametrize("bad_result", [None, (1, 2, 3), "not a definition"])
def test_non_definition_return_is_rejected(bad_result):
    def wrong(operation, *, device_operands):
        del operation, device_operands
        return bad_result

    implementations = PulseImplementationMap()
    implementations.add(ops.X, wrong)
    rule = implementations.implementation_for(ops.X, device_operands=("q0",))
    with pytest.raises(PulseImplementationError, match="PulseDefinition"):
        _invoke_pulse_rule(rule, ops.X, device_operands=("q0",))


def test_pulse_block_return_is_rejected():
    block = PulseBlock(0.0, (), (), ())

    def wrong(operation, *, device_operands):
        del operation, device_operands
        return block

    implementations = PulseImplementationMap()
    implementations.add(ops.X, wrong)
    rule = implementations.implementation_for(ops.X, device_operands=("q0",))
    with pytest.raises(PulseImplementationError, match="PulseDefinition"):
        _invoke_pulse_rule(rule, ops.X, device_operands=("q0",))


def test_shared_registry_rejections_keep_family_neutral_wording():
    class VariableGate(ops.Operation):
        name = "VariableGate"
        _num_subsystems = None

    implementations = PulseImplementationMap()
    with pytest.raises(TypeError, match="implementation maps only support") as excinfo:
        implementations.add(VariableGate, "not a rule")
    assert "matrix implementation map" not in str(excinfo.value)
