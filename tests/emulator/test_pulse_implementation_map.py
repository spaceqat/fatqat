"""Pulse-specific behaviour of PulseImplementationMap and its error policy.

The instance/class key normalization, fixed-arity checking, device-operand
arity, mutually exclusive registration modes, removal, and copy independence
all live in the shared `implementation._operation_registry` mechanics and are
covered exhaustively against the matrix family in `tests/test_implementation.py`.
Re-testing them here would only re-exercise the same code through a second
wrapper, so this module keeps a small delegation set proving `PulseImplementationMap`
composes that registry correctly (wrapping, device-specific lookup, copying)
and then focuses on what is genuinely pulse-only: the locked
`_invoke_pulse_rule` error policy.
"""

import pytest

from fatqat import ops
from fatqat.emulator.pulse import (
    PulseBlock,
    PulseDefinition,
    PulseImplementationMap,
    SampledControl,
    _invoke_pulse_rule,
)
from fatqat.errors import (
    BackendValidationError,
    PulseImplementationError,
    UnsupportedOperationError,
)


def _definition(model):
    return PulseDefinition(
        1.0,
        (SampledControl(model.drive_control("q0"), [0.0, 1.0], [0.0, 0.0]),),
        (model.resource("q0"),),
    )


def _dummy_rule(operation, *, targets, model, calibration):
    return _definition(model)


# --- delegation to the shared registry mechanics ----------------------------


def test_registration_wrapping_lookup_and_copy_delegate_to_the_shared_registry():
    """One pass over the seams PulseImplementationMap actually owns.

    Not a re-test of registry mechanics: it checks that the pulse map hands
    the registry a rule factory (so a bad rule is rejected), keys by operation
    family, resolves ordered device operands, and copies independently.
    """
    m = PulseImplementationMap()
    with pytest.raises(TypeError, match="callable"):
        m.add(ops.CZ, "not a rule")

    m.add(ops.CZ, _dummy_rule, device_operands=("q0", "q1"))
    assert m.supports(ops.CZ)
    assert m.supported_operations() == frozenset({type(ops.CZ)})
    assert m.implementation_for(ops.CZ, device_operands=("q0", "q1")) is not None
    assert m.implementation_for(ops.CZ, device_operands=("q1", "q0")) is None
    assert m.implementation_for(ops.CZ) is None
    assert m.device_operands_for(ops.CZ) == frozenset({("q0", "q1")})

    clone = m.copy()
    clone.add(ops.CZ, _dummy_rule, device_operands=("q1", "q0"))
    assert m.device_operands_for(ops.CZ) == frozenset({("q0", "q1")})
    assert clone.device_operands_for(ops.CZ) == frozenset({("q0", "q1"), ("q1", "q0")})

    m.remove(ops.CZ)
    assert not m.supports(ops.CZ)


def test_add_uses_family_neutral_wording_for_a_shared_registry_rejection():
    class VariableGate(ops.Operation):
        name = "VariableGate"
        _num_subsystems = None

    m = PulseImplementationMap()
    with pytest.raises(TypeError, match="implementation maps only support") as excinfo:
        m.add(VariableGate, "not a rule")
    assert "matrix implementation map" not in str(excinfo.value)


# --- locked error policy: _invoke_pulse_rule --------------------------------


def test_invoke_pulse_rule_returns_the_definition_on_success(model, calibration):
    m = PulseImplementationMap()
    m.add(ops.CZ, _dummy_rule)
    rule = m.implementation_for(ops.CZ)

    result = _invoke_pulse_rule(
        rule, ops.CZ, targets=(), model=model, calibration=calibration
    )
    assert isinstance(result, PulseDefinition)


def test_invoke_pulse_rule_wraps_unexpected_exceptions(model, calibration):
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


def test_invoke_pulse_rule_propagates_backend_validation_error_unwrapped(
    model, calibration
):
    def rejecting_rule(operation, *, targets, model, calibration):
        raise BackendValidationError("target order contradicts orientation")

    m = PulseImplementationMap()
    m.add(ops.CZ, rejecting_rule)
    rule = m.implementation_for(ops.CZ)

    with pytest.raises(BackendValidationError, match="orientation"):
        _invoke_pulse_rule(
            rule, ops.CZ, targets=(), model=model, calibration=calibration
        )


def test_invoke_pulse_rule_propagates_unsupported_operation_error_unwrapped(
    model, calibration
):
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
def test_invoke_pulse_rule_rejects_non_pulse_definition_return(
    bad_result, model, calibration
):
    def wrong_type_rule(operation, *, targets, model, calibration):
        return bad_result

    m = PulseImplementationMap()
    m.add(ops.CZ, wrong_type_rule)
    rule = m.implementation_for(ops.CZ)

    with pytest.raises(PulseImplementationError, match="PulseDefinition"):
        _invoke_pulse_rule(
            rule, ops.CZ, targets=(), model=model, calibration=calibration
        )


def test_invoke_pulse_rule_rejects_a_pulse_block_return(model, calibration):
    # A PulseBlock is the occurrence-bound execution value, not the reusable
    # definition a rule must return; returning one is a rule-authoring bug.

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
