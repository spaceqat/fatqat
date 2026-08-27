"""Tests for backend-independent direct pulse operations."""

import math
from dataclasses import FrozenInstanceError, dataclass

import pytest

import fatqat as fq
from fatqat._pulse_values import ControlChannel, PulseControl, TIME_EPSILON
from fatqat.operations import Operation, PulseOperation
from fatqat.emulator import SampledWaveform


@dataclass(frozen=True)
class _Channel(ControlChannel):
    label: str


def _control(label="drive", *, duration=1.0, offset=0.0):
    return PulseControl(
        _Channel(label), SampledWaveform((0.0, duration), (0.0, 0.0)), offset
    )


def test_pulse_operation_copies_controls_and_is_immutable():
    controls = [_control()]
    pulse = PulseOperation(1, controls)
    controls.append(_control("detuning"))

    assert pulse.name == "PulseOperation"
    assert pulse.duration == 1.0
    assert pulse.controls == (_control(),)
    assert pulse.num_subsystems == 0
    assert not hasattr(pulse, "channel")
    assert not hasattr(pulse, "components")
    with pytest.raises(FrozenInstanceError):
        pulse.duration = 2


@pytest.mark.parametrize(
    ("duration", "controls", "error"),
    [
        (0, (_control(),), ValueError),
        (-1, (_control(),), ValueError),
        (True, (_control(),), TypeError),
        (math.inf, (_control(),), ValueError),
        (1, (), ValueError),
        (1, (object(),), TypeError),
        (1, (_control(), _control()), ValueError),
        (1, (_control(duration=1, offset=TIME_EPSILON * 2),), ValueError),
    ],
)
def test_pulse_operation_rejects_invalid_structure(duration, controls, error):
    with pytest.raises(error):
        PulseOperation(duration, controls)


def test_pulse_operation_allows_shared_endpoint_tolerance_and_is_hashable():
    control = _control(duration=1.0, offset=TIME_EPSILON)
    pulse = PulseOperation(1.0, (control,))

    assert pulse == PulseOperation(1.0, (control,))
    assert hash(pulse) == hash(PulseOperation(1.0, (control,)))


def test_program_accepts_only_zero_target_pulse_operation():
    program = fq.Program(2)
    pulse = PulseOperation(1, (_control(),))

    program.add(pulse)
    assert program.operations[0].targets == ()
    with pytest.raises(ValueError, match="expects 0"):
        program.add(pulse, 0)


def test_min_targets_supports_fixed_and_variable_arity():
    class Variable(Operation):
        num_subsystems = None

    class ZeroMinimum(Operation):
        num_subsystems = None
        _min_subsystems = 0

    class FixedIgnoresMinimum(Operation):
        num_subsystems = 2
        _min_subsystems = 9

    assert Variable().min_targets == 1
    assert ZeroMinimum().min_targets == 0
    assert FixedIgnoresMinimum().min_targets == 2


@pytest.mark.parametrize("bad", [-1, 1.5, True, None])
def test_invalid_minimum_fails_at_subclass_definition(bad):
    with pytest.raises(ValueError, match="_min_subsystems"):
        type(
            "BadMinimum",
            (Operation,),
            {"num_subsystems": None, "_min_subsystems": bad},
        )
