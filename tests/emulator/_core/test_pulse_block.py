"""Definition validation and the single private target-binding transition."""

import pytest

from fatqat._pulse_values import PulseControl
from fatqat.emulator._core.target import _PreparedControlBinding, _TargetClaim
from fatqat.emulator._core.pulse import (
    PulseBlock,
    PulseDefinition,
)
from fatqat.errors import BackendValidationError
from fatqat.waveforms import SampledWaveform

_OWNER = object()


def _claim(ordinal=0):
    return _TargetClaim(_OWNER, "subsystem", ordinal)


def _control(model, subsystem_id="q0", *, offset=0.0):
    return PulseControl(
        model.drive_control(subsystem_id),
        SampledWaveform((0.0, 1.0), (0.0, 0.0)),
        offset,
    )


@pytest.mark.parametrize(
    ("duration", "controls", "match"),
    [
        (0.0, "one", "zero-duration"),
        (1.0, "none", "requires physical controls"),
        (1.0, "duplicate", "implicitly sum"),
        (1.0, "overrun", "extends"),
    ],
)
def test_definition_owns_model_independent_shape(model, duration, controls, match):
    values = {
        "one": (_control(model),),
        "none": (),
        "duplicate": (_control(model), _control(model)),
        "overrun": (_control(model, offset=0.5),),
    }[controls]
    with pytest.raises(BackendValidationError, match=match):
        PulseDefinition(duration, values)


def test_definition_has_no_target_owned_resource_claims(model):
    definition = PulseDefinition(1.0, (_control(model),))
    assert not hasattr(definition, "resource_claims")
    with pytest.raises(AttributeError):
        definition.duration = 2.0


def test_block_is_an_already_bound_immutable_occurrence(model):
    control = _control(model)
    claim = _claim()
    binding = _PreparedControlBinding("drive", (0,))

    block = PulseBlock(1.0, (control,), (binding,), (claim,))

    assert not hasattr(block, "model")
    assert block.control_bindings == (binding,)
    assert block.resource_claims == (claim,)


def test_block_requires_one_prevalidated_binding_per_control(model):
    with pytest.raises(BackendValidationError, match="one control binding"):
        PulseBlock(1.0, (_control(model),), (), (_claim(),))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"condition": ()},
        {"start_time": -1.0},
        {"target_indices": (0, 0)},
    ],
)
def test_block_still_validates_occurrence_only_fields(model, kwargs):
    with pytest.raises(BackendValidationError):
        PulseBlock(0.0, (), (), (_claim(),), **kwargs)
