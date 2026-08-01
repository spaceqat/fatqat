"""Validation tests for the engine-neutral resolved pulse values."""

import json
from pathlib import Path

import numpy as np
import pytest

from fatqat.emulator.pulse import (
    PhaseShift,
    PulseBlock,
    PulseDefinition,
    SampledControl,
)
from fatqat.emulator.superconducting import load_physics_model
from fatqat.errors import BackendValidationError

_FIXTURES = Path(__file__).parent / "fixtures"


def _model():
    return load_physics_model(
        json.loads((_FIXTURES / "sc_transmon_exchange.json").read_text())
    )


def _control(model, *, tlist=(0.0, 1.0), offset=0.0):
    return SampledControl(
        model.drive_control("q0"), np.array(tlist), np.array([0.0, 0.0]), offset
    )


def test_pulse_block_rejects_invalid_timing_and_same_channel_summation():
    model = _model()
    with pytest.raises(BackendValidationError, match="strictly increasing"):
        _control(model, tlist=(0.0, 0.0))
    with pytest.raises(BackendValidationError, match="extends"):
        PulseBlock(model, 1.0, (_control(model, offset=0.5),), (model.resource("q0"),))
    with pytest.raises(BackendValidationError, match="implicitly sum"):
        PulseBlock(
            model,
            1.0,
            (_control(model), _control(model)),
            (model.resource("q0"),),
        )
    with pytest.raises(BackendValidationError, match="zero-duration"):
        PulseBlock(model, 0.0, (_control(model),), (model.resource("q0"),))
    with pytest.raises(BackendValidationError, match="do not cover"):
        PulseBlock(model, 1.0, (_control(model),), (model.resource("q1"),))
    with pytest.raises(BackendValidationError, match="do not cover"):
        PulseBlock(
            model,
            1.0,
            (
                SampledControl(
                    model.exchange_control("q0", "q1"), [0.0, 1.0], [0.0, 0.0]
                ),
            ),
            (model.resource("q0"),),
        )
    with pytest.raises(BackendValidationError, match="unknown channel reference"):
        PulseBlock(
            model,
            1.0,
            (SampledControl(model.coupling("q0", "q1"), [0.0, 1.0], [0.0, 0.0]),),
            (model.resource("q0"), model.resource("q1"), model.coupling("q0", "q1")),
        )


def test_exchange_child_uses_only_the_exchange_control_not_the_pair_resource():
    model = _model()
    block = PulseBlock(
        model,
        1.0,
        (SampledControl(model.exchange_control("q0", "q1"), [0.0, 1.0], [0.0, 0.0]),),
        (model.resource("q0"), model.resource("q1"), model.coupling("q0", "q1")),
    )
    (exchange,) = block.controls
    assert exchange.channel == model.exchange_control("q0", "q1")
    assert exchange.channel.kind == "exchange"
    assert model.coupling("q0", "q1") in block.resource_claims
    assert model.coupling("q0", "q1") not in {child.channel for child in block.controls}


def test_exchange_child_requires_the_pair_resource_claim_not_just_endpoints():
    model = _model()
    with pytest.raises(BackendValidationError, match="do not cover"):
        PulseBlock(
            model,
            1.0,
            (
                SampledControl(
                    model.exchange_control("q0", "q1"), [0.0, 1.0], [0.0, 0.0]
                ),
            ),
            (model.resource("q0"), model.resource("q1")),
        )


def test_pulse_block_rejects_cross_model_handles_and_keeps_arrays_immutable():
    model = _model()
    other = _model()
    with pytest.raises(BackendValidationError, match="foreign"):
        PulseBlock(
            model,
            1.0,
            (SampledControl(other.drive_control("q0"), [0.0, 1.0], [0.0, 0.0]),),
            (model.resource("q0"),),
        )

    block = PulseBlock(
        model,
        1.0,
        (_control(model),),
        (model.resource("q0"),),
        (PhaseShift(model.frame("q0"), 0.2),),
    )
    assert not block.controls[0].tlist.flags.writeable
    assert not block.controls[0].coefficients.flags.writeable


# --- PulseDefinition: model-independent structural validation ---------------
#
# PulseDefinition shares its structural checks with PulseBlock through the
# module-level `_validate_*` helpers, so every model-independent rejection
# below mirrors a PulseBlock rejection above. A PulseDefinition carries no
# PhysicsModel, so it cannot detect a foreign (cross-model) handle - that
# check happens only at PulseBlock construction, the conversion boundary
# that owns a model.


def test_pulse_definition_rejects_invalid_timing_and_same_channel_summation():
    model = _model()
    with pytest.raises(BackendValidationError, match="extends"):
        PulseDefinition(1.0, (_control(model, offset=0.5),), (model.resource("q0"),))
    with pytest.raises(BackendValidationError, match="implicitly sum"):
        PulseDefinition(
            1.0,
            (_control(model), _control(model)),
            (model.resource("q0"),),
        )
    with pytest.raises(BackendValidationError, match="zero-duration"):
        PulseDefinition(0.0, (_control(model),), (model.resource("q0"),))
    with pytest.raises(BackendValidationError, match="unknown channel reference"):
        PulseDefinition(
            1.0,
            (SampledControl(model.coupling("q0", "q1"), [0.0, 1.0], [0.0, 0.0]),),
            (model.resource("q0"), model.resource("q1"), model.coupling("q0", "q1")),
        )


def test_pulse_definition_requires_at_least_one_resource_claim():
    with pytest.raises(BackendValidationError, match="at least one model resource"):
        PulseDefinition(0.0, (), ())


def test_pulse_definition_rejects_duplicate_resource_claim():
    model = _model()
    with pytest.raises(BackendValidationError, match="duplicate resource claim"):
        PulseDefinition(0.0, (), (model.resource("q0"), model.resource("q0")))


def test_pulse_definition_is_immutable_and_copy_owned():
    model = _model()
    definition = PulseDefinition(1.0, (_control(model),), (model.resource("q0"),))
    assert not definition.controls[0].tlist.flags.writeable
    assert not definition.controls[0].coefficients.flags.writeable
    with pytest.raises(AttributeError):
        definition.duration = 2.0


def test_pulse_definition_accepts_a_foreign_handle_only_a_model_bound_check_can_reject():
    model = _model()
    other = _model()
    definition = PulseDefinition(
        1.0,
        (SampledControl(other.drive_control("q0"), [0.0, 1.0], [0.0, 0.0]),),
        (model.resource("q0"),),
    )
    assert definition.controls[0].channel == other.drive_control("q0")

    with pytest.raises(BackendValidationError, match="foreign"):
        PulseBlock(
            model,
            definition.duration,
            definition.controls,
            definition.resource_claims,
            definition.post_actions,
        )
