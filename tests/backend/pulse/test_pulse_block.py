"""Validation tests for the engine-neutral resolved pulse values."""

import json
from pathlib import Path

import numpy as np
import pytest

from fatqat.backends.pulse.resolved import PhaseShift, PulseBlock, SampledControl
from fatqat.backends.pulse.superconducting import load_physics_model
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
            (SampledControl(model.coupling("q0", "q1"), [0.0, 1.0], [0.0, 0.0]),),
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
    assert not block.children[0].tlist.flags.writeable
    assert not block.children[0].coefficients.flags.writeable
