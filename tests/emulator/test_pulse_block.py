"""Validation tests for the engine-neutral resolved pulse values.

`PulseDefinition` and `PulseBlock` are independent constructors that must both
call the shared `_validate_*` helpers, so the model-independent rejections are
parametrized over both: testing one alone would not catch the other quietly
dropping a validation call. The model-bound checks (foreign handles, claim
coverage, envelope realizability) are `PulseBlock`-only, because a definition
carries no model to ask.
"""

import numpy as np
import pytest

from fatqat.emulator.pulse import (
    PhaseShift,
    PulseBlock,
    PulseDefinition,
    SampledControl,
)
from fatqat.errors import BackendValidationError


def _control(model, *, tlist=(0.0, 1.0), offset=0.0):
    return SampledControl(
        model.drive_control("q0"), np.array(tlist), np.array([0.0, 0.0]), offset
    )


def _as_definition(model, duration, controls, claims, post_actions=()):
    return PulseDefinition(duration, controls, claims, post_actions)


def _as_block(model, duration, controls, claims, post_actions=()):
    return PulseBlock(model, duration, controls, claims, post_actions)


_CONSTRUCTORS = pytest.mark.parametrize(
    "construct", (_as_definition, _as_block), ids=("definition", "block")
)


# --- model-independent structural checks, shared by both constructors -------


@_CONSTRUCTORS
@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("control_overruns", "extends"),
        ("same_channel_twice", "implicitly sum"),
        ("controls_on_zero_duration", "zero-duration"),
        ("non_channel_as_channel", "unknown channel reference"),
        ("no_resource_claims", "at least one model resource"),
        ("duplicate_resource_claim", "duplicate resource claim"),
    ),
)
def test_both_pulse_values_apply_the_shared_structural_validators(
    model, construct, case, match
):
    claims = (model.resource("q0"),)
    arguments = {
        "control_overruns": (1.0, (_control(model, offset=0.5),), claims),
        "same_channel_twice": (1.0, (_control(model), _control(model)), claims),
        "controls_on_zero_duration": (0.0, (_control(model),), claims),
        "non_channel_as_channel": (
            1.0,
            (SampledControl(model.coupling("q0", "q1"), [0.0, 1.0], [0.0, 0.0]),),
            (
                model.resource("q0"),
                model.resource("q1"),
                model.coupling("q0", "q1"),
            ),
        ),
        "no_resource_claims": (0.0, (), ()),
        "duplicate_resource_claim": (
            0.0,
            (),
            (model.resource("q0"), model.resource("q0")),
        ),
    }[case]

    with pytest.raises(BackendValidationError, match=match):
        construct(model, *arguments)


@_CONSTRUCTORS
def test_both_pulse_values_copy_and_freeze_control_arrays(model, construct):
    value = construct(model, 1.0, (_control(model),), (model.resource("q0"),))
    assert not value.controls[0].tlist.flags.writeable
    assert not value.controls[0].coefficients.flags.writeable
    with pytest.raises(AttributeError):
        value.duration = 2.0


def test_sampled_control_rejects_a_non_increasing_grid(model):
    with pytest.raises(BackendValidationError, match="strictly increasing"):
        _control(model, tlist=(0.0, 0.0))


# --- model-bound checks: PulseBlock only ------------------------------------


def test_pulse_block_rejects_claims_that_do_not_cover_a_driven_control(model):
    with pytest.raises(BackendValidationError, match="do not cover"):
        PulseBlock(model, 1.0, (_control(model),), (model.resource("q1"),))


def test_exchange_control_requires_the_pair_resource_not_just_endpoints(model):
    exchange = SampledControl(
        model.exchange_control("q0", "q1"), [0.0, 1.0], [0.0, 0.0]
    )
    with pytest.raises(BackendValidationError, match="do not cover"):
        PulseBlock(
            model, 1.0, (exchange,), (model.resource("q0"), model.resource("q1"))
        )

    block = PulseBlock(
        model,
        1.0,
        (exchange,),
        (model.resource("q0"), model.resource("q1"), model.coupling("q0", "q1")),
    )
    # The pair resource is a scheduling claim, never a driven channel.
    assert model.coupling("q0", "q1") in block.resource_claims
    assert model.coupling("q0", "q1") not in {child.channel for child in block.controls}


def test_pulse_block_rejects_a_complex_envelope_on_a_real_only_channel(model):
    """Realness is the model's rule, enforced when a block binds to it.

    A detuning channel drives a Hermitian generator directly, so a complex
    envelope is unrealizable. Rejecting it here means `run()` and
    `propagator()` report the identical error instead of one of them failing
    later during solver binding.
    """
    complex_detuning = SampledControl(
        model.detuning_control("q0"), [0.0, 1.0], [0.1 + 0.2j, 0.1 + 0.2j]
    )
    with pytest.raises(BackendValidationError, match="detuning.*must be real"):
        PulseBlock(model, 1.0, (complex_detuning,), (model.resource("q0"),))


def test_drive_channel_still_accepts_a_complex_two_quadrature_envelope(model):
    drive = SampledControl(
        model.drive_control("q0"), [0.0, 1.0], [0.1 + 0.2j, 0.3 - 0.4j]
    )
    block = PulseBlock(model, 1.0, (drive,), (model.resource("q0"),))
    assert np.allclose(block.controls[0].coefficients.imag, (0.2, -0.4))


def test_only_the_model_bound_constructor_can_reject_a_foreign_handle(
    model, build_model_and_calibration
):
    other, _ = build_model_and_calibration()
    foreign = SampledControl(other.drive_control("q0"), [0.0, 1.0], [0.0, 0.0])

    # A definition carries no model, so it cannot detect the foreign handle.
    definition = PulseDefinition(1.0, (foreign,), (model.resource("q0"),))
    assert definition.controls[0].channel == other.drive_control("q0")

    with pytest.raises(BackendValidationError, match="foreign"):
        PulseBlock(
            model,
            definition.duration,
            definition.controls,
            definition.resource_claims,
            definition.post_actions,
        )


def test_pulse_block_binds_post_action_frames(model):
    block = PulseBlock(
        model,
        1.0,
        (_control(model),),
        (model.resource("q0"),),
        (PhaseShift(model.frame("q0"), 0.2),),
    )
    assert block.post_actions[0].frame == model.frame("q0")
