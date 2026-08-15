"""Structural target-vocabulary tests."""

from dataclasses import FrozenInstanceError

import pytest

from fatqat.emulator._core.target import (
    Frame,
    ResourceClaim,
    _ControlAddress,
    _ControlBinding,
    _FrameAddress,
    _GateBinding,
    _TargetClaim,
)
from fatqat.emulator._core import Frame as ExportedFrame
from fatqat.emulator._core import ResourceClaim as ExportedResourceClaim


def test_control_and_frame_addresses_are_frozen_structural_values():
    control = _ControlAddress("transmon", "drive", ("q0", 2))
    global_control = _ControlAddress("atom-2level", "drive")
    frame = _FrameAddress("transmon", ("q0", 2))

    assert control.operands == ("q0", 2)
    assert global_control.operands == ()
    assert frame.operands == ("q0", 2)
    assert control != _ControlAddress("atom-3level", "drive", ("q0", 2))
    with pytest.raises(FrozenInstanceError):
        control.kind = "detuning"


def test_global_control_names_every_affected_target():
    claims = (
        _TargetClaim(object(), "site", 0),
        _TargetClaim(object(), "site", 1),
    )
    binding = _ControlBinding("drive", (0, 1), claims)

    assert binding.device_operands == (0, 1)


def test_control_binding_can_allow_target_owned_additional_claims():
    owner = object()
    claim = _TargetClaim(owner, "site", 3)
    binding = _ControlBinding(
        "exchange", ("q3",), (claim,), allows_additional_claims=True
    )

    assert binding.allows_additional_claims
    assert binding.device_operands == ("q3",)


def test_claim_ownership_is_target_local_without_identity_checks():
    first_owner = object()
    second_owner = object()
    first = _TargetClaim(first_owner, "site", 0)
    same = _TargetClaim(first_owner, "site", 0)
    foreign = _TargetClaim(second_owner, "site", 0)

    assert first.owner is first_owner
    assert first == same
    assert first is not same
    assert first != foreign
    assert {first, same, foreign} == {first, foreign}


def test_resolved_bindings_copy_mutable_aggregate_inputs():
    owner = object()
    claim = _TargetClaim(owner, "site", 0)
    claims = [claim]
    operands = ["q0"]

    control = _ControlBinding("drive", operands, claims)
    gate = _GateBinding(claims, operands)
    claims.clear()
    operands.append("q1")

    assert control.device_operands == ("q0",)
    assert control.claims == (claim,)
    assert gate.claims == (claim,)
    assert gate.device_operands == ("q0",)
    assert hash(control) == hash(control)
    assert hash(gate) == hash(gate)


def test_marker_exports_are_the_exact_target_types():
    assert Frame is ExportedFrame
    assert ResourceClaim is ExportedResourceClaim
