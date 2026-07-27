"""Native SC operation realization checks without a solver dependency."""

import json
from math import pi, sqrt
from pathlib import Path

import numpy as np
import pytest

from fatqat import ops
from fatqat.emulator.resolved import (
    PhaseShift,
    PhaseSwap,
    realize_native_operation,
)
from fatqat.emulator.superconducting import (
    ControlChannelRef,
    load_calibration_spec,
    load_physics_model,
)
from fatqat.errors import BackendValidationError

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


def test_rx_ry_are_hann_drag_complex_drives_without_physical_z_control():
    model, calibration = _model_and_calibration()
    theta = pi / 2
    rx = realize_native_operation(
        ops.RX(theta), (model.resource("q0"),), model=model, calibration=calibration
    )
    ry = realize_native_operation(
        ops.RY(theta), (model.resource("q0"),), model=model, calibration=calibration
    )

    assert rx.start_ns is None
    assert rx.duration_ns == 20.0
    assert len(rx.children) == 1
    assert isinstance(rx.children[0].channel, ControlChannelRef)
    assert rx.children[0].channel.kind == "drive"
    assert np.isclose(rx.children[0].tlist[0], 0.0)
    assert np.isclose(rx.children[0].tlist[-1], rx.duration_ns)
    assert np.isclose(rx.children[0].coefficients[0], 0.0)
    assert np.isclose(rx.children[0].coefficients[-1], 0.0)
    assert np.allclose(ry.children[0].coefficients, 1j * rx.children[0].coefficients)
    assert isinstance(rx.post_actions[0], PhaseShift)


def test_rz_is_zero_duration_frame_only_and_preserves_angle_ordering():
    model, calibration = _model_and_calibration()
    block = realize_native_operation(
        ops.RZ(0.7), (model.resource("q1"),), model=model, calibration=calibration
    )

    assert block.duration_ns == 0.0
    assert block.children == ()
    assert block.resource_claims == (model.resource("q1"),)
    assert block.post_actions == (PhaseShift(model.frame("q1"), 0.7),)


def test_iswap_area_and_frame_swap_use_one_full_edge_control():
    model, calibration = _model_and_calibration()
    block = realize_native_operation(
        ops.iSwap,
        (model.resource("q0"), model.resource("q1")),
        model=model,
        calibration=calibration,
    )

    (exchange,) = block.children
    assert exchange.channel == model.exchange_control("q0", "q1")
    assert exchange.channel.kind == "exchange"
    assert block.resource_claims == (
        model.resource("q0"),
        model.resource("q1"),
        model.coupling("q0", "q1"),
    )
    assert np.isclose(np.trapezoid(exchange.coefficients.real, exchange.tlist), -pi / 2)
    assert block.post_actions == (PhaseSwap(model.frame("q0"), model.frame("q1")),)


def test_cz_is_atomic_oriented_detuning_plus_parked_exchange():
    model, calibration = _model_and_calibration()
    block = realize_native_operation(
        ops.CZ,
        (model.resource("q0"), model.resource("q1")),
        model=model,
        calibration=calibration,
    )

    detuning, exchange = block.children
    assert detuning.channel == model.detuning_control("q0")
    assert exchange.channel == model.exchange_control("q0", "q1")
    assert block.resource_claims == (
        model.resource("q0"),
        model.resource("q1"),
        model.coupling("q0", "q1"),
    )
    assert exchange.start_offset_ns == 3.0
    assert exchange.duration_ns == 54.0
    assert np.isclose(detuning.coefficients[0], 0.0)
    assert np.isclose(detuning.coefficients[-1], 0.0)
    assert np.isclose(
        np.trapezoid(exchange.coefficients.real, exchange.tlist), pi / sqrt(2)
    )
    assert all(isinstance(action, PhaseShift) for action in block.post_actions)
    with pytest.raises(BackendValidationError, match="orientation"):
        realize_native_operation(
            ops.CZ,
            (model.resource("q1"), model.resource("q0")),
            model=model,
            calibration=calibration,
        )
