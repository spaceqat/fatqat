"""Native SC operation realization checks without a solver dependency."""

import dataclasses
from math import pi, sqrt

import numpy as np
import pytest

from fatqat import ops
from fatqat.emulator.pulse import (
    PhaseShift,
    PhaseSwap,
)
from fatqat.emulator.superconducting_realization import (
    default_superconducting_pulse_implementation_map,
)
from fatqat.emulator.superconducting import ControlChannelRef
from fatqat.errors import BackendValidationError


def _resolve(operation, targets, *, model, calibration):
    """Resolve one operation through the default map, exactly as lowering does."""
    rule = default_superconducting_pulse_implementation_map().implementation_for(
        operation
    )
    return rule(operation, targets=targets, model=model, calibration=calibration)


def test_rx_ry_are_hann_drag_complex_drives_without_physical_z_control(
    model, calibration
):
    theta = pi / 2
    rx = _resolve(
        ops.RX(theta), (model.resource("q0"),), model=model, calibration=calibration
    )
    ry = _resolve(
        ops.RY(theta), (model.resource("q0"),), model=model, calibration=calibration
    )

    assert rx.duration == 20.0
    assert len(rx.controls) == 1
    assert isinstance(rx.controls[0].channel, ControlChannelRef)
    assert rx.controls[0].channel.kind == "drive"
    assert np.isclose(rx.controls[0].tlist[0], 0.0)
    assert np.isclose(rx.controls[0].tlist[-1], rx.duration)
    assert np.isclose(rx.controls[0].coefficients[0], 0.0)
    assert np.isclose(rx.controls[0].coefficients[-1], 0.0)
    assert np.allclose(ry.controls[0].coefficients, 1j * rx.controls[0].coefficients)
    assert isinstance(rx.post_actions[0], PhaseShift)


def test_rz_is_zero_duration_frame_only_and_preserves_angle_ordering(
    model, calibration
):
    definition = _resolve(
        ops.RZ(0.7), (model.resource("q1"),), model=model, calibration=calibration
    )

    assert definition.duration == 0.0
    assert definition.controls == ()
    assert definition.resource_claims == (model.resource("q1"),)
    assert definition.post_actions == (PhaseShift(model.frame("q1"), 0.7),)


def test_iswap_area_and_frame_swap_use_one_full_edge_control(model, calibration):
    definition = _resolve(
        ops.iSwap,
        (model.resource("q0"), model.resource("q1")),
        model=model,
        calibration=calibration,
    )

    (exchange,) = definition.controls
    assert exchange.channel == model.exchange_control("q0", "q1")
    assert exchange.channel.kind == "exchange"
    assert definition.resource_claims == (
        model.resource("q0"),
        model.resource("q1"),
        model.coupling("q0", "q1"),
    )
    assert np.isclose(np.trapezoid(exchange.coefficients.real, exchange.tlist), -pi / 2)
    assert definition.post_actions == (PhaseSwap(model.frame("q0"), model.frame("q1")),)


def test_cz_is_atomic_oriented_detuning_plus_parked_exchange(model, calibration):
    definition = _resolve(
        ops.CZ,
        (model.resource("q0"), model.resource("q1")),
        model=model,
        calibration=calibration,
    )

    detuning, exchange = definition.controls
    assert detuning.channel == model.detuning_control("q0")
    assert exchange.channel == model.exchange_control("q0", "q1")
    assert definition.resource_claims == (
        model.resource("q0"),
        model.resource("q1"),
        model.coupling("q0", "q1"),
    )
    assert exchange.start_offset == 3.0
    assert exchange.duration == 54.0
    assert np.isclose(detuning.coefficients[0], 0.0)
    assert np.isclose(detuning.coefficients[-1], 0.0)
    assert np.isclose(
        np.trapezoid(exchange.coefficients.real, exchange.tlist), pi / sqrt(2)
    )
    detuning_phase = float(np.trapezoid(detuning.coefficients.real, detuning.tlist))
    assert definition.post_actions == (PhaseShift(model.frame("q0"), detuning_phase),)
    with pytest.raises(BackendValidationError, match="orientation"):
        _resolve(
            ops.CZ,
            (model.resource("q1"), model.resource("q0")),
            model=model,
            calibration=calibration,
        )


def test_cz_missing_edge_recipe_names_the_declared_edge(model, calibration):
    edgeless_calibration = dataclasses.replace(
        calibration,
        recipes={**calibration.recipes, "cz": {"edges": []}},
    )

    with pytest.raises(BackendValidationError, match=r"no CZ recipe.*'q0'-'q1'"):
        _resolve(
            ops.CZ,
            (model.resource("q0"), model.resource("q1")),
            model=model,
            calibration=edgeless_calibration,
        )
