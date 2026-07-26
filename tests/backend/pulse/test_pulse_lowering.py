"""Unplaced pulse lowering and shared-boundary preservation."""

import json
from pathlib import Path

import pytest

import fatqat as fq
from fatqat.backends import MeasurementStep, ResetStep
from fatqat.backends.pulse.backend import PulseBackend
from fatqat.backends.pulse.resolved import PulseBlock
from fatqat.backends.pulse.superconducting import (
    load_calibration_spec,
    load_physics_model,
)
from fatqat.errors import BackendValidationError

_FIXTURES = Path(__file__).parent / "fixtures"


def _backend():
    model = load_physics_model(
        json.loads((_FIXTURES / "sc_transmon_exchange.json").read_text())
    )
    calibration = load_calibration_spec(
        json.loads((_FIXTURES / "sc_transmon_exchange_calibration.json").read_text()),
        model,
    )
    return PulseBackend(model, calibration)


def test_lowering_produces_unplaced_blocks_and_preserves_boundaries_and_guards():
    backend = _backend()
    program = fq.Program(2, 1)
    program.add(fq.ops.RX(0.4), 0)
    program.add_measurement(0, 0)
    program.add(fq.ops.RZ(0.2), 1, condition=(0, 0))
    program.add(fq.ops.Reset, 1, condition=(0, 0))
    plan, facts = backend._lower_program(program)

    assert [type(step) for step in plan] == [
        PulseBlock,
        MeasurementStep,
        PulseBlock,
        ResetStep,
    ]
    assert plan[0].start_ns is None
    assert plan[2].condition == ((0, 0),)
    assert plan[3].condition == ((0, 0),)
    assert facts.has_measurement and facts.has_reset and facts.has_guarded_pulse


def test_lowering_rejects_absent_edges_and_reversed_cz_orientation():
    backend = _backend()
    disconnected_document = json.loads(
        (_FIXTURES / "sc_transmon_exchange.json").read_text()
    )
    disconnected_document["parameters"]["couplings"] = []
    disconnected = load_physics_model(disconnected_document)
    calibration_document = json.loads(
        (_FIXTURES / "sc_transmon_exchange_calibration.json").read_text()
    )
    calibration_document["recipes"]["cz"]["edges"] = []
    disconnected_backend = PulseBackend(
        disconnected, load_calibration_spec(calibration_document, disconnected)
    )
    iswap = fq.Program(2)
    iswap.add(fq.ops.iSwap, (0, 1))
    with pytest.raises(BackendValidationError, match="no declared coupling"):
        disconnected_backend.run(iswap)

    reversed_cz = fq.Program(2)
    reversed_cz.add(fq.ops.CZ, (1, 0))
    with pytest.raises(BackendValidationError, match="orientation"):
        backend.run(reversed_cz)
