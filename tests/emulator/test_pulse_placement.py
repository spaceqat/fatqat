"""Conservative private placement tests."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fatqat.emulator.execution import place_pulse_run
from fatqat.emulator.resolved import PulseBlock, SampledControl
from fatqat.emulator.superconducting import load_physics_model
from fatqat.errors import BackendValidationError

_FIXTURES = Path(__file__).parent / "fixtures"


def _model():
    return load_physics_model(
        json.loads((_FIXTURES / "sc_transmon_exchange.json").read_text())
    )


def _block(model, subsystem_id, duration):
    return PulseBlock(
        model,
        duration,
        (
            SampledControl(
                model.drive_control(subsystem_id),
                np.array([0.0, duration]),
                np.zeros(2),
            ),
        ),
        (model.resource(subsystem_id),),
    )


def test_asap_and_alap_share_makespan_but_place_independent_work_differently():
    model = _model()
    blocks = (
        _block(model, "q0", 2.0),
        _block(model, "q1", 1.0),
        _block(model, "q0", 1.0),
    )

    asap = place_pulse_run(blocks, boundary_ns=5.0, mode="ASAP")
    alap = place_pulse_run(blocks, boundary_ns=5.0, mode="ALAP")

    assert asap.starts_ns == (5.0, 5.0, 7.0)
    assert alap.starts_ns == (5.0, 7.0, 7.0)
    assert asap.end_ns == alap.end_ns == 8.0
    assert all(actual is source for actual, source in zip(asap.blocks, blocks))


def test_pair_claims_conservatively_conflict_with_endpoint_work():
    model = _model()
    pair = PulseBlock(
        model,
        2.0,
        (SampledControl(model.exchange_control("q0", "q1"), [0.0, 2.0], [0.0, 0.0]),),
        (
            model.resource("q0"),
            model.resource("q1"),
            model.coupling("q0", "q1"),
        ),
    )
    q0 = _block(model, "q0", 1.0)
    run = place_pulse_run((pair, q0), boundary_ns=0.0)
    assert run.starts_ns == (0.0, 2.0)


def test_explicit_placement_rejects_mixed_reverse_and_preboundary_starts():
    model = _model()
    q0 = _block(model, "q0", 1.0)
    with pytest.raises(BackendValidationError, match="either all explicit"):
        place_pulse_run((replace(q0, start_ns=0.0), q0), boundary_ns=0.0)
    with pytest.raises(BackendValidationError, match="reverses source order"):
        place_pulse_run(
            (replace(q0, start_ns=10.0), replace(q0, start_ns=0.0)),
            boundary_ns=0.0,
        )
    with pytest.raises(BackendValidationError, match="current execution boundary"):
        place_pulse_run((replace(q0, start_ns=1.0),), boundary_ns=2.0)

    adjacent = place_pulse_run(
        (replace(q0, start_ns=1.0), replace(q0, start_ns=2.0)),
        boundary_ns=1.0,
    )
    assert adjacent.starts_ns == (1.0, 2.0)
