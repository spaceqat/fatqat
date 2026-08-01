"""Conservative private placement tests."""

from dataclasses import replace

import numpy as np
import pytest

from fatqat.emulator.scheduling import schedule_pulse_run
from fatqat.emulator.pulse import PulseBlock, SampledControl
from fatqat.errors import BackendValidationError


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


def test_asap_and_alap_share_makespan_but_place_independent_work_differently(model):
    blocks = (
        _block(model, "q0", 2.0),
        _block(model, "q1", 1.0),
        _block(model, "q0", 1.0),
    )

    asap = schedule_pulse_run(blocks, boundary_time=5.0, mode="ASAP")
    alap = schedule_pulse_run(blocks, boundary_time=5.0, mode="ALAP")

    assert asap.starts == (5.0, 5.0, 7.0)
    assert alap.starts == (5.0, 7.0, 7.0)
    assert asap.end_time == alap.end_time == 8.0
    assert all(actual is source for actual, source in zip(asap.blocks, blocks))


def test_pair_claims_conservatively_conflict_with_endpoint_work(model):
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
    run = schedule_pulse_run((pair, q0), boundary_time=0.0)
    assert run.starts == (0.0, 2.0)


def test_explicit_placement_rejects_mixed_reverse_and_preboundary_starts(model):
    q0 = _block(model, "q0", 1.0)
    with pytest.raises(BackendValidationError, match="either all explicit"):
        schedule_pulse_run((replace(q0, start_time=0.0), q0), boundary_time=0.0)
    with pytest.raises(BackendValidationError, match="reverses source order"):
        schedule_pulse_run(
            (replace(q0, start_time=10.0), replace(q0, start_time=0.0)),
            boundary_time=0.0,
        )
    with pytest.raises(BackendValidationError, match="current execution boundary"):
        schedule_pulse_run((replace(q0, start_time=1.0),), boundary_time=2.0)

    adjacent = schedule_pulse_run(
        (replace(q0, start_time=1.0), replace(q0, start_time=2.0)),
        boundary_time=1.0,
    )
    assert adjacent.starts == (1.0, 2.0)
