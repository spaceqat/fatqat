"""Conservative private placement tests."""

from dataclasses import replace

import pytest

from fatqat.emulator._core.scheduling import schedule_pulse_run
from fatqat.emulator._core.pulse import PulseBlock
from fatqat.emulator._core.target import _PreparedControlBinding, _TargetClaim
from fatqat._pulse_values import PulseControl
from fatqat.errors import BackendValidationError
from fatqat.emulator import SampledWaveform

_OWNER = object()
_SUBSYSTEM_CLAIMS = {
    "q0": _TargetClaim(_OWNER, "subsystem", 0),
    "q1": _TargetClaim(_OWNER, "subsystem", 1),
}
_COUPLING_CLAIM = _TargetClaim(_OWNER, "coupling", 0)


def _block(model, subsystem_id, duration):
    controls = (
        PulseControl(
            model.control.drive(subsystem_id),
            SampledWaveform((0.0, duration), (0.0, 0.0)),
        ),
    )
    claim = _SUBSYSTEM_CLAIMS[subsystem_id]
    binding = _PreparedControlBinding("drive", (0 if subsystem_id == "q0" else 1,))
    return PulseBlock(duration, controls, (binding,), (claim,))


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
    controls = (
        PulseControl(
            model.control.exchange("q0", "q1"),
            SampledWaveform((0.0, 2.0), (0.0, 0.0)),
        ),
    )
    claims = (
        _SUBSYSTEM_CLAIMS["q0"],
        _SUBSYSTEM_CLAIMS["q1"],
        _COUPLING_CLAIM,
    )
    binding = _PreparedControlBinding("exchange", (0, 1))
    pair = PulseBlock(2.0, controls, (binding,), claims)
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
