"""Private placement must preserve pulse-block identity and boundaries."""

import json
from pathlib import Path

import pytest

import fatqat as fq
from fatqat.backends.pulse.backend import PulseBackend
from fatqat.backends.pulse.execution import execute_with_boundaries
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


def _plan(*operations):
    program = fq.Program(2, 1)
    for operation, target, *condition in operations:
        program.add(
            operation, target, **({"condition": condition[0]} if condition else {})
        )
    return _backend()._lower_program(program)[0]


def test_asap_placement_preserves_identity_order_and_serializes_claims():
    plan = _plan((fq.ops.RX(0.2), 0), (fq.ops.RY(0.3), 0), (fq.ops.RX(0.4), 1))
    seen = []
    _, end = execute_with_boundaries(plan, seen.append, lambda step, time: None)

    assert len(seen) == 1
    blocks = seen[0].blocks
    assert blocks[0] is not plan[0]
    assert [block.duration_ns for block in blocks] == [20.0, 20.0, 20.0]
    assert [block.start_ns for block in blocks] == [0.0, 20.0, 0.0]
    assert end == 40.0


def test_boundaries_and_guarded_blocks_flush_continuous_runs():
    plan = _plan(
        (fq.ops.RX(0.2), 0),
        (fq.ops.RZ(0.1), 1, (0, 0)),
        (fq.ops.RY(0.3), 0),
    )
    seen, boundaries = [], []
    _, end = execute_with_boundaries(
        plan, seen.append, lambda step, time: boundaries.append((step, time))
    )

    assert len(seen) == 3
    assert [len(run.blocks) for run in seen] == [1, 1, 1]
    assert boundaries == []
    assert [run.start_ns for run in seen] == [0.0, 20.0, 20.0]
    assert end == 40.0


def test_mixed_or_conflicting_explicit_placement_fails():
    plan = _plan((fq.ops.RX(0.2), 0), (fq.ops.RY(0.3), 0))
    from dataclasses import replace

    with pytest.raises(BackendValidationError, match="all explicit starts"):
        execute_with_boundaries(
            (replace(plan[0], start_ns=0.0), plan[1]), lambda run: None, lambda *_: None
        )
    with pytest.raises(BackendValidationError, match="conflicting resource claims"):
        execute_with_boundaries(
            (replace(plan[0], start_ns=0.0), replace(plan[1], start_ns=0.0)),
            lambda run: None,
            lambda *_: None,
        )
